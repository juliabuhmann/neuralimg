#!/usr/bin/python

import os
import shutil
import tempfile
import h5py
import mahotas as mh

from neuralimg.crag import crag_costs as cc
from neuralimg.crag import merge_mc as mt
from neuralimg.base.pyprocess import *
from neuralimg import dataio
from neuralimg.evaluation import voi
from neuralimg.evaluation import rand


class LossType(object):

    HAUSDORFF = 'hausdorff'
    RAND = 'rand'
    OVERLAP = 'overlap'
    HEURISTIC = 'heuristic'
    ASSIGNMENT = 'assignment'

class CragGenerator(object):

    def __init__(self, out_p):
        self.project_file = None
        self.best_effort = None
        self.gt = None
        self.sps = None
        self.raws = None
        self.membranes = None
        self.histories = None
        self.solution = None
        # Prepare project structure
        if not os.path.exists(out_p):
            os.makedirs(out_p)
        self.output = out_p

        self.logs = os.path.join(out_p, 'log')
        if not os.path.exists(self.logs):
            os.mkdir(self.logs)

    def generate_crag(self, groundtruth, superpixels, raws, membranes, config_file, 
        max_zlink=200, histories=None, histories_thresh=None, overwrite=False, indexes=None):
        """ Class for generating Candidate Region Adjacency Graphs.
        Params
        ---------
        groundtruth: string
            Path to folder containing groundtruth images
        superpixels: string
            Path containing supervoxel images
        raws: string
            Path containing raw images
        membranes: string
            Path containing neuron boundaries
        config_file: string
            Configuration file path for the CRAG creation
        max_zlink: integer
            Maximum Hausdorff distance for crag candidates according to image 
            resolution (in config file) 
            (e.g. if resX and resY resolution of 4, a value of 200 is translated into 50 pixels)
        histories: string
            Path containing history for each of the sections in the volume.
            If not provided, they will be computed using the candidate_mc tool
        histories_thresh: float
            Value to use to cut the merge histories. Only values above the given threshold
            will be kept. To disable this option, set to None
        overwrite: boolean
            Whether to overwrite a past project
        indexes: list or ndarray
            List containing first and last index to consider from the superpixels folder, considering the list
            of ordered names. To use all, set to None
        """
        if not os.path.isfile(config_file):
            raise ValueError('A config file must be provided')

        self.gt = groundtruth
        self.sps = superpixels
        self.raws = raws
        self.membranes = membranes
        self.histories = histories
        self.hist_thresh = histories_thresh
        self._check_paths()

        self._generate_project(config_file, max_zlink, overwrite, indexes)

    def _check_paths(self):
        """ Checks if all needed paths exist """
        dataio.check_folder(self.gt)
        dataio.check_folder(self.sps)
        dataio.check_folder(self.raws)
        dataio.check_folder(self.membranes)

    def read_crag(self, crag):
        """ Reads an existing CRAG from its project file
        Params
        ---------
        crag: crag
            Path to the CRAG project file
        """
        if not dataio.valid_volume_path(crag):
            raise ValueError('A valid HDF5 file must be provided')

        self.project_file = crag

    def _generate_project(self, config_file, zlink, overwrite, inds):
        """ Generates the HDF of the project """
        # If histories were not provided, use them
        if self.histories is None:
            tmpath = tempfile.mkdtemp()
            # TODO: generate only the ones corresponding to the selected indices
            self._generate_merge_histories(tmpath)
            print('Merge trees generated!')

        # Cut histories if threshold provided
        if self.hist_thresh is not None:
            thresh_path = tempfile.mkdtemp()
            self._threshold_merge_histories(self.hist_thresh, thresh_path)
            print('Merged histories thresholded using threshold of %f in folder %s' 
                % (self.hist_thresh, thresh_path))

        # Select subset of data for the crag
        if inds is not None:
            print('Subseting in interval (%d, %d)' %(inds[0], inds[1]))
            self.gt = _restrict_set(self.gt, inds)
            self.sps = _restrict_set(self.sps, inds)
            self.raws = _restrict_set(self.raws, inds)
            self.histories = _restrict_set(self.histories, inds, False)
            self.membranes = _restrict_set(self.membranes, inds)

        # Prepare file
        self.hdf = os.path.join(self.output, 'hdf')
        if not os.path.exists(self.hdf):
            os.mkdir(self.hdf)
        self.project_file = os.path.join(self.hdf, 'training_dataset.h5')

        if os.path.exists(self.project_file) and overwrite is False:
            raise IOError('Project file already exists!')

        print('Generating project file %s ...' % self.project_file)

        # Create CRAG
        args = [ "cmc_create_project", "-c", config_file,
            "--supervoxels=" + self.sps, "--mergeHistory=" + self.histories,
            "--groundTruth=" + self.gt,  "--intensities=" + self.raws,
            "--boundaries=" + self.membranes, "-p", self.project_file,
            "--mergeHistoryWithScores", "--maxZLinkHausdorffDistance=" + str(zlink)
        ]
        p = Process(args, os.path.join(self.logs, 'generate_project.log'))
        p.execute()

        # Clean tmps if path used
        if self.histories is None:
            shutil.rmtree(tmpath)
        if self.hist_thresh is not None:
            shutil.rmtree(thresh_path)
        if inds is not None:
            shutil.rmtree(self.gt)
            shutil.rmtree(self.sps)
            shutil.rmtree(self.raws)
            shutil.rmtree(self.histories)
            shutil.rmtree(self.membranes)

    def _generate_merge_histories(self, tmpath):
        """ Creates the merge history for all sections in the given path """
        print('Extracting merge trees with Candidate_mc ...')
        mt.MCTreeExtractor(self.sps, self.membranes).extract(tmpath)
        self.histories = tmpath

    def _threshold_merge_histories(self, thresh, tmpath):
        """ Thresholds the current merge trees by the input value """
        mt.thresh_histories(self.histories, thresh, tmpath)
        self.histories = tmpath

    def extract_features(self, config_file):
        """ Extracts the features for the CRAG """

        if not os.path.isfile(config_file):
            raise ValueError('A config file must be provided')

        if self.project_file is None:
            raise IOError('No CRAG project has been read/generated')

        args = [
            "cmc_extract_features", "-c", config_file,
            "-p", self.project_file,
        ]

        p = Process(args, os.path.join(self.logs, "extract_features.log"))
        p.execute()

    def extract_best_effort(self, config_file, loss_type=LossType.ASSIGNMENT,
        best_effort=None, logs=None, overwrite=False):
        """ Extracts the best effort CRAG from the fed data into the project
        Params
        ----------
        conf_file: string
            Path to the configuration file
        loss_type: LossType
            Type of loss to use to compute the best effort from the groundtruth.
            By default uses the assignment loss (recommended)
        best_effort: string
            Path to the folder where best effort need to be saved. It is
            always stored in the project itself, this is an additional option.
            If set to None, no output is generated.
        logs: string
            Folder where to store the logs of the process
        overwrite: boolean
            Whether to overwrite previous best effort output in the given location.
            Only used when best effort option has a valid path
        """

        if not os.path.isfile(config_file):
            raise ValueError('A config file must be provided')

        if self.project_file is None:
            raise IOError('No CRAG project has been read/generated')

        if loss_type not in [LossType.HAUSDORFF, LossType.RAND, LossType.OVERLAP,
            LossType.HEURISTIC, LossType.ASSIGNMENT]:
            raise ValueError('Not valid loss type {}'.format(loss_type))

        # Extract best effort using Assignment Solver
        args = ["cmc_train", "-c", config_file, "-p", self.project_file, "--dryRun",
                "--assignmentSolver", "--bestEffortLoss=" + loss_type]

        # Append best-effort if requested
        if best_effort is not None:
            if not os.path.exists(best_effort):
                os.makedirs(best_effort)
            else:
                if overwrite is False:
                    raise IOError('Best effort folder already exists')
            args.append("--exportBestEffort=" + best_effort)

        logs_path = None if logs is None else os.path.join(logs, 'extract_best.log')
        p = Process(args, logs_path)
        p.execute()

        self.best_effort = best_effort

    def solve(self, node_bias, edge_bias, outp, iterations):
        """ Solves the inference problem given the costs assigned
        Params
        ---------
        node_bias: double
            Bias to be added to each node weight
        edge_bias: double
            Bias to be added to each edge weight
        outp: string
            Folder where to extract the solution found
         """

        if not os.path.isdir(outp):
            dataio.create_dir(outp)

        if self.project_file is None:
            raise IOError('No CRAG project has been read/generated')

        args = [
            "cmc_solve", "-f", str(node_bias), "-b", str(edge_bias),
            "-p", self.project_file, "--exportSolution=" + str(outp),
            "--numIterations=" + str(iterations), "--readOnly"
        ]

        p = Process(args, os.path.join(self.logs, "solve.log"))
        p.execute()

        self.solution = outp

    def evaluate_solution(self, solution=None, init=None, end=None):
        """  Evaluates previously computed solution or provided in the argument
        :param solution: Path to solution of the CRAG inference process. Set to None to use
            the previously computed one
        :param init: First section in the interval to evaluate. Set to None for all
        :param end: Last section in the interval to evaluate. Set to None for all
        :return: Returns the Adjusted rand and Variation of information of the solution
        """

        if self.project_file is None:
           raise IOError('No CRAG project has been read/generated')

        if self.solution is None and solution is None:
           raise ValueError('No solution has been saved')

        # Select best effort
        sol = solution if self.solution is None else self.solution
        with h5py.File(self.project_file) as f:
            gt = f['volumes']['groundtruth']
            return evaluate_solution(sol, gt, init, end)

    def visualize(self, costs_name='best-effort'):
        """ Visualizes the CRAG in the input path 
        :param path: Path to the HDF5 file or to the CRAG folder
        :param costs_name: Name of the costs to display. By default visualizes the 'best-effort' costs """
        args = [ "crag_viewer", "-p", self.project_file, "--showCosts=" + costs_name]
        #crag_viewer -p sampleA/mc_heuristic_600/hdf/training_dataset.h5 "$@"
        p = Process(args, os.path.join(self.logs, 'view.log'))
        p.execute()

    def get_crag_file(self):
        """ Returns the path to the CRAG project file """
        return self.project_file

    def evaluate_best_effort(self, init=None, end=None, best=None):
        """ Gives VOI and Adjusted Rand measures about the quality of 
        the solution proposed by the best effort of the CRAG. Additionally, TED
        scores can also be included
        Params
        ---------
        init: integer
            First section in the interval to evaluate. Set to None for all
        end: integer
            Last section in the interval to evaluate. Set to None for all
        best: string
            If solution has not been computed before, must contain the path to it
         """

        if self.project_file is None:
            raise IOError('No CRAG project has been read/generated')

        if self.best_effort is None and best is None:
            raise ValueError('No soluton has been saved')

        # Select best effort
        best = best if self.best_effort is None else self.best_effort

        with h5py.File(self.project_file) as f:
            gt = f['volumes']['groundtruth']
            return evaluate_solution(best, gt, init, end)


def evaluate_solution(sp_path, gt, init, end):
    """ Evaluate a pair of corresponding images (superpixel-groundtruth) and returns
    the corresponding Adjusted Rand and Variation of Information metrics
        :param sp_path: Superpixel folder
        :param gt: Groundtruth volume in HDF5 file
        :param init: First section in the interval to evaluate. Set to None for all
        :param end: Last section in the interval to evaluate. Set to None for all
    """
    # Extract superpixel names
    sps_imgs = dataio.FileReader(sp_path).extract_files()

    # Compute interval
    start = 0 if init is None else init
    ending = len(sps_imgs) - 1 if end is None else end
    if ending is not None and ending > len(sps_imgs):
        print('Cannot evaluate further than section %d' % len(sps_imgs) - 1)
    print('Evaluating CRAG in interval (%d-%d)' % (start, ending))

    # Compute average metrics
    r, vs, vm = 0.0, 0.0, 0.0
    for i in range(start, ending):
        print('Evaluating image %s, iteration %d' % (sps_imgs[i], i))
        sp = mh.imread(sps_imgs[i])
        r += rand.adapted_rand(sp, gt[i], all_stats=False)
        v = voi.voi(sp, gt[i])
        vs += v[0]
        vm += v[1]
    # Average over evaluated images
    num = ending - start + 1
    return r/float(num), vs/float(num), vm/float(num)


def _restrict_set(folder, inds, img=True):
    """ Copies the corresponding files in the folder into a 
    temporary location and returns the path
    :param folder: Folder to get images from
    :param inds: Indexes of files to select from folder (indexes of sorted filenames)
    :param img: Whether the files in the folder refer to images or not """
    # Select files accordint to mode, get subset
    files = dataio.FileReader(folder) if img is True else \
        dataio.FileReader(folder, exts=['.txt', '.dat', '.data'])
    names = files.extract_files()
    subset = names[inds[0]:inds[1]]
    # Copy subset into temporary location
    tmpath = tempfile.mkdtemp()
    for i in subset:
        dst = os.path.join(tmpath, os.path.basename(i))
        shutil.copyfile(i, dst)
    return tmpath


def evaluate_costs(crag_path, merges, ends, iterations, starting, ending, tmp=None):
    """ Evaluates the assignation of the given merges and end
    scores and returns the corresponding stats. Assumes node features have
    already been assigned
        :param crag_path: CRAG file path
        :param merges: Merge score to assign
        :param ends: End weight to assign
        :param iterations: Maximum number of iterations
        :param folder: Folder where to generate solution
        :param starting: Starting section to evaluate. Set to None for all
        :param ending: Ending section to evaluate. Set to None for all
    """
    # Manage temporary file creation
    if tmp is None:
        tmp_path = tempfile.mkdtemp()
    else:
        dataio.create_dir(tmp)
        tmp_path = tmp
    # Copy crag into temporary file
    copy_path = os.path.join(tmp_path, 'crag.h5')
    shutil.copyfile(crag_path, copy_path)
    # Generate and evaluate solution
    cg = generate_solution(copy_path, merges, ends, iterations, tmp_path)
    rand_score, voi_split, voi_merge = cg.evaluate_solution(init=starting, end=ending)
    # Clean temporary data
    shutil.rmtree(tmp_path)
    return {'rand': rand_score, 'voi_split': voi_split, 'voi_merge': voi_merge}


def generate_solution(crag_path, merges, ends, iterations, folder, model=None, batch_size=128):
    """ Evaluates the assignation of the given merges and end
    scores and returns the corresponding stats. If node features need also to be assigned, 
    a path to the model folder must be provided.
        :param crag_path: CRAG file path
        :param merges: Merge score to assign
        :param ends: End weight to assign
        :param iterations: Maximum number of iterations
        :param folder: Folder where to generate solution
        :param model: Folder where the trained network is. To avoid setting the node features
            in this step, set this to None
        :param batch_size: Batch of the size to use for descriptor extraction. If no model
            provided, this is not used
    Returns
    -------
        cg: Crag class containing the solution
    """
    print('Generating solution for merge score %f and end score %f' % (merges, ends))

    # Save updated costs
    costs = cc.CragCostManager(crag_path)
    costs.update_node_weights(merges, weights=None)
    costs.update_edge_weights(ends)
    if model is not None:
        costs.update_node_features(model)
    costs.save()

    # Evaluate resulting crag
    aux_crag = os.path.join(folder, 'project')
    cg = CragGenerator(aux_crag)
    cg.read_crag(crag_path)
    cg.solve(merges, ends, folder, iterations)
    return cg

