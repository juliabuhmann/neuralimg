#!/usr/bin/python

from joblib import Parallel, delayed

import re
import numpy as np
import tempfile
import shutil

from neuralimg import dataio
from neuralimg.evaluation import rand
from neuralimg.evaluation import voi
from neuralimg.crag import crag as cr
from neuralimg.crag import merge_trees as mt
from neuralimg.dataio import *
from neuralimg.base import pyprocess as ps
from neuralimg.image import segment as seg

# TODO: Implement grid search for both segmentation parameters + threshold 
# parameters

def call_ted(segmented, truth, shift, files, split_background=False, threads=5):
    """ Calls TED on the given groundtruth and superpixel files
    and stores the corrected versions of the images and tge splits and merges in the
    given folder
    """

    if files is None and split_background is True:
        raise ValueError('Output for TED output files must be provided' +
            'to retrieve separate scores for background and non-background' +
            'superpixels')

    if files is not None:
        create_dir(files)

    ted_args = [ 
        "ted",
        "--reconstruction", segmented,
        "--groundTruth", truth,
        "--maxBoundaryShift", str(shift),
        "--reportVoi",
        "--reportRand",
        "--reportDetectionOverlap=false",
        "--numThreads=" +  str(threads)
    ]

    print('Calling TED with files attribute %s' % files)
    if files is not None:
        ted_args.append("--tedErrorFiles=" + files)

    proc = ps.Process(ted_args, out_str=True)
    output = proc.execute()

    # TED FP and TED FN have been erased because --haveBackground option has 
    # been disabled

    stats = { token : search_tag(output, token) for token in [ 'TED FS', 'TED FM',
        'VOI split', 'VOI merge', 'RAND'] }

    # If requested, separate scores for TED FS and TED FM for background and
    # non - background splits and merges
    if split_background is True:
        name = os.path.splitext(os.path.basename(segmented))[0]
        split_background_label(stats, os.path.join(files, name + '.merges.data'),
            os.path.join(files, name + '.splits.data'))
    return stats


def search_threshold(superpixels, truth, histories, ted_shift, mweight=10, merge_values=15, 
    split_bg=False, out_stats=None, stop=False, workers=3):
    """ 
        Evaluates the given segmentation parametrization and finds the approximate optimal
        threshold on the merge tree history. Tracks a set of cuts in increasing order
        in the score merging history values and selects the one that has the minimum difference 
        between weighted merges and splits. Since this measure (usually) monotonically 
        decreases for increasing thresholds, we can stop when we find the first local minimum
        Args:
            superpixels: Folder containing superpixel images
            truth: Folder containing groundtruth images
            histories: Folder containing merge histories for the segmeneted images
            ted_shift: minimum shift in the neuron boundaries for considering errors
            mweight: weight to apply to merges. To be between 1 and 20.
            merge_values: merge values to test for the input configuration
            split_bg: whether to split the TED measures for background and
                non-background errors
            out_stats: Path where to store the results for the evaluation. None to disable
            stop: Whether to stop when best weighted configuration has been met
            workers: Number of workers to use. Default: 3
        Return:
            best: Stats and threshold for the best configuration, where the 
                distance between weighted merges and the splits is minimum 
            data: track of all results up to best
    """

    if merge_values <= 1:
        raise ValueError('The number of values to test must be ' 
            'in interval [2, inf)')

    # Get cuts uniformly
    hist_scores = mt.load_history_values(histories)
    cuts = [np.percentile(hist_scores, i) 
        for i in np.linspace(0, 100, merge_values)]

    print('Selected cuts: {}'.format(cuts))
    data, best = [], None
    for index, i in enumerate(cuts):

        print('--- Evaluating threshold {}'.format(str(i)))

        # Create tempath at each iteration
        out_ted = tempfile.mkdtemp()

        # Compute weighted score and choose minimum
        stats = evaluate_merge(superpixels, truth, histories, i, ted_shift,
            split_bg, outp=out_ted, workers=workers)
        weighted_sign = stats['TED FM'] * mweight - stats['TED FS']
        weighted = np.abs(weighted_sign)

        # Clean temporary folder
        shutil.rmtree(out_ted)

        # Store values
        current = {'thresh': i, 'weighted': weighted, 'merges': stats['TED FM'], \
            'splits': stats['TED FS']}
        data.append(current)

        if best is None or best['weighted'] > current['weighted']:
            best = current

        if stop == True:
            # Check for local minimum
            is_minimum = False
            if index == 1:
                is_minimum = current['weighted'] > data[index-1]['weighted']
            elif index > 1 and index < merge_values - 1:
                is_minimum = current['weighted'] > data[index-1]['weighted'] \
                    and data[index-2]['weighted'] > data[index-1]['weighted']

            # If found and option enabled, return first
            if is_minimum == True:
                best = data[index - 1]
                break

    if out_stats is not None:
        save_stats(data, out_stats)

    return best, data


def evaluate_supervoxels(sp_folder, gt_folder):
    """ Evaluates the segmented images against their groundtruth and provides
    the Adjusted Rand and Variation of Information (VOI) metrics [split, merge]
        Args:
            sp_folder: Path to superpixel folder
            gt_folder: Path to corresponding groundtruth folder
    """
    sp_files = dataio.FileReader(sp_folder).extract_files()
    gt_files = dataio.FileReader(gt_folder).extract_files()
    r, vs, vm = 0.0, 0.0, 0.0
    for (s, g) in zip(sp_files, gt_files):
        print('Evaluating image %s against %s' % (s, g))
        sp, gt = mh.imread(s), mh.imread(g)
        r += rand.adapted_rand(sp, gt, all_stats=False)
        v = voi.voi(sp, gt)
        vs += v[0]
        vm += v[1]
    return r, [vs, vm]


def evaluate_merge(sp, gt, hist, thresh, ted_shift, split_bg, outp, workers=2):
    """
    Given the input images and the merge history, gets the score resulting
    from merging up to the given threshold
        Args:
            sp: Path to superpixel folder
            gt: Path to corresponding groundtruth folder
            hist: Merge histories for the input images (file or list of histories)
            thresh: Merge threshold
            ted_shift: Shift pixels to use for evaluation
            split_bg: whether to split the TED measures for background and non-background errors
            outp: Path where to store the TED output files (merges and splits)
            workers: Number of workers to use. By default: 3
        Return:
            Stats of the segmentation
    """

    images, gts, hists = read_folders(sp, gt, hist)

    with Parallel(n_jobs=workers) as parallel:

        # Read stats for each segmentation in the given folders and compute in 
        # parallel way
        jobs = []
        for (i, g, h) in zip(images, gts, hists):
            jobs.append(delayed(evaluate_segmentation)(i, g, h, thresh, ted_shift, split_bg, outp=outp))
        total_stats = parallel(jobs)

        return average_stats(total_stats)


def evaluate_segmentation(sp_img, gt, hist, thresh, ted_shift, split_bg, outp):
    """
    Evaluates the segmentation of the input file (path or image), the groundtruth image,
    the merge history (path or list) and the merge threshold. The output of the TED can be stored
    in a folder or disabled by setting 'outp' to None. 'split_bg' to True separates the errors between
    background and non-background (background with label 0)
        Args:
            sp_img: Path to superpixel image
            gt: Path to corresponding groundtruth image
            hist: Merge history for the input image
            thresh: Merge threshold
            ted_shift: Shift pixels to use for evaluation
            split_bg: whether to split the TED measures for background and non-background errors
            outp: Path where to store the TED output files (merges and splits)
        Return:
            Stats of the segmentation
    """

    # Get superpixel image
    merged = mt.merge_superpixels(sp_img, hist, thresh) \
        if thresh >= 0.0 else mh.imread(sp_img)

    # Create temporary file for the merged superpixels
    # If threshold negative, no merge
    sp_path = sp_img
    if thresh >= 0.0:
        sp_path = tempfile.mkstemp(suffix='.tif')[1]
        mh.imsave(sp_path, merged)

    print('Evaluating image %s as %s' % (sp_img, sp_path))
    print('Saving into %s' % outp)
    stats = get_ted_stats(sp_path, gt, ted_shift, outp, split_bg, len(np.unique(merged)))
    stats['MT THRESH'] = thresh

    # Do not forget to erase temporary file
    if thresh >= 0.0:
        os.remove(sp_path)

    return stats


def evaluate_crag(sps, gts, raws, mems, hists, create_conf, features_conf, 
    effort_conf, dist, thresh, tmp=None):
    """ Single CRAG evaluation using Assignment loss for the best effort extraction
        :param sps: Superpixel folder 
        :param gts: Groundtruth folder
        :param raws: Raw image folder
        :param mems: Membrane probability folder
        :param hists: Merge tree histories 
        :param create_conf: Project creation configuration file
        :param features_conf: Feature computation configuration file
        :param effort_conf: Effort configuration file
        :param dist: List of distances to evaluate
        :param thresh: Threshold to apply to the merge tree history
        :param tmp: Folder to use as temporary folder. Set to None to use OS one
    """

    print('Evaluating CRAG for maxZLink %d ...' % dist)

    # Auxiliar temp folder for current crag
    if tmp is None:
        tmp_out = tempfile.mkdtemp() 
    else:
        tmp_out = os.path.join(tmp, str(dist))
        create_dir(tmp_out)

    best_folder = os.path.join(tmp_out, 'best')

    # Generate crag, extract features and best effort
    cragen = cr.CragGenerator(tmp_out)
    cragen.generate_crag(gts, sps, raws, mems, create_conf, max_zlink=dist, 
        histories=hists, histories_thresh=thresh)
    cragen.extract_features(features_conf)
    cragen.extract_best_effort(effort_conf, cr.LossType.ASSIGNMENT, best_folder)

    # Evaluate best effort against groundtruth and save results
    rand, voi = cragen.evaluate_best_effort()

    # Delete temp folder
    shutil.rmtree(tmp_out)

    return {'rand': rand, 'voi': voi, 'dist': dist}


def evaluate_crags(sps, gts, raws, mems, hists, create_conf, features_conf, 
    effort_conf, dists, nworkers=3, thresh=None, outp=None, tmp=None):
    """ Returns the stats of evaluating the best effort from the crag and
    the groundtruth with each of the maximum hausdorff distance using heuristic loss
        :param sps: Superpixel folder 
        :param gts: Groundtruth folder
        :param raws: Raw image folder
        :param mems: Membrane probability folder
        :param hists: Merge tree histories 
        :param create_conf: Project creation configuration file
        :param features_conf: Feature computation configuration file
        :param effort_conf: Effort configuration file
        :param dists: List of distances to evaluate
        :param thresh: Threshold to apply to the merge tree history
        :param dists: List of distances to evaluate
        :param nworkers: Number of workers to use
        :param outp: Output file where to store the resulting stats. Set to None to disable
        :param tmp: Folder to use as temporary folder. Set to None to use OS one
    """

    # Create directory if provided
    if tmp is not None:
        create_dir(tmp)

    jobs, stats = [], []

    with Parallel(n_jobs=nworkers) as parallel:
        # Iterate through possible distances
        for d in dists:
            jobs.append(delayed(evaluate_crag)(sps, gts, raws, mems, hists, create_conf,
                features_conf, effort_conf, dist=d, thresh=thresh, tmp=tmp))
        stats = parallel(jobs)

    if outp is not None:
        save_stats(stats, outp)

    return stats


def evaluate_crag_weights(path, config_file, end_scores, merge_scores, outp, tmp=None, nworkers=3):
    """ Evaluates the RAND and VOI for the best effort using a grid search approach for both
    the merge and the ending thao
        :param path: Path to the CRAG to test
        :param config_file: Best effort configuration file
        :param end_scores: List of end_scores to test
        :param merge_scores: List of merge scores to test
        :param outp: Path where to store the stats
        :parma tmp: Path to use as temporary file. Set to None for using the default OS folder
        :param nworkers: Number of parallel workers to use
    """
    # Initialize tmp folder
    if tmp is None:
        tmp_path = tempfile.mkdtemp()
    else:
        create_dir(tmp)
        tmp_path = tmp

    # Create a copy of the original CRAG
    crag_path = os.path.join(tmp_path, 'crag.h5')
    shutil.copyfile(path, crag_path)

    # Generate a job for each configuration
    jobs = []
    with Parallel(n_jobs=nworkers) as parallel:
        for e in end_scores:
            for m in merge_scores:
                jobs.append(delayed(cr.evaluate_assignation)(crag_path, m, e, 
                config_file, tmp_path))
        stats = parallel(jobs)

    # Delete temporary folder
    shutil.rmtree(tmp_path)

    save_stats(stats, outp)


def segmentation_grid(membranes, groundtruth, masks, sigmas, ted_shift, split_bg, workers=2):
    """
    Evaluates the watershed segmentation under a grid of parameters
        Args:
            membranes: Folder containing the membrane probability images
            groundtruth: Path to corresponding groundtruth image
            masks: List of possible mask sizes to explore
            sigmas: List of sigmas to explore
            ted_shift: TED boundary shift tolerance
            split_bg: Whether to split scores for background and
                non-background TED errors
            workers: Maximum number of workers to instantiate
        Return:
            Average stats of the segmentation
    """

    images, gts = read_folders(membranes, groundtruth)
    total_stats = []

    # Create jobs so each grid configuration is tested
    # on whole set of pairs
    with Parallel(n_jobs=workers) as parallel:

        for ms in masks:
            for s in sigmas:

                print('Evaluating sigma %f and mask %d' % (s, ms))

                jobs = []

                # Initialize watershed segmenter
                # Tempfile is only needed if split_bg requested
                outp = tempfile.mkdtemp()
                ws = seg.Watershed(s, ms)
                for (i, g) in zip(images, gts):
                    # Save superpixel into tmp folder
                    print('Iteration for %s and %s' % (i, g))
                    jobs.append(delayed(get_ted_stats_img)(i, ws, g, ted_shift, split_bg, outp))

                # Append average for the current configuration
                stats = average_stats(parallel(jobs))
                stats['SIGMA'] = s
                stats['MASK'] = ms
                total_stats.append(stats)

                if outp is not None:
                    shutil.rmtree(outp)

    return total_stats


def read_folders(sp, gt, hist=None):
    """ Reads the list of files in the input images and merge histories"""
    images = FileReader(sp).extract_files()
    images.sort()
    gts = FileReader(gt).extract_files()
    gts.sort()
    if hist is not None:
        hists = FileReader(hist, exts=['.txt', '.dat', '.data']).extract_files()
        hists.sort()
        return images, gts, hists
    else:
        return images, gts


def get_ted_stats_img(mem, watershed, gt, ted_shift, split_bg, tmp_path):
    """ Return the ted stats for the given groundtruth and the resulting
    segmentation with the input setting and the given membrane probability """
    mem_img = mh.imread(mem)
    sps = watershed.segment(mem_img)
    num_regions = len(np.unique(sps))
    # Store superpixels in tmp path
    sps_path = os.path.join(tmp_path, os.path.splitext(os.path.basename(mem))[0] + '.tif')
    mh.imsave(sps_path, sps.astype(float))
    return get_ted_stats(sps_path, gt, ted_shift, tmp_path, split_bg, num_regions)


def get_ted_stats(sp_path, gt, ted_shift, outp, split_bg, num_regions):
    """ Compute extended stats from ted """
    # Compute merge tree image using input threshold and save image
    stats = call_ted(sp_path, gt, ted_shift, files=outp, split_background=split_bg)

    # STATS - Add some information to stats
    stats['NUM'] = num_regions
    stats['TED SHIFT'] = ted_shift
    return stats


def get_background_splits(path):
    """ Returns the number of splits where background is uniquely involved"""
    with open(path) as f:
        for line in f:
            chunks = line.split('\t')
            if chunks[0] == '0':
                a = len(chunks) - 2
                # Except from 0 label and \n
                return len(chunks) - 3
        return 0


def get_background_merges(path):
    """ Returns the number of merges where background is uniquely involved"""
    count = 0
    with open(path) as f:
        for line in f:
            chunks = line.split('\t')
            # String is splitted counting \n as a chunk
            if len(chunks) == 4 and (chunks[1] == '0' or chunks[2] == '0'):
                count += 1
        return count


def split_background_label(stats, merges_file, splits_file):
    """ Generates two separate labels for background and non-background 
    splits and merges """
    splits_backg = get_background_splits(splits_file)
    merges_backg = get_background_merges(merges_file)
    # Separate merge scores
    stats['TED FM'] = stats['TED FM'] - merges_backg
    stats['TED FM BG'] = merges_backg
    # Separate split scores
    stats['TED FS'] = stats['TED FS'] - splits_backg
    stats['TED FS BG'] = splits_backg


def search_tag(output, token):
    """ Parses list of comma-separated pairs of identifiers and real numbers:
        e.g. TED FS: 23, RAND: 0.92
    and returns the associated number for the given identifier """
    m = re.search(token + ' *: * ([+-]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?)', output)
    return float(m.group(1))


def average_stats(total_stats):
    """ Returns the average of the input set of stats """
    keys = total_stats[0].keys()
    avg_stats = {}
    for k in keys:
        avg = sum([stats[k] for stats in total_stats])/float(len(total_stats))
        avg_stats[k] = avg
    return avg_stats


def save_stats(all_stats, path):
    """ Write stats and the algorithm headers into the destination file """

    # Write output into file
    f = open(path, 'w')

    # Get stats headers. Assume at least one result is
    # generated and that all stats have the same information
    stats_keys = all_stats[0].keys()

    # Write headers as list of algorithm parameters and stats' columns
    head = ""
    for ident, j in enumerate(stats_keys):
        if ident > 0:
            head = head + " "
        head = head + j.replace(' ', '_')
    head = head + "\n"
    f.write(head)

    # Write all stats
    for stats in all_stats:
        line = ""
        for ident, j in enumerate(stats_keys):
            if ident > 0:
                line = line + " "
            line = line + str(stats[j])
        line = line + '\n'
        f.write(line)

    f.close()

