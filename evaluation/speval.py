#!/usr/bin/python

from joblib import Parallel, delayed

import re
import numpy as np
import tempfile
import shutil
import logging

from neuralimg import dataio

from neuralimg.evaluation import rand
from neuralimg.evaluation import voi

from neuralimg.crag import crag as cr
from neuralimg.crag import merge_mc as mt

from neuralimg.dataio import *
from neuralimg.base import pyprocess as ps

from neuralimg.image import segment as seg
from neuralimg.image import preproc as pr

logging.basicConfig(filename='evaluationB.log',level=logging.DEBUG)


def call_ted(segmented, truth, shift, files=None, split_background=False, threads=5):
    """ Calls TED on the given groundtruth and superpixel files
    and stores the corrected versions of the images and tge splits and merges in the
    given folder
    Args:
        segmented: Path to segmented image
        truth: Path to groundtruth
        shift: Tolerance TED shift
        files: Where to store the output files. Set None to disable
        split_background: Whether to split merges/splits between non-background and background
        threads: Maximum threads to use
    """

    if split_background is True and files is None:
        raise ValueError('Needed an output folder in order to compute' + 
            ' separate merges and splits for background and non-background regions')

    ted_args = [ 
        "ted",
        "--reconstruction", segmented,
        "--groundTruth", truth,
        "--maxBoundaryShift", str(shift),
        "--reportVoi",
        "--reportRand",
        "--reportDetectionOverlap=false",
        "--numThreads=" +  str(threads),
    ]

    if files is not None:
        ted_args.append("--tedErrorFiles=" + files)

    proc = ps.Process(ted_args, out_str=True)
    output = proc.execute(verbose=False)

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


def get_ted_stats(sp_path, gt, ted_shift, outp, split_bg, num_regions, ted_workers=1):
    """ Compute extended stats from ted """
    # Compute merge tree image using input threshold and save image
    stats = call_ted(sp_path, gt, ted_shift, files=outp, 
        split_background=split_bg, threads=ted_workers)

    # STATS - Add some information to stats
    stats['NUM'] = num_regions
    stats['TED SHIFT'] = ted_shift
    return stats


def optimal_segmentation(masks, sigmas, membranes, truth, ted_shift, split_bg=False, 
    mweight=10, merge_values=15, stop=False, workers=3, ted_workers=1, tmp_path=None):
    """ 
        Evaluates the full segmentation parametrization and returns the one that is optimal.
        More precisel, performs a grid search over masks and sigmas of the watershed algorithm.
        For each setting, performs a search over a set of possible thresholds, getting the best balance
        between merge and split error given the weight of the merge errors given. The minimum of these
        thresholds is considered the optimal segmentation. 
        Args:
            masks: List of masks to test
            sigmas: List of sigmas to test
            membranes: Membrane predictions
            truth: Folder containing groundtruth images
            ted_shift: minimum shift in the neuron boundaries for considering errors
            split_bg: Whether to split between errors and background errors
            mweight: weight to apply to merges. To be between 1 and 20.
            merge_values: Number of cuts to test for the merge tree histories
            stop: Whether to stop when first minimum found for a given threshold value.
            workers: Number of workers to use. Default: 3
            ted_workers: Number of workers to use for each image evaluation. Should be kept
                low since the total number of threads will be of the order of workers * ted_workers
            tmp_path: Folder to use as temporary path for intermediate files. Set to None to use OS default.
        Return:
            mask, sigma and threshold to use
    """
    with Parallel(n_jobs=workers) as parallel:
        jobs = []
        for ms in masks:
            for s in sigmas:
                jobs.append(delayed(get_threshold)(ms, s, membranes, truth, ted_shift, 
                    split_bg, mweight, merge_values, stop, ted_workers, tmp_path))

        #  Append average for the current configuration
        stats = parallel(jobs)

    # Show stats
    for i in stats:
        best, all_s, sigma, mask = i
        logging.debug('Showing sigma %f and mask %d' % (sigma, mask))
        logging.debug('Best: {}'.format(best))
        logging.debug('All: {}'.format(all_s))
        logging.debug('-----------------------------------------------\n\n')
    # Return best
    score = [i[0]['merges'] for i in stats] # Select the ones with less merges
    return stats[score.index(min(score))]


def get_threshold(mask, sigma, mems, truth, ted_shift, split_bg=False, mweight=10, 
    merge_values=15, stop=False, ted_workers=1, tmp_path=None):
    """ Gets the best threshold in the merge trees computed on the segmentation 
    given by the input parameters 
    Args:
        masks: List of masks to test
        sigmas: List of sigmas to test
        membranes: Membrane predictions
        truth: Folder containing groundtruth images
        ted_shift: minimum shift in the neuron boundaries for considering errors
        split_bg: Whether to split between errors and background errors
        mweight: weight to apply to merges. To be between 1 and 20.
        merge_values: Number of cuts to test for the merge tree histories
        stop: Whether to stop when first minimum found for a given threshold value.
        workers: Number of workers to use. Default: 3
        ted_workers: Number of workers to use for each image evaluation. Should be kept
            low since the total number of threads will be of the order of 
            workers * ted_workers
        tmp_path: Folder to use as temporary path for intermediate files. Set to None 
            to use OS default.
    Return:
        best setting, all settings, sigma and mask
    """
    logging.info('Evaluating sigma %f and mask %d' % (sigma, mask))

    # Prepare temporary folder
    if tmp_path is not None:
        create_dir(tmp_path)
        outp = os.path.join(tmp_path, '_'.join(['s', str(sigma), 'm', str(mask)]))
        create_dir(outp)
    else:
        outp = tempfile.mkdtemp()

    sps = os.path.join(outp, 'sps')
    hists = os.path.join(outp, 'hists')

    # Save superpixels
    proc = pr.DatasetProc(mems)
    proc.read()
    proc.segment(mask, sigma)
    proc.save_data(sps)

    # Create merge trees
    mt.MCTreeExtractor(sps, mems).extract(hists)

    # Get optimal threshold
    best, all_stats = search_threshold(sps, truth, hists, ted_shift, mweight, 
        merge_values, split_bg, out_stats=None, stop=stop, ted_workers=ted_workers)

    # Clean temporary folder
    shutil.rmtree(outp)

    return best, all_stats, sigma, mask


def search_threshold(superpixels, truth, histories, ted_shift, mweight=10, merge_values=15, 
    split_bg=False, out_stats=None, stop=False, ted_workers=1):
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
            merge_values: Number of cuts to test for the merge tree histories
            split_bg: whether to split the TED measures for background and
                non-background errors
            out_stats: Path where to store the results for the evaluation. None to disable
            stop: Whether to stop when best weighted configuration has been met
            ted_workers: Number of workers to use for each image evaluation. Should be kept low.
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

    logging.debug('Selected cuts: {}'.format(cuts))
    data, best = [], None
    for index, i in enumerate(cuts):

        # Compute weighted score and choose minimum
        stats = evaluate_merge(superpixels, truth, histories, i, ted_shift,
            split_bg=split_bg, outp=None, ted_workers=ted_workers)
        weighted_sign = stats['TED FM'] * mweight - stats['TED FS']
        weighted = np.abs(weighted_sign)

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


def evaluate_merge_parallel(sp, gt, hist, thresh, ted_shift, nworkers=5, 
    split_bg=False, ted_workers=1, outp=None):
    """
    Given the input images and the merge history, gets the score resulting
    from merging up to the given threshold
        Args:
            sp: Path to superpixel folder
            gt: Path to corresponding groundtruth folder
            hist: Merge histories for the input images (file or list of histories)
            thresh: Merge threshold
            ted_shift: Shift pixels to use for evaluation
            nworkers: Number of parallel jobs to use
            split_bg: whether to split the TED measures for background and non-background errors
            workers: Number of workers to use. By default: 3
            ted_workers: Number of workers to use for each image evaluation. Should be kept low.
            outp: Path where to store results. Set to None to disable
        Return:
            Stats of the segmentation
    """
    with Parallel(n_jobs=nworkers) as parallel:

        images, gts, hists = read_folders(sp, gt, hist)

        # One job for each segmented image
        jobs = []
        for (i, g, h) in zip(images, gts, hists):
            jobs.append(delayed(evaluate_segmentation)(i, g, h, thresh, ted_shift,
                split_bg, outp=None, ted_workers=ted_workers))
        total_stats = parallel(jobs)

    avg_stats = average_stats(total_stats)
    if outp is not None:
        save_stats([avg_stats], outp)
    return avg_stats


def evaluate_merge(sp, gt, hist, thresh, ted_shift, split_bg=False, outp=None, ted_workers=1):
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
            outp: Path where to store the TED output files (merges and splits). Set None to disable.
            workers: Number of workers to use. By default: 3
            ted_workers: Number of workers to use for each image evaluation. Should be kept low.
        Return:
            Stats of the segmentation
    """

    images, gts, hists = read_folders(sp, gt, hist)
    total_stats = [evaluate_segmentation(i, g, h, thresh, ted_shift, split_bg, 
        outp, ted_workers) for (i, g, h) in zip(images, gts, hists)]
    return average_stats(total_stats)


def evaluate_segmentation(sp_img, gt, hist, thresh, ted_shift, split_bg, outp, 
    ted_workers):
    """
    Evaluates the segmentation of the input file (path) with respect to the the groundtruth image,
    and the threshold to cut from the merge history.
        Args:
            sp_img: Path to superpixel image
            gt: Path to corresponding groundtruth image
            hist: Merge history for the input image (path or list format)
            thresh: Merge threshold. Set to negative to avoid using thresholding. 
                Then merge history will be ignored
            ted_shift: Shift pixels to use for evaluation
            split_bg: whether to split the TED measures for background and non-background errors
            outp: Path where to store the TED output files (merges and splits). Set None to disable.
            ted_workers: Number of workers to use for each image evaluation. Should be kept low.
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
        fd, sp_path = tempfile.mkstemp(suffix='.tif')
        mh.imsave(sp_path, merged)

    num_regions = len(np.unique(merged))
    stats = get_ted_stats(sp_path, gt, ted_shift, outp, split_bg, 
        num_regions, ted_workers=ted_workers)
    stats['MT THRESH'] = thresh

    # Do not forget to erase temporary file, if used
    if thresh >= 0.0:
        os.close(fd)
        os.remove(sp_path)

    return stats


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
        logging.debug('Evaluating image %s against %s' % (s, g))
        sp, gt = mh.imread(s), mh.imread(g)
        r += rand.adapted_rand(sp, gt, all_stats=False)
        v = voi.voi(sp, gt)
        vs += v[0]
        vm += v[1]
    return r, [vs, vm]


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

    logging.info('Evaluating CRAG for maxZLink %d ...' % dist)

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

                logging.info('Evaluating sigma %f and mask %d' % (s, ms))

                jobs = []

                # Initialize watershed segmenter
                # Tempfile is only needed if split_bg requested
                outp = tempfile.mkdtemp()
                ws = seg.Watershed(s, ms)
                for (i, g) in zip(images, gts):
                    # Save superpixel into tmp folder
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

