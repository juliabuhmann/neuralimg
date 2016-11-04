#!/usr/bin/python

from neuralimg.evaluation import speval as ev

""" Evaluates the superpixel segmentation under different mask sizes and gaussian sigmas of the filters """

# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth'. Regions that contain less than 7 pixels have also
# been erased)

# Fixed parameters
split_bg = False    # Whether to consider or not separate errors for background and foreground
mweight = 10    # Weight to use for the merges wrt to the splits
ted_shift = 25  # TED tolerance

# Data
masks = [3, 5, 7, 9, 11, 13, 15]
sigmas = [0.5, 1]
#masks = [3, 5]
#sigmas = [0.5, 1]
#truth = 'data/crag/gt'
#membranes = 'data/crag/mem'
truth = '/DataDisk/morad/cremi/groundtruth_large_unconnected/sampleA'
membranes = '/DataDisk/morad/cremi/membrane/sampleA'

stop = True # Whether to stop when minimum found or compute stats for all cuts
m_values = 12   # Cuts to perform to the merge history
nworkers = 7    # Number of simultaneous workers
ted_workers = 4 # Number of workers to use for each ted execution. Total threads will be of order nworkers * ted_workers
#tmp_path = '/DataDisk/morad/tmp'    # Folder to use as temporary in case tmp OS folder is limited
tmp_path = None # Use OS tmp filesystem

best, all_data, sigma, mask = ev.optimal_segmentation(masks, sigmas, membranes, truth, ted_shift, 
    split_bg=split_bg, mweight=mweight, merge_values=m_values, stop=stop, 
    workers=nworkers, ted_workers=ted_workers, tmp_path=tmp_path)
print('Best configuration found: mask %d and sigma %f' % (mask, sigma))
print('Best threshold scores: {}'.format(best))
print('All threshold scores: {}'.format(all_data))
