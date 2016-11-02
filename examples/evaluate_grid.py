#!/usr/bin/python

from neuralimg.evaluation import speval as ev

""" Evaluates the superpixel segmentation under different mask sizes and gaussian sigmas of the filters """

# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth'. Regions that contain less than 7 pixels have also
# been erased)

masks = [3, 5]
sigmas = [0.5]
ted_shift = 25
split_bg = True
membranes = 'data/crag/mem'
truth = 'data/crag/gt'

stats = ev.segmentation_grid(membranes, truth, masks, sigmas, ted_shift, True, workers=3)
ev.save_stats(stats, 'stats_grid_' + str(ted_shift) + '.dat')
