#!/usr/bin/python

from neuralimg.evaluation import speval as ev
import os

""" Returns the stats of the superpixel segmentation (merge history to be created using extract_merge_trees) """

# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth')

superpixels = 'data/crag/sps'
truth = 'data/crag/gt'
hist = os.path.join(superpixels, 'histories_gala')
thresh = -1
ted_shift = 25
split_bg = False
outp = 'tmpA/'

seg = ev.evaluate_merge(superpixels, truth, hist, thresh, ted_shift, split_bg, outp, workers=3)
