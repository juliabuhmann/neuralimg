#!/usr/bin/python

from neuralimg.evaluation import speval as ev
import os

""" Returns the stats of the superpixel segmentation (merge history to be created using extract_merge_trees) """

# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth')

superpixels = 'data/crag/sps'
truth = 'data/crag/gt'
hist = os.path.join(superpixels, 'histories_mc')
thresh = 0.070
ted_shift = 25
split_bg = False    # Always set to False for CREMI
workers = 8 # Parallel jobs
ted_w = 3
outp = 'hola.txt'

seg = ev.evaluate_merge_parallel(superpixels, truth, hist, thresh, ted_shift, 
    nworkers=workers, split_bg=split_bg, ted_workers=ted_w, outp=outp)
print(seg)
