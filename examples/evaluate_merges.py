#!/usr/bin/python

from neuralimg.evaluation import speval as ev
import os

""" The merge trees create a structure where the top element is a root supervoxel including all of them.
This example evaluates several thresholds from both sets of merge trees and returns the best configuration found
given the desired balance between split and merge errors (merge history to be created using extract_merge_trees) """

# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth')

superpixels = 'data/crag/sps'
truth = 'data/crag/gt'
ted_shift = 25
split = False

# Candidate MC trees
histories = os.path.join(superpixels, 'histories_mc')
print('Evaluating candidate MC merge trees ...')
best, data = ev.search_threshold(superpixels, truth, histories, ted_shift,
    out_stats=os.path.join(superpixels, 'mc_merges.dat'), split_bg=split)
print('Best configuration found:')
print(best)

# Gala trees
#histories = os.path.join(superpixels, 'histories_gala')
#print('Evaluating gala merge trees ...')
#best, data = ev.search_threshold(superpixels, truth, histories, ted_shift,
#    out_stats=os.path.join(superpixels, 'gala_merges.dat'), split_bg=split)
#print('Best configuration found:')
#print(best)
