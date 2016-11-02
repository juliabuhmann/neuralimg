#!/usr/bin/python

from neuralimg.evaluation import speval as ev

""" Evaluates superpixels given the groundtruth and returns the Adapted Rand and the
Variation of Information (merge history to be created using extract_merge_trees) """


# Important: for a fair evaluation, segments from neurons that are the same
# but not connected should be separated in an updated groundtruth version (what
# we call 'unconnected groundtruth')

superpixels_gala = 'data/crag/gala_heuristic/best'
superpixels_mc = 'data/crag/mc_heuristic/best'
truth = 'data/crag/gt'

# GALA
rand_gala, voi_gala = ev.evaluate_volumes(superpixels_gala, truth)

# MC
rand_mc, voi_mc = ev.evaluate_volumes(superpixels_mc, truth)

print('Gala scores. RAND: %f, VOI: %f' % (rand_gala, voi_gala))
print('MC scores. RAND: %f, VOI: %f' % (rand_mc, voi_mc))
