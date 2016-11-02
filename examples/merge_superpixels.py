#!/usr/bin/python

from neuralimg.crag import merge_trees as mt
import os

""" Merge supervoxels up to a certain score given the computed merge trees
and store the results. It is done with two different merge history trees """

if __name__ == '__main__':

    root = 'data/crag/'
    sps = os.path.join(root, 'sps')

    # Gala superpixels
    gala_thresh = 0.6835
    histories_gala = os.path.join(sps, 'histories_gala')
    out_gala = os.path.join(root, 'sps_gala' + str(gala_thresh).replace('.', ''))
    mt.merge_dataset(sps, histories_gala, gala_thresh, out_gala)

    # MC superpixels
    gala_thresh = 0.060392
    histories_gala = os.path.join(sps, 'histories_mc')
    out_gala = os.path.join(root, 'sps_mc' + str(gala_thresh).replace('.', ''))
    mt.merge_dataset(sps, histories_gala, gala_thresh, out_gala)

