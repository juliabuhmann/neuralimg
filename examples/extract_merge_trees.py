#!/usr/bin/python

from neuralimg.crag import merge_trees as mt
import os

""" Extracts merge history tree out of the superpixels """

if __name__ == '__main__':

    gt_folder = 'data/crag/gt'
    sps_folder = 'data/crag/sps'
    mems_folder = 'data/crag/mem_norm'

    # MC Extractor, test both formats
    mc_path = os.path.join(sps_folder, 'histories_mc')
    mt.MCTreeExtractor(sps_folder, mems_folder).extract(mc_path)

    # Using gala with some subset data
    g_path = os.path.join(sps_folder, 'histories_gala')
    mt.GalaTreeExtractor(sps_folder, mems_folder, gt_folder).extract(g_path)

