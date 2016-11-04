#!/usr/bin/python

import numpy as np
import os
import h5py

from neuralimg import dataio
from neuralimg.base import pyprocess as prs
from neuralimg.crag.merge_trees import *


####################################################################################
#
# This file presents a class for extracting merge trees: MCTreeExtractor
# It uses already built projects that are specified in their headers.
# 
# Merge history entries have the form:
#    region_old_1   region_old_2    new_region  score
#####################################################################################


class MCTreeExtractor(MergeTreeExtractor):


    def __init__(self, sp, mem, group='stack', smallRegion=None, int_thresh=None):
        """ Membrane and superpixels are read and parameters are set
            :param sp: Superpixel folder or HDF5 file
            :param mem: Membrane folder or HDF5 file
            :param group: Field where to look for the data if input is HDF5
            :param smallRegion: Maximal size for a region to be considered small. If 
                not None, small superpixels will be merged first
            :param int_thresh: Mean intensity needed to consider a supervoxel to be small.
                To consider all, set to None.
        """
        MergeTreeExtractor.__init__(self, sp, mem)
        self.group = group
        self.small = smallRegion
        self.int_thresh = int_thresh

        if smallRegion is None and int_thresh is not None:
            raise ValueError('Intensity threshold cannot be set if a maximal' +  
                'size for small regions is not specified')

    def process_h5(self, inp):
        """ Dumps images in the volume into a temporary folder """
        with h5py.File(inp, 'r') as f:
            imgs = f[self.group][:]
        name = os.path.splitext(os.path.basename(inp))[0]
        dst = os.path.join(self.tmp, name)
        dataio.volume_to_folder(imgs, dst)
        return dst

    def process_folder(self, inp):
        return inp

    def extract_trees(self, outp):

        dataio.check_folder(self.sp)
        dataio.check_folder(self.mem)

        sp_files = dataio.FileReader(self.sp).extract_files()
        mem_files = dataio.FileReader(self.mem).extract_files()

        # Iterate through all files and extract history independently
        for (i, j) in zip(sp_files, mem_files):
            file_name = os.path.splitext(os.path.basename(j))[0]
            output_file = os.path.join(outp, file_name + ".txt")
            history = self._extract_specific(i, j, output_file)
            print('Created merge history {}'.format(history))

    def _extract_specific(self, sp, mem, path):
        """ Extracts the history of a pair of paths for the corresponding membrane
        and superpixel views of the same image and stores it in destionation """

        dataio.check_file(sp)
        dataio.check_file(mem)

        args = [
            "merge_tree",
            "--initialSuperpixels=" + sp,
            "--source=" +  mem,
            "--mergeHistory=" + path,
        ]

        # Region size options
        if self.small is None:
            args.append("--dontConsiderRegionSize")
        else:
            if self.int_thresh is not None:
                args.append('--smallRegionThreshold2=' + str(self.small))
                args.append('--intensityThreshold=' + str(self.int_thresh))
            else:
                args.append('--smallRegionThreshold1=' + str(self.small))

        # Launch process
        p = prs.Process(args)
        p.execute()

        return path

    def clean_specific(self):
        print('Do nothing on exit ...')
