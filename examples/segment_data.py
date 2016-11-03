# !/usr/bin/python

import os
from neuralimg.image import preproc as pr
from neuralimg.crag.merge_mc import *

""" Generates superpixels given the membrane probabilities """

mask = 7
sigma = 0.25
sps_folder = 'data/crag/sps'
mems_folder = 'data/crag/mem'
mems_norm = 'data/crag/mem_norm'

# Extract images
proc = pr.DatasetProc(mems_folder)
proc.read()
proc.segment(mask, sigma)
proc.save_data(sps_folder)

# Generate normalized membranes [0,1]
proc = pr.DatasetProc(mems_folder)
proc.read()
proc.normalize()
proc.save_data(mems_norm)

# MC Extractor
mc_path = os.path.join(sps_folder, 'histories_mc')
MCTreeExtractor(sps_folder, mems_norm).extract(mc_path)
