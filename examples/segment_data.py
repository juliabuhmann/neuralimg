# !/usr/bin/python

from neuralimg.image import preproc as pr

""" Generates superpixels given the membrane probabilities """

mask = 7
sigma = 0.25
gt_folder = 'data/crag/gt'
sps_folder = 'data/crag/sps'
mems_folder = 'data/crag/mem'

# Extract images
proc = pr.DatasetProc(mems_folder)
proc.read()
proc.segment(mask, sigma)
proc.save_data(sps_folder)

# MC Extractor
mc_path = os.path.join(sps_folder, 'histories_mc')
mt.MCTreeExtractor(sps_folder, mems_folder).extract(mc_path)
