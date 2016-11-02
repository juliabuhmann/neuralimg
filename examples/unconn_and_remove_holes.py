# !/usr/bin/python

from neuralimg.image import preproc

""" Preprocesses groundtruth so it joins small supervoxels with neighboring ones
and gives separate ids to unconnected regions even if they belong to the same neuron
(for ease of 2D evaluation) """

# From folder
p = preproc.DatasetProc('data/crag/gt')
p.read()
p.join_small()
p.save_data('data/crag/gt_large')
p.split_labels()
p.save_data('data/crag/gt_large_unconnected')

