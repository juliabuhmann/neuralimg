# !/usr/bin/python

import os
from neuralimg.dataio import *
from neuralimg.image import datasets as dd

""" Extracts triplet datasets for each section of the CREMI data """

# Gather crag paths
crag_sampleA = '/DataDisk/morad/all/sample_A/training_dataset.h5'
crag_sampleB = '/DataDisk/morad/all/sample_B/training_dataset.h5'
crag_sampleC = '/DataDisk/morad/all/sample_C/training_dataset.h5'
crags = [crag_sampleA, crag_sampleB, crag_sampleC]
names = ['sampleA', 'sampleB', 'sampleC']


# Some predictions are really poor due to noisy membrane predictions
# It is best to ignore them during the data creation to avoid bad quality data
# Not to be modified (just if membrane predictions improved)
excluded = [
        [], # Sample A
        [15,16,44,45,77], # Sample B
        [14,74,86]  # Sample C
]

root_output = '/DataDisk/morad/all/datasets'    # Destination root output
create_dir(root_output)
data_config = 'data/confs/data.conf'    # Change data settings according to needs

# Generate all datasets
for (crag, ex, name) in zip(crags, excluded, names):
    dataset_path = os.path.join(root_output, name + '.h5')
    print('Generating data for crag %s into %s' % (crag, dataset_path))
    dgen_pair = dd.TripletDataGen(data_config, dataset_path)
    crag_opts = dd.DataGenOptions(crag, init=None, end=None, exclude=ex)
    dgen_pair.generate_dataset([crag_opts])

