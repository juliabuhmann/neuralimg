# !/usr/bin/python

from neuralimg.image import datasets as dd

""" Extracts a Triplet dataset out of an existing CRAG (if it does not exist, generate using crag_gen_example) """

crag_path = '/DataDisk/morad/test/sampleA/hdf/training_dataset.h5'

print('Testing triplet dataset ...')
dgen_pair = dd.TripletDataGen('data/confs/data.conf', '/DataDisk/morad/cremi/tests/ref_ids_dataset.h5')
crag1 = dd.DataGenOptions(crag_path, init=None, end=None, exclude=[])
dgen_pair.generate_dataset([crag1])
