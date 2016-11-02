# !/usr/bin/python

from neuralimg.image import datasets as dd

""" Creates dataset from an existing CRAG (create using crag_gen_example) if it does not exist """

crag_path = 'out/project/hdf/training_dataset.h5'
crag = dd.DataGenOptions(crag_path, init=0, end=3, exclude=[])  
# Can create using all (init and end None) or restriction the sections where to 
# extract images from

print('Testing paired dataset single ...')

dgen_pair = dd.PairDataGen('data/confs/data.conf', 'out_pair.h5')
dgen_pair.generate_dataset([crag])

print('Testing paired dataset double ...')

dgen_pair = dd.PairDataGen('data/confs/data.conf', 'out_pair_double.h5')
dgen_pair.generate_dataset([crag, crag])

print('Testing triplet dataset ...')
dgen_pair = dd.TripletDataGen('data/confs/data.conf', 'out_triplet.h5')
dgen_pair.generate_dataset([crag])
