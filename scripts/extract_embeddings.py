# ! /usr/bin/python
from bokeh.io import curdoc

from neuralimg.training import siamese as si
import os
from ml_utils import machine_learning_utils as ml
import numpy as np
import h5py
import time
from neuralimg.evaluation import helpers
from neuralimg.crag import crag_utils as cu
from pycmc import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

crag_path = '/raid/julia/documents/Dropbox/best_effort/heuristic_100sections/sampleA.hdf'
# crag_path = '/raid/julia/documents/Dropbox/best_effort/heuristic_25sections/sampleA_bestEffort.hdf'

model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_III/221238/'
outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/feature_output/sampleA_heur100.hdf'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank7/sampleA.h5'
CALCULATE_DISTANCE = True
PRINT_DISTANCE = True

imh = 128
imw = 128
clabels = ['binary', 'intensity', 'raw']
cpositions = [0, 2, 1]
padding = 1.2
normalise = True

crag, volumes, solution, ebb, ein = cu.read_crag(crag_path)
siamese = si.TripletSiamese(dataset, crag_path)
siamese.initialize_test(model_dir)

crag_ids = []
batch_size = 100

new_f = h5py.File(outputfile, 'w')
# del new_f['embedding']
new_f.create_dataset('embedding', shape=(0, 128), maxshape=(None, 128), compression='gzip',
                     chunks=True)
# new_f.create_dataset('test', shape=(0, 3, 3, 128, 128), maxshape=(None, 3, 3, 128, 128), compression='gzip',
#                      chunks=True)




start = time.time()
image_array= np.empty((batch_size, 3, imh, imw))
rel_crags = []
for n in crag.nodes():
    if crag.type(n) == CragNodeType.AssignmentNode:
        crag_ids.append(crag.id(n))
        rel_crags.append(n)

crag_ids = crag_ids
rel_crags = rel_crags
start_tim = time.time()
total_num = len(rel_crags)
current_start = 0
verbose_int = 10
num_iterations = total_num // (batch_size) + 1
print(num_iterations, 'number of iterations')
print('processing %i' %len(rel_crags))
number_of_images = batch_size
sanity_check_n = []
for ii in range(num_iterations):
    # for ii in range(2):
    end_batch = current_start + batch_size
    if end_batch > total_num:
        print('reached end', total_num)
        end_batch = total_num
        number_of_images = end_batch - current_start
        image_array = np.empty((number_of_images, 3, imh, imw))

    if ii % verbose_int == 0:
        print('%i of %i processing. took %0.3f, estimated time in mins = %0.3f' % (
            ii * batch_size, total_num, time.time() - start_tim,
            (time.time() - start_tim) * (num_iterations - ii) / float(verbose_int) / 60.))
        start_tim = time.time()

    for image_slice in range(number_of_images):
        n = rel_crags[current_start+image_slice]
        sanity_check_n.append(n)
        slice = volumes.getVolume(n)
        image_array[image_slice, ...] = cu.build_image(slice, ein, ebb,
                       clabels, cpositions, imh, imw, padding, normalise)

    # Image has shape: (3, 128, 128)
    # (?,  # Channels, Height, Width), eg. (?, 3, 128, 128)
    features = siamese.get_descriptors(image_array)
    new_f['embedding'].resize(end_batch, axis=0)
    new_f['embedding'][current_start:end_batch, :] = features

    current_start += batch_size

for ii, san_n in enumerate(sanity_check_n):
    assert crag_ids[ii] == crag.id(san_n)

assert len(crag_ids) == new_f['embedding'].shape[0]
new_f.create_dataset('ref_ids', data=np.array(crag_ids))
new_f.close()




# data = data_ori[:, 2, ...].squeeze()
# data = np.moveaxis(data, 3, 1)
# negative = siamese.get_descriptors(data)
#
#
# siamese.finalize_data()
