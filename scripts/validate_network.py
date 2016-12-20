# ! /usr/bin/python
from bokeh.io import curdoc

from neuralimg.training import siamese as si
import os
from ml_utils import machine_learning_utils as ml
import numpy as np
import h5py
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/sampleA.h5'
# dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/dummy2.h5'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/triplets/sampleA.h5'
outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/model_triplet_output_rank3modelII/distances_sampleA.h5'
models = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/221238/'

datasetname='data'
base_dir_session = '/raid/julia/projects/fragment_embedding/nn_training/'
session_dir, tensorboard_dir = ml.folder_structure_for_tensorflow(base_dir_session)

os.mkdir(tensorboard_dir + '/test')
os.mkdir(tensorboard_dir + '/train')

print(tensorboard_dir)

crag_path = '/raid/julia/documents/Dropbox/best_effort/heuristic_100sections/sampleA.hdf'
crag_path = None
model_dir = '/raid/julia/projects/fragment_embedding/nn_training/smallMLPmodel/'

if not datasetname == 'data':
    siamese = si.TripletSiamese(dataset, crag_path, testdataname='test')
else:
    siamese = si.TripletSiamese(dataset, crag_path)

siamese.initialize_test(model_dir)

total_num = siamese.data[datasetname].shape[0]


# total_num = 100

def euclidean_distance(m1, m2):
    subt = np.subtract(m1, m2)
    squared = (subt) ** 2
    dist = np.sum(squared, axis=1)
    return dist


dist_collection_pos = []
dist_collection_neg = []
current_start = 0
batch_size = 100
verbose_int = 10

dist_pos_arr = np.empty((total_num))
dist_neg_arr = np.empty((total_num))
start_tim = time.time()

num_iterations = total_num // (batch_size)+1
print(num_iterations, 'number of iterations')
for ii in range(num_iterations):
    # for ii in range(2):
    end_batch = current_start + batch_size
    if end_batch > total_num:
        end_batch = total_num

    if ii % verbose_int == 0:
        print('%i of %i processing. took %0.3f, estimated time in mins = %0.3f' % (
        ii * batch_size, total_num, time.time() - start_tim,
        (time.time() - start_tim) * (num_iterations-ii) / float(verbose_int) / 60.))
        print(np.mean(dist_pos_arr[:current_start]))
        print(np.mean(dist_neg_arr[:current_start]))

        start_tim = time.time()
    print(current_start, end_batch)
    data_ori, labels = siamese.get_validation_data(current_start, end_batch)

    # feed = siamese.build_feed(data, labels, dropout=False, augm=False)
    # outs = siamese.test_session.run(siamese.outputs, feed_dict=feed)
    # print outs[0].shape
    # NUM x ? x H x W x Channels
    # Required shape
    # print data_ori.shape
    data = data_ori[:, 0, ...].squeeze()
    data = np.moveaxis(data, 3, 1)
    anchor = siamese.get_descriptors(data)

    data = data_ori[:, 1, ...].squeeze()
    data = np.moveaxis(data, 3, 1)
    positive = siamese.get_descriptors(data)

    data = data_ori[:, 2, ...].squeeze()
    data = np.moveaxis(data, 3, 1)
    negative = siamese.get_descriptors(data)

    dist_pos = euclidean_distance(anchor, positive)
    dist_pos_arr[current_start:end_batch] = dist_pos

    dist_neg = euclidean_distance(anchor, negative)
    dist_neg_arr[current_start:end_batch] = dist_neg

    # dist_collection_pos.append(np.expand_dims(dist_pos, axis=0))
    # dist_collection_neg.append(dist_neg)

    current_start += batch_size
# print dist_pos[0].shape
# dist_pos = np.vstack((dist_pos[0], dist_pos[1]))
# dist_pos = np.concatenate((dist_pos[0], dist_pos[1]), axis=0)
print(dist_pos_arr.shape)
# print 'pos', np.mean(dist_pos)
# print 'neg', np.mean(dist_neg)
# print np.mean(dist)
# print dist.shape
# print subt.shape
# print (anchor-positive).shape
# dist_pos = np.linalg.norm(anchor-positive)
# print dist_pos.shape

# outs = session.run(self.outputs, feed_dict=feed)
# dists = siamese.test_session.run([siamese.dist_pos, siamese.dist_neg], feed_dict=feed)
# siamese.get_descriptors(data)
# siamese.perform_test(data, None, siamese.test_session)

# print data.shape
# output = siamese.get_descriptors(data[:])
# print output.shape

fp_indeces = np.where(dist_neg_arr < dist_pos_arr)

# Accuracy
print('percentage of wrongly classified items', len(fp_indeces) / float(total_num))

siamese.finalize_data()

# store the distances
f = h5py.File(outputfile, 'w')
f.create_dataset('positive_dist', data=dist_pos_arr)
f.create_dataset('negative_dist', data=dist_neg_arr)
f.create_dataset('false_pos', data=np.array(fp_indeces))
f.close()
