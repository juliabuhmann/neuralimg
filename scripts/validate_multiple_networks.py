# ! /usr/bin/python
from bokeh.io import curdoc

from neuralimg.training import siamese as si
import os
from ml_utils import machine_learning_utils as ml
import numpy as np
import h5py
import time
from neuralimg.evaluation import helpers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/triplets_rank3/sampleA.h5'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank3/sampleA.h5'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/triplets/sampleB.h5'

# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/smallMLPmodel/'

# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/221238/'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/sampleArank3model_II/distances_sampleA.h5'

# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/20161206/tensorboard/213915/'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/allSamplesrank3model/distances_sampleA_training.h5'

# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model/221238/'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/sampleArank3model/distances_sampleA_training.h5'

# Retrained model with more rank
model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_III/221238/'
outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/sampleB/sampleBrank3.h5'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank5models/rank3contsampleA/sampleA.h5'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank5models/rank3contsampleA/sampleA_training.h5'

# Retrained model with early strop (rank5)
# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_III/backupstep/221238/'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank5models/rank3contsampleA/sampleA_earlystop.h5'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank5models/rank3contsampleA/sampleA_training.h5'

# Model with only rank1
# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/20161201/tensorboard/173928/'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank5models/rank3contsampleA/sampleArank1.h5'
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/sampleB/sampleBrank1.h5'

# Model with activations (but with unit normalization)
# model_dir = '/raid/julia/projects/fragment_embedding/nn_training/20161128/tensorboard/192507/'

CALCULATE_DISTANCE = True
PRINT_DISTANCE = True


if CALCULATE_DISTANCE:
    crag_path = None
    datasetname = 'data'
    if not datasetname == 'data':
        siamese = si.TripletSiamese(dataset, crag_path, testdataname='test')
    else:
        siamese = si.TripletSiamese(dataset, crag_path)


    siamese.initialize_test(model_dir)
    total_num = siamese.data[datasetname].shape[0]
    # if total_num > 20000:
    #     total_num = 20000
    print(total_num)

    def euclidean_distance(m1, m2):
        subt = np.subtract(m1, m2)
        squared = (subt) ** 2
        dist = np.sqrt(np.sum(squared, axis=1))
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



        current_start += batch_size


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

if PRINT_DISTANCE:
    print(model_dir)
    print(dataset)
    print(outputfile)
    helpers.print_distance_statistics(dataset, outputfile)