# Provide GT Ids with specific cell types, load a list that maps slice node ID to GT Id and get thus cell
# type labels for slice node IDs. Check whether the different cell types are reflected in the embedding.
from numpy import genfromtxt
from neuralimg.training import siamese as si
import os
from ml_utils import machine_learning_utils as ml
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, metrics)
from random import shuffle
import h5py
from skeleton import networkx_utils
import networkx as nx
from scipy import ndimage
import helpers


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_dir = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/221238/'
crag_path = '/raid/julia/documents/Dropbox/best_effort/heuristic_100sections/sampleA.hdf'
crag_path = None
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/sampleA.h5'
gt_to_slicenodeid = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/gtID_to_slicenodes.txt'

outputpathfig = '/raid/julia/documents/Dropbox/Doktorarbeit/2016/NIPS/presentations/tsne.png'


def plot_all_three(data):
    f, axarr = plt.subplots(3, 3)
    plt.axis('off')
    for number in range(3):
        axarr[number, 0].imshow(data[number, 0, :, :])
        axarr[number, 1].imshow(data[number, 1, :, :], cmap='gray')
        axarr[number, 2].imshow(data[number, 2, :, :], cmap='gray')

    # positive example
    axarr[0, 0].set_title('ANCHOR')
    # axarr[1, 0].set_title('Positive Example dist %0.4f' % pos_dist, color='green')
    # axarr[2, 0].set_title('Negative Example dist %0.4f' % neg_dist, color='red')
    plt.tight_layout()

    # remove the x and y ticks
    for ax in list(axarr.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])


def get_dict(gt_labels, slice_nodes):
    gt_to_slice_nodes = {}
    for gt_label, slice_node in zip(gt_labels, slice_nodes):
        gt_label = int(gt_label)
        slice_node= int(slice_node)
        if gt_label in gt_to_slice_nodes:
            gt_to_slice_nodes[gt_label].append(slice_node)
        else:
            gt_to_slice_nodes[gt_label] = [slice_node]
    return gt_to_slice_nodes


def get_activations_from_slice_nodes(siamese, slice_nodes, ref_ids, filehandle, feature_dim=128):
    activation_list = []
    for slice_node in slice_nodes:
        if slice_node in ref_ids:
            tripID = ref_ids.index(slice_node)
            data = filehandle['data'][tripID, 0, ...]

            # Required shape (?, channel, height, width)
            data = np.expand_dims(data, axis=0)
            activation = siamese.get_descriptors(data)
            activation_list.append(activation)
        else:
            print('slice node not in list', slice_node)
    return activation_list



my_data = genfromtxt(gt_to_slicenodeid, delimiter=' ')
slice_nodes = my_data[:, 0]
gt_labels = my_data[:, 2]

f_triplets = h5py.File(dataset, 'r')
ref_ids = list(f_triplets['ref_ids'].value)
ref_ids = [int(ref_id) for ref_id in ref_ids]
print(ref_ids)


siamese = si.TripletSiamese(dataset, crag_path)
siamese.initialize_test(model_dir)


gliacells = [20474, 5918, 187749]
normal_neurons = [4557, 11664, 9988, 11901]

gt_to_slicenodeid = get_dict(gt_labels, slice_nodes)


# for neuron in gliacells:
activations = []
number_of_items = 0

for neuron in gliacells[:1]:
    # slice_nodes = gt_to_slicenodeid[neuron][:30]
    slice_nodes = gt_to_slicenodeid[neuron]
    single_acitvations = get_activations_from_slice_nodes(siamese, slice_nodes, ref_ids, f_triplets)
    number_of_items += len(single_acitvations)
    activations.extend(single_acitvations)

print('loaded %d for glia cells' %number_of_items)
label_list1 = [0]*number_of_items

number_of_items = 0
for ii, neuron in enumerate(normal_neurons):
    slice_nodes = gt_to_slicenodeid[neuron]
    single_acitvations = get_activations_from_slice_nodes(siamese, slice_nodes, ref_ids, f_triplets)
    number_of_items += len(single_acitvations)
    activations.extend(single_acitvations)
    # label_list1.extend([ii+1]*len(single_acitvations))
activations = np.array(activations)[:, 0, :]
label_list2 = [1]*number_of_items
label_list1.extend(label_list2)
print('loaded %d for normal neurons ' %number_of_items)


assert len(label_list1) == activations.shape[0]

X = metrics.pairwise.pairwise_distances(activations, Y=None, metric='euclidean', n_jobs=1)
X = X + np.amin(X)+0.1

tsne = manifold.TSNE(n_components=2, init='random', random_state=0, metric='precomputed', n_iter=1000)
t0 = time()
X_tsne = tsne.fit_transform(X)
print('ready')

label_list = np.array(label_list1)
label_list_bin = label_list > 0
label_list_bin = label_list_bin.astype(np.int)

helpers.plot_embedding(X_tsne, label_list,
               "t-SNE embedding of membrane trained CNN activations (time %.2fs)" %
               (time() - t0))
plt.savefig(outputpathfig, format='png', dpi=1000)
plt.show()


