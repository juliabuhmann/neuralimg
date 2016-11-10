#!/usr/bin/python

from scipy.misc import imresize

import numpy as np
import os
import shutil
import tempfile
from operator import itemgetter

import neuralimg.dataio as dataio
from neuralimg.training import siamese as si
from neuralimg.training import ml as ml
from neuralimg.crag import crag_utils as cu

from pycmc import *


""" Class to update CRAGs with hyperparameters and trained models """


OUTLIER_VALUE = 10.0


class CragCostManager(object):

    def __init__(self, path):
        """ Reads essential data from the CRAG for cost update """
        # Read node features, edge features and feature weights
        self.path = path
        self.crag, self.volumes, self.crag_solution, self.ebb, self.ein = cu.read_crag(path)
        self.store = Hdf5CragStore(path)

        self.nf = NodeFeatures(self.crag)
        try:
            self.store.retrieveNodeFeatures(self.crag, self.nf)
        except Exception:
            print('No node features found. Creating from scratch ...')

        self.ef = EdgeFeatures(self.crag)
        try:
            self.store.retrieveEdgeFeatures(self.crag, self.ef)
        except Exception:
            print('No edge features found. Creating from scratch ...')

        self.fw = FeatureWeights()
        try:
            self.store.retrieveFeatureWeights(self.fw)
        except Exception:
            print('No features weights found. Creating from scratch ...')

        # Read nodes and no assignment edges
        self.nodes = self.crag.nodes()
        edges = self.crag.edges()

        self.assign_nodes = [i for i in self.nodes if self.crag.type(i) == CragNodeType.AssignmentNode]
        self.no_assign_edges = [i for i in edges if self.crag.type(i) == CragEdgeType.NoAssignmentEdge]

    def _read_data_metadata(self, path):
        """ Reads the metadata from the dataset expected in the model """
        metadata = dataio.read_pickle(os.path.join(path, 'metadata.dat'))
        imh, imw = metadata['data_config']['height'], metadata['data_config']['width']
        padding = metadata['data_config']['padding']
        normalise = metadata['data_config']['normalise']
        return imh, imw, padding, normalise, metadata['clabels'], metadata['cpositions']

    def get_node_weights(self):
        """ Returns the merge score for the assignment nodes and the
        weights for the slice nodes """
        merge = self.fw.__getitem__(CragNodeType.AssignmentNode)
        weights = self.fw.__getitem__(CragNodeType.SliceNode)
        return merge, weights

    def get_edge_weights(self):
        """ Returns the end score for non assignment edges """
        return self.fw.__getitem__(CragEdgeType.NoAssignmentEdge)

    def get_feature_vector(self, node_id):
        """ Returns the feature vector for the given node (node or id) """
        return self.nf.__getitem__(self.crag.nodeFromId(node_id))

    def update_node_weights(self, merge_score, feature_size=128, weights=None):
        """ Updates weights for Slice Nodes and Assignment nodes
        Assignment nodes are assigned a unique weight with value of the merge score
        and slice nodes are assigned a weight of ones with length equal to the
        slice node feature size
        :param merge_score: Score ot be assigned to assigment nodes
        :param feature_size: Size of the slice node feature size.
        :param weights: List of weights to assign to the slice nodes.
            Sould have same size as the descriptors stored in the nodes.
            If set to None, it sets a list of 1s with size corresponding 
            to size of the node features
        """

        if feature_size is not None and weights is not None:
            raise ValueError('Feature size and weight list cannot be not None' +
                ' at the same time. Input either of them')

        # Update assignment node
        self.fw.__setitem__(CragNodeType.AssignmentNode, [merge_score])

        real_feature_size = self._get_slice_feature_size()
        weights = weights if weights is not None else [1] * feature_size
        if len(weights) != real_feature_size:
            print('Warning: Assigning a feature weight vector different from the ' +
                ' size of the feature size of the slice nodes. This will only ' + 
                ' work if slice node features are modified before saving ')
        # Update slice node
        self.fw.__setitem__(CragNodeType.SliceNode, weights)

    def _get_slice_feature_size(self):
        """ Returns the feature size of the first slice node to find """
        for i in self.crag.nodes():
            if self.crag.type(i) == CragNodeType.SliceNode:
                return len(self.nf.__getitem__(i))
        raise RuntimeError('Could not find a slice node in the crag')

    def update_edge_weights(self, end_score):
        """ Updates weights for no assignment edges with the
        input core
        :param end_score: End score for the no assignment edges.
        """
        self.fw.__setitem__(CragEdgeType.NoAssignmentEdge, [end_score])

    def update_node_features(self, model):
        """ Updates node features using the folowing rule:
                - Assignment nodes: updated with the norm of the L2 distance
                between its slices and a one: (|f1 - f2|^2, 1)
                - Slice nodes: updated with the output of the siamese network
            :param model: path to model folder to use for getting the node descriptors
        """
        if not os.path.isdir(model):
            raise ValueError('Folder for model does not exist')

        # Create triplet network
        net = si.TripletSiamese()
        net.initialize_test(model)

        # Read data metadata expected in the model
        imh, imw, pad, norm, names, pos = self._read_data_metadata(model)

        # We iterate over slice nodes and use those assignment nodes
        # for the next section. Note that we are computing redundant data
        # as a node can appear first and then appear as connected to another one
        for i, current in enumerate(self.nodes):

            if i % 100 == 0:
                print('Processing node %d' % i)

            # Care only about the slice nodes here
            if self.crag.type(current) != CragNodeType.SliceNode:
                continue

            # Get info form current node
            current_depth = cu.get_depth(current, self.volumes)

            # Get assignment nodes connected to current node
            ans = cu.get_connected_an(self.crag, current)

            # print('Slice node has assignment nodes assigned')
            # Node has assignments available
            nodes = []
            # Search for assignment nodes connecting to next section
            for an in ans:
                other = cu.get_other_slice(self.crag, current, an)
                if cu.get_depth(other, self.volumes) > current_depth:
                    nodes.append([other, an])

            # Get slice images and compute descriptors
            slices = [current] + [j[0] for j in nodes]
            slice_imgs = self._get_images(slices, imh, imw, pad, norm, names, pos)
            batch = _build_input_from_slices(slice_imgs, names, imh, imw)
            desc = net.get_descriptors(batch)

            # Assign descriptor for each slice
            for i in range(len(slices)):
                self.nf.__setitem__(slices[i], desc[i, ...].tolist())

            # Assign ratio to the assignment nodes, if exist
            if len(nodes) > 0:
                ratios = _get_ratios(desc[0], desc[1:], OUTLIER_VALUE)
                for ((_, an), r) in zip(nodes, ratios):
                    self.nf.__setitem__(an, [r])

            # Track length of the descriptor
            length = desc.shape[1]

        net.finalize_test()

        return length

    def update_edge_features(self, value):
        """ Updates edge featues with the input value
            :param value: Value to assign to all given edges
        """
        for n in self.no_assign_edges:
            self.ef.__setitem__(n, [value])

    def save(self):
        """ Updates the modified features/weights into the original CRAG file """
        self._validate()
        print('Crag validated: OK. Saving node features ---')
        self.store.saveNodeFeatures(self.crag, self.nf)
        print('Saving edge features ...')
        self.store.saveEdgeFeatures(self.crag, self.ef)
        print('Saving feature weights ...')
        self.store.saveFeatureWeights(self.fw)

    def _item_length(self, manager, inp):
        """ Returns the length of the element given by the manager (edge/feature/weight)
        and the given key """
        return len(manager.__getitem__(inp))

    def _validate(self):
        """ Validates that all node/edge feature and weights are consistent """

        def _validate_set(elem_set, set_manager, elem_name):
            for i in elem_set:
                feat_len = self._item_length(set_manager, i)
                weight_len = self._item_length(self.fw, self.crag.type(i))
                if feat_len != weight_len:
                    raise RuntimeError('Incompatible feature(' + str(feat_len) + ') and weight(' + 
                        str(weight_len) + ') for ' + elem_name + ' ' + str(self.crag.id(i)) +
                        ' of type ' + str(self.crag.type(i)))

        _validate_set(self.crag.nodes(), self.nf, 'node')
        _validate_set(self.crag.edges(), self.ef, 'edge')

    def _get_images(self, node_slices, imh, imw, padding, norm, labels, positions):
        """ Obtains the padded images corresponding to the input slice nodes """
        slice_imgs = []
        # Get volume for each slice
        slices = [self.volumes.getVolume(i) for i in node_slices]
        # Build slice image according to each channel
        for (i, s) in enumerate(slices):
            bb = cu.compute_bbs([s], padding, imh, imw, self.ein)[0]
            img = cu.build_image(s, self.ein, self.ebb, labels, positions, imh, imw, padding, norm)
            slice_imgs.append(img)
        return slice_imgs


def _build_input_from_slices(imgs, ch_names, imh, imw):
    """ Builds the network input from the buffer """
    batch = np.empty((len(imgs), len(ch_names), imh, imw))
    for i in range(len(imgs)):
        batch[i, ...] = imgs[i]
    return batch


def _get_ratios(ref_desc, other_desc, outlier_ratio):
    """ Given a reference descriptor and a list of connected slices, returns
    the score for each corresponding assignment node """
    # Update assignment feature as L2 norm
    l2_dists = [np.sqrt(sum(np.power(ref_desc - d, 2))) for d in other_desc]
    # print('L2 dists: {}'.format(l2_dists))
    # Compute edge for outliers
    out_edge = ml.compute_outlier_edge(l2_dists)
    # print('Outlier edge: {}'.format(out_edge))
    # Compute maximum from non outliers
    maxim = max([d for d in l2_dists if d <= out_edge])
    # Set outliers with maximum distance
    num_values = len(l2_dists)
    ratios = [0] * num_values
    for i in range(num_values):
        if l2_dists[i] > out_edge:
            ratios[i] = outlier_ratio
        else:
            ratios[i] = float(l2_dists[i]/maxim)
    # print('Ratios: {}'.format(ratios))
    return ratios 

