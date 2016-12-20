#!/usr/bin/python

import numpy as np
import os
import h5py
import tempfile
import random
import shutil
import abc

from neuralimg.training import ml
import neuralimg.dataio as dataio
from neuralimg.image import datasetconf as dc
from neuralimg.crag import crag_utils as cu

from pycmc import *

# Dataset generation class. This class generates datasets of pairs or triplets.
# Given a slice node in a section:
#   - Positive instaces are built using slice nodes in the next section 
#   connected through an assignment node and that are part of the best effort
#   - Negative instaces are built using slice nodes in the next section 
#   connected through an assignment node and that are NOT part of the best 
#   effort
#
# Image patches are extracted around the correspondign slice nodes and 3 
# possible image channels are selected: binary, raw and membrane prediction.
#
# Pair datasets are build using positive and negative instances.
# Triplet datasets are built using a reference slice node, its correspondance 
# slice node in the next section and the closests non-corresponding slice node 
# in the next section.


DATA_TAG = 'data'
LABEL_TAG = 'labels'
REF_TAG = 'ref_ids'
IND_TAG = 'indices'
BUF_SIZE = 512


class DataGenOptions(object):

    def __init__(self, path, init=None, end=None, exclude=[]):
        """ Define the inputs for data generation from a single CRAG
            :param path: Path to where CRAG is
            :param init: Initial section to consider (first section is 0). 
                Set to None for considering sections from the beggining
            :param end: Last section to consider. Set to None to consider
                sections until last one
            :param exclude: List of section to exclude from the interval
        """
        self.path = path
        self.init = init
        self.end = end
        self.exclude = exclude


class DatasetGen(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, config, outp):
        self.labels = None
        self.channel_map = None
        self.output_path = outp
        self.buf = None
        self.buf_l = None
        self.ref_buf = None

        # Read configuration file
        if not os.path.isfile(config):
            raise IOError('Config file not found')
        self.conf = dc.DatasetGenConf()
        self.conf.read(config)


    @abc.abstractmethod
    def _prepare_instances(self, crag, volumes, solution):
        """ Returns the set of instances depending on the subclass """

    @abc.abstractmethod
    def get_sections(self):
        """ Returns the number of sections per instance used in the subclass"""

    @abc.abstractmethod
    def store_label(self):
        """ Returns whether the label is needed in the subclass"""

    def generate_dataset(self, crags):
        """Generates data given the input CRAG and appends it to the existing data

        Parameters
        ----------
        crags: list of CragOptions
            List of CRAG characteristics
        """

        # Update data to append new instances
        self._initialize_data()

        # Initializes the output method
        self._init_output()

        # Create temporary file for outputing files 
        tmpath = tempfile.mkdtemp()

        for i in crags:
            self.generate_crag_data(i, tmpath)

        # Split into training, validation, testing
        if self.conf.sets is True:
            self._define_sets()

        self._finalize()

        # Clear temporary folder
        shutil.rmtree(tmpath)

    def generate_crag_data(self, current_crag, tmpath):
        """ Generates the data for the input crag dataset """

        print('Switching to CRAG {}'.format(current_crag.path))

        crag, volumes, solution, ebb, ein = cu.read_crag(current_crag.path)

        # Generate instances according to subclass
        total = list(self._prepare_instances(crag, volumes, solution))

        print('Detected %d instances in CRAG %s' % (len(total), current_crag.path))

        # Shuffle option as parameter
        if self.conf.sets is True:
            # Randomize so training, validation and testing are
            # not consecutive
            random.shuffle(total)

        for i, instance in enumerate(total):

            # Slices contain [anchor, positive, negative]
            slices, ref_id = instance[0]

            if not self._is_instance_excluded(slices, current_crag.init,
                current_crag.end, current_crag.exclude):

                if i % 100 == 0:
                    print('Generating instance {} out of {}'.format(str(i), 
                        str(len(total))))

                # Add current instance info to buffer
                label = instance[1] if self.store_label() is True else None
                self._add_to_buffer(ref_id, slices, volumes, ein, ebb, label)

                # Data is dumped into dataset when buffer is full
                if self.count % BUF_SIZE == 0:
                    self._dump_buffer(BUF_SIZE)

        print('Instance counter after reading %s is %d'
            % (current_crag.path, self.count))

    def _init_output(self):
        """ Prepares the storage of the dataset output """

        if os.path.isfile(self.output_path):
            raise IOError('File {} already exists'.format(self.output_path))
        if not dataio.valid_volume_path(self.output_path):
                raise ValueError('Given output path is not a valid HDF5 file')

        # Create HDF5. Remeber to close it at at the end
        self.h5_file = h5py.File(self.output_path, 'w')

        # Initialize data as empty and resize it when needed. Maximum size: none 
        data_shape = (0, self.get_sections(), self.dims, self.conf.height, self.conf.width)
        max_shape = (None, self.get_sections(), self.dims, self.conf.height, self.conf.width)

        # Create group for data, labels, refs and channel maps
        self.h5_file.create_dataset(DATA_TAG, data_shape, compression='gzip', chunks=True, maxshape=max_shape)
        self.h5_file.create_dataset(REF_TAG, (0,), compression='gzip', chunks=True, maxshape=(None, ))
        if self.store_label() is True:
            self.h5_file.create_dataset(LABEL_TAG, (0,), compression='gzip', maxshape=(None,), chunks=True)
        labels_type = h5py.special_dtype(vlen=str)
        labels_data = np.asarray(list(self.channel_map.keys()), dtype=object)
        self.h5_file.create_dataset('clabels', data=labels_data, dtype=labels_type)
        self.h5_file.create_dataset('cpositions', data=np.asarray(list(self.channel_map.values())))

        # Store metadata in separate dataset
        self.h5_file.attrs.create('height', data=self.conf.height)
        self.h5_file.attrs.create('width', data=self.conf.width)
        self.h5_file.attrs.create('padding', data=self.conf.padding)
        self.h5_file.attrs.create('normalise', data=self.conf.normalise)

    def _finalize(self):
        """ Makes all actions needed before finishing """
        # Store data that is left
        if self.count % BUF_SIZE != 0:
            print('Data still need to be saved. Saving ...')
            read = int(self.count/BUF_SIZE) * BUF_SIZE
            self._dump_buffer(self.count - read)
        # Close output file
        print('Closing output file ...')
        self.h5_file.close()

    def _dump_buffer(self, num):
        """ Dumps a subset (buffer) of the dataset into the given group """
        # Resize dataset
        self._resize_data(num)
        # Append new data
        self.h5_file[DATA_TAG][self.count - num:self.count, ...] = self.buf[0:num]
        self.h5_file[REF_TAG][self.count - num:self.count, ...] = self.ref_buf[0:num]
        if self.store_label() is True:
            self.h5_file[LABEL_TAG][self.count - num:self.count] = self.buf_l[0:num]

    def _initialize_data(self):
        """ Initializes the arrays containing the dataset"""

        self.dims = self.conf.binary + self.conf.raw + self.conf.intensity
        if self.dims == 0:
            raise ValueError('At least one channel must be activated')

        # Initialize empty array
        self.count = 0
        self.buf = np.empty([BUF_SIZE, self.get_sections(), self.dims, 
           self.conf.height, self.conf.width])

        # Initialize id buffer
        self.ref_buf = np.empty([BUF_SIZE])

        # Initialize empty label buffer, if needed
        if self.store_label() is True:
            self.buf_l = np.empty([BUF_SIZE]) 

        # Compute feature map if it does not exist
        # All new sets appended will be aligned to first
        self._map_channels()

    def _resize_data(self, num):
        """ Resizes HDF5 file to cope with more data """
        current = self.h5_file[DATA_TAG].shape[0]
        self.h5_file[DATA_TAG].resize(current + num, axis=0)
        self.h5_file[REF_TAG].resize(current + num, axis=0)
        if self.store_label() is True:
            self.h5_file[LABEL_TAG].resize(current + num, axis=0)

    def _map_channels(self):
        """ Maps each channel in ascending order """
        channels = [cu.DChannel.BINARY, cu.DChannel.RAW, cu.DChannel.INTENSITY]
        activated = [self.conf.binary, self.conf.raw,  self.conf.intensity]
        chosen = [i for i in range(0, len(activated)) if activated[i] == 1]
        self.channel_map = {channels[chosen[i]]: i for i in range(0, len(chosen)) }

    def _add_to_buffer(self, ref_id, slices, volumes, ein, ebb, label):
        """ Add the input image sections into the buffer """
        buf_pos = self.count % BUF_SIZE
        imh, imw = self.buf.shape[3], self.buf.shape[4]
        clabels = list(self.channel_map.keys())
        cpositions = list(self.channel_map.values())

        # print(imh, imw, clabels, cpositions)
        # print(self.conf.padding, self.conf.normalise)

        # Insert data buffer information
        for i in range(len(slices)):
            self.buf[buf_pos, i, ...] = cu.build_image(slices[i], ein, ebb,
                clabels, cpositions, imh, imw, self.conf.padding, self.conf.normalise)

        # Insert label, if provided
        if label is not None:
            self._add_label(label)

        # Store reference node id
        self._add_node_ref(ref_id)

        # Add buffer counter
        self.count += 1



    def _add_label(self, label):
        """ Adds a label into the buffer """
        self.buf_l[self.count % BUF_SIZE] = label

    def _add_node_ref(self, ref_id):
        """ Adds the node id of the reference slice of the current instance """
        self.ref_buf[self.count % BUF_SIZE] = ref_id

    def _define_sets(self):
        """ Splits the data into training, validation and testing
        according to the given configuration """

        print('Splitting into training, validation and testing ...')

        # If labels do not matter, split randomly
        if self.store_label() is False:
            self.conf.mode = 'random'
        labels = self.h5_file[LABEL_TAG][:] \
            if self.store_label() is True else self.h5_file[DATA_TAG].shape[0]

        train, val, test = ml.split(labels, self.conf.ratioT,
            self.conf.ratioV, self.conf.mode, shuffle = True)

        # Store indices in output
        ind_group = self.h5_file.create_group(IND_TAG)
        ind_group.create_dataset('training', data=train, compression='gzip')
        ind_group.create_dataset('validation', data=val, compression='gzip')
        ind_group.create_dataset('testing', data=test, compression='gzip')

    def _is_instance_excluded(self, slices, init, end, excluded):
        """ Returns whether any of the sections in the input instance is linked
        to any of the sections to exclude """
        for i in slices:
            depth = cu.get_slice_depth(i)
            # Breaks lower bound interval
            if init is not None and (depth < init):
                return True
            # Breaks upper bound interval:
            if end is not None and (depth > end):
                return True
            # Check if excluded from banned sections
            if depth in excluded:
                return True
        return False


class PairDataGen(DatasetGen):

    def __init__(self, config, outp):
        DatasetGen.__init__(self, config, outp)

    def get_sections(self):
        return 2

    def store_label(self):
        return True

    def _prepare_instances(self, crag, volumes, solution):
        """ Splits assignment nodes into positive (if belong to CRAG solution)  and
         negative examples. If shuffle is True, examples are taken randomly from the
         CRAG (not in order). Pairs of node-label are returned """

        # Read assignment nodes
        nodes_positive = []
        nodes_negative = []
        # Gather positive and negative examples
        for n in crag.nodes():
            if crag.type(n) == CragNodeType.AssignmentNode:
                nodes = cu.get_opposite(crag, n)
                ref_id = crag.id(nodes[0])
                slices = cu.get_slices(volumes, nodes)
                if solution.selected(n):
                    nodes_positive.append([slices, ref_id])
                else:
                    nodes_negative.append([slices, ref_id])

        # Add class label
        zipped_pos = zip(nodes_positive, [1] * len(nodes_positive))
        zipped_neg = zip(nodes_negative, [0] * len(nodes_negative))

        if self.conf.balance is True:
            zipped_pos, zipped_neg = balance_classes(zipped_pos, zipped_neg)

        print('Number of positive examples: {}'.format(len(list(zipped_pos))))
        print('Number of negative examples: {}'.format(len(list(zipped_neg))))

        return zipped_pos + zipped_neg


class TripletDataGen(DatasetGen):

    def __init__(self, config, outp):
        DatasetGen.__init__(self, config, outp)

    def get_sections(self):
        return 3

    def store_label(self):
        return False

    def _prepare_instances(self, crag, volumes, solution):
        """ For each assignment node from the solution, searches for
        an assignment node out of the solution that contains any of the slices
        connected to the first node """

        # Read assignment nodes
        instances = []
        rank_collection = []
        pos_was_not_included = 0
        for n in crag.nodes():
            if crag.type(n) == CragNodeType.AssignmentNode:

                if solution.selected(n):

                    # Get positive slices and negative possible slices
                    n1, n2 = cu.get_opposite(crag, n)
                    slice1, slice2 = cu.get_slices(volumes, [n1, n2])
                    neg_n1 = cu.get_nassigned_nodes(crag, volumes, solution, n1)

                    ref_center = get_center(slice1)
                    dist_of_pos = np.sqrt(sum((ref_center-get_center(slice2))**2))

                    # Build instance only if a negative assignment for first has 
                    # been found. Among all, choose closets in space
                    # print neg_n1, 'neg_n1'
                    if len(neg_n1) > 0:
                        third, dists = get_closest(slice1, neg_n1, return_dists=True)
                        neg_dist = np.sqrt(sum((ref_center-get_center(third))**2))
                        instances.append([[slice1, slice2, third], crag.id(n1)])
                        # print self.conf.rank
                        if self.conf.rank > 1:
                            indeces = sorted(range(len(dists)), key=lambda x: dists[x])
                            assert indeces[0] == dists.index(min(dists))
                            assert indeces[0] == dists.index(neg_dist)

                            number_of_nodes_to_comp = min(self.conf.rank, len(neg_n1))
                            for rank in range(1, number_of_nodes_to_comp):

                                third = neg_n1[indeces[rank]]
                                instances.append([[slice1, slice2, third], crag.id(n1)])

                        dists = sorted(dists)

                        was_inside = False
                        rank = None
                        for ii, neg_dist in enumerate(dists):
                            if dist_of_pos <= neg_dist:
                                was_inside = True
                                rank = ii
                                break
                        if was_inside:
                            rank_collection.append(rank)
                        else:
                            pos_was_not_included += 1

        print('mean rank', np.mean(rank_collection), 'average not included',
              pos_was_not_included / float((len(rank_collection) + pos_was_not_included)))
        print('in %0.3f percent was amongst top 3' %(np.sum(np.array(rank_collection)<3)/float(len(rank_collection))))
        print('in %0.3f percent was amongst top 5' %(np.sum(np.array(rank_collection)<5)/float(len(rank_collection))))
        # Add label, though it is not used
        return zip(instances, )


def get_closest(ref, others, return_dists=False):
    """ Retrieves node closests to the reference node according to L2 distance"""
    ref_center = get_center(ref)
    dists = [np.sqrt(sum((ref_center-get_center(i))**2)) for i in others]
    if return_dists:
        return others[dists.index(min(dists))], dists
    else:
        return others[dists.index(min(dists))]


def balance_classes(positive, negative):
    """ Balances positive and negative classes so the remaining data contains
    the same number of classes per each class """
    least = min(len(positive), len(negative))
    return positive[0:least], negative[0:least]


def get_center(vol):
    """ Computes the center of the bounding box """
    bb = vol.getBoundingBox()
    center = np.empty((2))
    for i in range(2):
        size = bb.max().__getitem__(i) - bb.min().__getitem__(i)
        center[i] = bb.min().__getitem__(i) + np.floor(size/2.0)
    return center

