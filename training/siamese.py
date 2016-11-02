#!/usr/bin/python

from neuralimg.crag import crag_utils as cu
from neuralimg.training import net_utils as nu
from neuralimg import dataio

import numpy as np
import sys
import os
import abc
import tensorflow as tf
import h5py
import time
import datetime as dt
import matplotlib.pyplot as plt
import tempfile

# This class trains and tests a Siamese network that can be trained with pairs or triplets
# Definition in the network is provided in a configuration file including the parameters.
# Some minor parameters such as the stride in the convolutional layer and the size and stride of
# the max pooling have been hardcoded for simplicity, but it can be easily extended in the code


class NetworkMode(object):

    LOSS = 'loss'
    OUTPUT = 'output'


class SiameseNetwork(object):

    DATA_AUGM = 8
    FLIPS = 2
    CONV_LAB = 'conv'
    FULLY_LAB = 'fullyc'

    __metaclass__ = abc.ABCMeta

    def __init__(self, path=None, crag_path=None):
        """ Initializes a SiameseNetwork
            :param path: Path to HDF5 dataset. In case model needs to be loaded
                and test data to be provided, can be set to None
            :param crag_path: Path to CRAG to consider for mean rank computation
                considering related sections.
        """
        self.data_path, self.data, self.crag_path = path, None, crag_path
        self.crag, self.volumes, self.solution = None, None, None
        self.conf, self.scope_name = None, 'siamese'
        # Loss and rank  tracking
        self.mean_rank, self.rank_name = [], 'ranks.dat'
        self.train_loss, self.val_loss, self.loss_name = [], [], 'loss.dat'
        # Params
        self.data_augm_enabled, self.config = False, None
        # Indices
        self.train_ind, self.val_ind, self.test_ind = None, None, None
        # Writing variables
        self.summary_op, self.sum_writer = None, None
        # Image and batch params
        self.batch_size, self.batch_take = None, None
        self.imgs, self.depth, self.imh, self.imw = [None] * 4
        # Placeholders variables
        self.pls, self.pl_names, self.pl_labels, self.pl_dropout = [None] * 4
        # Steps accounts for training step only
        self.step, self.pl_aug = None, None
        self.layers_out = []
        # Metadata from dataset
        self.metadata = {}
        # Test attributes
        self.test_session = None

        if self.data_path is not None:
            print('Reading data ...')
            self.read_data()
        if self.crag_path is not None:
            print('Reading CRAG file ...')
            self.read_crag()

    @abc.abstractmethod
    def get_placeholder_suffixes_specific(self):
        """ Returns the placeholders according to the subclass """

    @abc.abstractmethod
    def get_specific_loss(self, outputs, labels):
        """ Defines the specific loss function to use considering all branch outputs"""

    @abc.abstractmethod
    def get_label_batch(self, inds):
        """ Returns the labels for the corresponding boolean mask """

    @abc.abstractmethod
    def get_positive_instances(self, inds):
        """ From the input instance indeces returns the positive ones from the subclass """

    ###################################
    ######## DATA & CONFIG FUNCTIONS
    ###################################

    def check_options(self, path):
        """ Parser the network options
            :param path: Path where configuration is stored
        """
        self.config = nu.read_conf(path)

        # Store 2 batch sizes for data augmentation. Data augmented
        # option is the number of images d to use to generate versions.
        # So 8 x d images in the batch correspond to 'replicas'.
        self.batch_size = self.config.batch_s
        self.batch_take = self.config.batch_s
        if self.config.augm is not None:
            if self.config.augm * self.DATA_AUGM > self.config.batch_s:
                print('Instances augmented cannot be longer than batch size')
            self.batch_take = self.batch_size - (self.DATA_AUGM * self.config.augm)
            self.data_augm_enabled = True

    def read_data(self):
        """ Reads the data characteristics from the dataset provided  """
        # Open data to read some stats
        f = h5py.File(self.data_path, 'r')
        size = f['data'].shape[:]

        # Read dimensions
        if len(size) != 5:
            raise ValueError('Data must have the form N x D x C x H x W')
        if size[1] != len(self.get_placeholder_suffixes_specific()):
            raise ValueError('Depth {} is not compatible with network type'.format(size[1]))
        self.imgs, _, self.depth, self.imh, self.imw = size

        # Read indices
        self.train_ind = f['indices']['training'][:]
        self.val_ind = f['indices']['validation'][:]
        self.test_ind = f['indices']['testing'][:]

        # Read references
        self.ref_ids = f['ref_ids'][:]

        # Store channel map and configuration
        data_config_dict = {k: f.attrs.values()[i]
                            for (i, k) in enumerate(f.attrs.keys())}
        self.metadata = {'clabels': f['clabels'][:], 'cpositions': f['cpositions'][:], 
		'data_config': data_config_dict}

        f.close()

    def read_crag(self):
        """ Reads CRAG file into the class """
        self.crag, self.volumes, self.solution, self.ein, self.ebb = cu.read_crag(self.crag_path)

    def _load_characteristics(self, model_path):
        """ Load data characteristics from a trained model
        :param model_path: Folder where model is stored
        """
        meta = nu.load_model_data(model_path)
        self.depth = len(meta['clabels'])
        self.imh = meta['data_config']['height']
        self.imw = meta['data_config']['width']
        self.metadata = meta

    def data_aug_enabled(self):
        """ Returns whether data augmentation has been enabled """
        return self.data_augm_enabled is True

    def define_placeholders(self, mode=NetworkMode.LOSS):
        """ Initializes all placeholders needed in the network """
        self.pls, self.pl_names = self.get_data_placeholders(mode) # Data
        self.pl_labels = tf.placeholder(tf.float32, shape=[None], name='labels')
        self.pl_dropout = tf.placeholder(tf.float32, name='keep_probability') 
        self.pl_aug = tf.placeholder(tf.int32, name='augment_instances', shape=[None])

    def get_data_placeholders(self, mode):
        """ Returns the input placeholders/constants for the data """
        pl_names = self.get_placeholder_suffixes(mode)
        training = [tf.placeholder(tf.float32, shape=(None, self.imh, self.imw,
                                                      self.depth), name=i) for i in pl_names]
        return training, pl_names

    def get_placeholder_suffixes(self, mode):
        """ Define input branches and their names according to execution
        mode. Note that for testing and getting descriptors it does not matter
        which subclass the network belongs to"""
        if mode == NetworkMode.LOSS:
            return self.get_placeholder_suffixes_specific()
        elif mode == NetworkMode.OUTPUT:
            return ['input']
        else:
            raise ValueError('Unknown mode %s' % mode)

    def augment_data(self, data, prefix):
        """ Generates the remaining instances of the batch by taking the
        rotations: 0, 90, 180 and 270 and their respective vertical and
        horizontal mirrorings """
        with tf.variable_scope("data_augmentation" + prefix):
            insts = self.pl_aug
            # Operations to be performed image by image (Tensorflow restrictions)
            for i in range(self.config.augm):
                # Consider all rotations of the selected random instance
                index = tf.slice(insts, [i], [1])
                augm = tf.squeeze(tf.gather(data, index, validate_indices=False))
                r90 = tf.image.transpose_image(augm)
                r180 = tf.image.flip_up_down(tf.image.flip_left_right(augm))
                r270 = tf.image.flip_up_down(tf.image.flip_left_right(r90))
                rotations = [augm, r90, r180, r270]

                # Flip up and down for each rotation and append
                for j in rotations:
                    tf.image.flip_up_down(j)
                    data = tf.concat(0, [data, 
                                         tf.expand_dims(tf.image.flip_up_down(j), 0)])
                    data = tf.concat(0, [data, 
                                         tf.expand_dims(tf.image.flip_left_right(j), 0)])

        return data

    def initialize_data(self, train=True):
        """ Data initialization: open dataset """
        self.data = h5py.File(self.data_path, 'r')
        # Check data augmentation for training and validation
        if train is True:
            if self.depth != 3 and self.data_aug_enabled():
                raise ValueError('Data augmentation on the fly within the Tensorflow graph ' +
                       ', 3-dimensional images are needed. Otherwise, numpy manual ' +
                       'operations must be implemented')

    def finalize_data(self):
        """ Do all tasks to perform at the end, such as closing the data file """
        self.data.close()


    ###################################
    ######## TRAINING FUNCTIONS
    ###################################

    def compute_training_loss(self, tr_data, tr_labels, session):
        """ Returns the training loss for the given data and the time taken
         and performs a training step """
        start_time = time.time()
        lt, lt2, next_step = \
            self.perform_training(tr_data, tr_labels, session)
        duration = time.time() - start_time
        return [lt, lt2], duration, next_step

    def compute_validation_loss(self, session):
        """ Returns the validation loss and the time taken """
        start_time = time.time()
        vloss = self.perform_validation(session)
        duration = time.time() - start_time
        return vloss, duration

    def train(self, outp, logs, config_path, retrain=True, alexnet=None, alex_net_layers=3):
        """ Buils the network given the read data and starts training
        To update an existing model in the folder, set configuration file
        path to None
        Parameters
        ----------
        outp: string
            Where to store the resulting model and the checkpoints and path
            where to load the model from, if already trained.
        logs: string
            Path where to store the logs of the process,
        config_path: string
            Configuration file path. If a configuration is found in the model folder (pretrained),
            then this is omitted (can be set tot None)
        retrain: boolean
            If a pretrained model exists in the given folder, whether to retrain model (True)
            or start from scratch (False)
        alexnet: string
            Path where Alexnet weights are. Set to None to disable loading from Alexnet weights
        alex_net_layers: int
            Number of layers, starting from the first, to load from Alexnet weights
        """
        self.initialize_data()
        dataio.create_dir(logs)

        if nu.exists_conf(outp):
            # Load configuration from stored model
            self.check_options(nu.load_configuration(outp))
        else:
            if config_path is None:
                raise ValueError('If a configuration is not found in the ' +
                    'folder model, one must be provided')
            # Read configurations, create output folder
            self.check_options(config_path)
            dataio.create_dir(outp)
            # If training from scratch copy config in out folder
            self.config.write(os.path.join(outp, 'config.conf'))

        g, saver = self._initialize_graph(NetworkMode.LOSS)

        with tf.Session(graph=g, config=self.configure_sess()) as session:

            # Initialize variables or load them from checkpoint
            check_path = tf.train.latest_checkpoint(outp)
            if check_path is None or retrain is False:
                print('Initializing network ...')
                tf.initialize_all_variables().run()
                # If provided, load alexnet weights for first layers
                if alexnet is not None:
                    print('Loading Alexnet weights for %d layers' % alex_net_layers)
                    self.load_alexnet(alexnet, alex_net_layers, session)
            else:
                print('Restoring network from: {}'.format(check_path))
                self.load_net_state(outp, saver, session, model_file=check_path)

            # Get initial step
            init_step = self.get_step(session) + 1
            print('Initial step is {}'.format(init_step))

            # Prepare summaries for Tensorboard visualization
            # session.graph_def gives a deprecated warning but is needed in version 0.8.0
            self.summary_op = tf.merge_all_summaries()
            self.sum_writer = tf.train.SummaryWriter(logs, session.graph_def)

            # Train until maximum step is reached
            step = init_step
            while step <= self.config.steps:
                s = self.training_step(session, step, saver, outp)
                step = s

            print('Storing final model ...')
            self.store_net_state(session, saver, step, outp, 'final')

        print('Reached maximum step!')
        self.finalize_data()

    def training_step(self, session, step, saver, outp):
        """ Performs a single training step """
        tr_data, tr_labels = self.get_batch_trainstep(step)

        # DEBUG
        m = [tr_data[0][i, ...].mean() for i in range(tr_data[0].shape[0])] 
        print('Training data means: {}'.format(m))

        # Training data - Update gradients
        train_loss, timet, next_step = \
            self.compute_training_loss(tr_data, tr_labels, session)

        # Print losses (train + validation)
        if step % self.config.loss_int == 0:

            # Validation - Check loss evolution without updating gradients
            val_loss, timev = self.compute_validation_loss(session)

            assert not np.isnan(val_loss[0])
            # If loss is Nan, probably something is wrong

            # Print, save losses and write plot
            self.print_loss(train_loss, step, 'training', timet)
            self.print_loss(val_loss, step, 'validation', timev)
            self.append_losses(step, train_loss, val_loss)
            self.display_loss(os.path.join(outp, 'loss.png'))

            # Early stop
            if self.config.early is not None \
                    and self.track_validation(step, val_loss[0], session, saver, outp) is True:
                print('Early stopping: Validation loss has not improved' +
                      ' for {} steps. Last best model has been stored'.format(self.config.early))
                self.finalize_data()
                sys.exit()

            # Track mean rank from testing
            self._track_test_rank(session, step, crag=self.crag_path)

        # Save state of network for visualization
        if step % self.config.summary_int == 0:
            print('Storing summary ...')
            self.store_summary(tr_data, tr_labels, step, session)
            self.sum_writer.flush()

        # Save checkpoint of the model
        if step % self.config.checkpoint_int == 0:
            self.store_net_state(session, saver, step, outp, 'checkp')

        return next_step

    def perform_training(self, batch_data, batch_labels, session):
        """ Trains the network with the input batch and returns the loss """
        feed = self.build_feed(batch_data, batch_labels, dropout=True, augm=True)
        r, l, l2, s = session.run([self.optimizer, self.loss, self.l2, self.step], feed_dict=feed)
        return l, l2, int(s) + 1

    def perform_validation(self, session):
        """Get predictions for the validation set by taking a random subset """
        valid_instances = self.config.valid_batch - self.DATA_AUGM * self.config.augm
        data, labels = self.get_validation_data(valid_instances)
        feed = self.build_feed(data, labels, dropout=False, augm=True)
        l = session.run([self.loss, self.l2], feed_dict=feed)
        return l

    def load_alexnet(self, alexnet_path, layers, session):
        """ Assigns AlexNet weights to the given number of convolutional layers """
        ours = self._define_weights()
        alexnet = np.load(alexnet_path)
        keys = sorted(alexnet.keys())
        for i, k in enumerate(keys):
            if i == layers:
                break
            pointer = i / 2
            extra = i % 2
            session.run(ours[pointer][extra].assign(alexnet[k]))

    def get_train_size(self):
        return len(self.train_ind)

    def get_batch_train(self, start, end):
        """ Returns training data in the given interval of indices (last no included] """
        return self.get_data(self.train_ind[start:end])

    def get_validation_data(self, subset=None):
        """ Returns the validation data according to given size.
        If size is None, then returns whole set """
        inds = self.val_ind if subset is None \
            else nu.get_random(subset, len(self.val_ind))
        return self.get_data(inds)

    def track_validation(self, step, new_value, session, saver, outp):
        """ Tracks the window of validation losses and checks whether the
        loss value has increased for the last steps or whether it has decreased
        less than a minimum amount. Returns whether training should stop
        according to an early stopping criteria"""
        print('Checking early stop ...')
        if len(self.val_loss) < self.config.early:
            print('Need at least {} values'.format(self.config.early))
            return False
        if self.val_loss[(step % self.config.loss_int) - self.config.early][0] \
                <= (new_value + self.config.min_decrease):
            return True
        else:
            nu.store_checkpoint(session, saver, step, outp, self.metadata, 'early')
            return False

    def get_batch_trainstep(self, step):
        """ Returns a batch from the training data according to the current step"""
        offset = (step * self.batch_take) % (self.get_train_size() - self.batch_take)
        print ('%s: step %d (%d-%d) (%d data, %d augmented)'
               % (dt.datetime.now(), step, offset, offset + self.batch_take - 1,
                  self.batch_take, self.batch_size - self.batch_take))
        return self.get_batch_train(offset, offset + self.batch_take)

    ###################################
    ######## TEST FUNCTIONS
    ###################################

    def initialize_test(self, folder):
        """ Initializes the test environment.
        IMPORTANT: When testing is over, must call finalize_test method """
        # Read options and graph. Data needs to be closed as well
        if self.data_path is not None:
            self.initialize_data(train=False)

        # Load data information
        self. _load_characteristics(folder)

        # Load model configuration and graph
        self.check_options(nu.load_configuration(folder))
        g, saver = self._initialize_graph(NetworkMode.OUTPUT)
        # Initialize session: NEEDS TO BE CLOSED
        #self.test_session = tf.Session(graph=g, config=self.configure_sess())
        self.test_session = tf.InteractiveSession(graph=g, config=self.configure_sess())
        #with tf.Session(graph=g, config=self.configure_sess()) as session:
        # Load model and writer
        #self.load_net_state(folder, saver, self.test_session)
        self.load_net_state(folder, saver, self.test_session)
        print('Test initialized properly!')
        print('Warning: Make sure the data fed follows the same characteristics' +
              ' as the data used for training (e.g. normalization) or results may not be good')

    def finalize_test(self):
        """ Releases resources used for testing"""
        if self.data_path is not None:
            self.finalize_data()
        self.test_session.close()
        print('Test finalized properly!')

    def compute_ranks(self, dist, data, selected):
        """ Computes the rank score for the reference instance and its
        corresponding row for the input data
        Parameters
        ----------
        dist: pointer to function
            Distance to use for ranking the test results.
        data: ndarray
            Batch data following these properties:
                - Reference instance is the first in the array
                - Positive matches are right after the reference instance
                - The other positions are non-matching patches
        selected: int
            Number of positive matches in the input data
        """

        if self.test_session is None:
            raise Exception('Test procedure has not been initialized. ' +
                            'Must call initialize_test() before ')
        return self._query_ranks(eval_data, session, dist, 1)[0]

    def get_descriptors(self, data, labels=None):
        """ Returns the descriptors for the input batch and the given trained model
        Parameters
        ----------
        data: ndarray
            Descriptor data
        labels: ndarray
            Labels from data. If network does not require labels (triplet)
            set to None
        """

        if self.test_session is None:
            raise Exception('Test procedure has not been initialized. ' +
                            'Must call initialize_test() before ')

        inp, labs = nu.to_output_data(data, labels)
        return self.perform_test(inp, labs, self.test_session)

    def build_evaluation_from_test(self, test_ind, ind, num):
        """ Given a test set, returns rows so first instance is the
        one from the given index, second row is the matching patch and
        the rest are non-corresponding patches from other test
        instance patch matches
        :param test_ind: Test indices from total data
        :param ind: Instance to generate data from
        :param num: Number of non-corresponding instances to use
        """
        # Length is 2 (patch + match) + num non corresponding patches
        ref_instance, _ = self.get_data(ind)
        # Created vector of size (num x sections x h x w x channels)
        branches = len(self.get_placeholder_suffixes_specific())
        # Set first and second row as reference and match
        slices = [ref_instance[0, 0, ...], ref_instance[0, 1, ...]]
        # Get positions in test indices belonging to
        # non-corresponding examples
        inds = nu.excluding_permutation(ind, test_ind)[0:num]
        for (i, index) in enumerate(inds):
            # Extract 'match position' from current instance
            inst = np.squeeze(self.get_data(index)[0])[1, ...]
            slices.append(inst)
        return nu._build_matrix(slices, branches)

    def build_evaluation_from_crag(self, index):
        """ Returns the data matrix to input into the network corresponding
        to the reference section from the given instance (index) as the
        first row and the rest are sections connected to the reference one 
        through an assignmnet node in the CRAG """

        def get_image(node):
            """ Given a CRAG slice node, returns the corresponding image """
            return cu.build_image(node, self.ein, self.ebb, self.metadata['clabels'], 
                self.metadata['cpositions'], self.imh, self.imw, 
                self.metadata['data_config']['padding'], self.metadata['data_config']['normalise'])

        ref_instance = np.squeeze(self.get_data(index)[0])[1, ...]
        other = cu.get_connected_slices(self.crag, self.volumes, 
            self.solution, self.ref_ids[index], mode='forward')
        # Build ordered list of slices in the final matrix
        branches = len(self.get_placeholder_suffixes_specific())
        # First, add rows in solution and then the rest
        true_pos = [np.moveaxis(get_image(pair[0]), 0, 2) \
            for (i, pair) in enumerate(other) if pair[1] == True]
        false_pos = [np.moveaxis(get_image(pair[0]), 0, 2) \
            for (i, pair) in enumerate(other) if pair[1] == False]
        all_slices = [ref_instance] + true_pos + false_pos
        return nu._build_matrix(all_slices, branches), len(true_pos)

    def get_test_loss(self):
        """ Returns the loss for the test data given the loaded model """

        if self.test_session is None:
            raise Exception('Test procedure has not been initialized. ' +
                            'Must call initialize_test() before ')

        # Iterate through test using small batch
        total_test = len(self.test_ind)
        loss = []
        for i in range(0, total_test, self.batch_size):
            # Save loss for current batch
            num = self.batch_size if i + self.batch_size <= total_test \
                else total_test - i
            inds = self.test_ind[i:i + num]
            data, labels = self.get_data(inds)
            l = self.perform_test_loss(data, labels, self.test_session)
            loss.append(l[0])
            print('Loss for batch (%d-%d) is %f' % (i, i + num, l))

        return loss

    def perform_test(self, data, labels, session):
        """ Feeds forward the input test data """
        feed = self.build_feed(data, labels, dropout=False, augm=False)
        outs = session.run(self.outputs, feed_dict=feed)
        return outs[0]

    def perform_test_loss(self, data, labels, session):
        """ Feeds forward the input test data """
        feed = self.build_feed(data, labels, dropout=False, augm=False)
        l = session.run([self.loss, self.l2], feed_dict=feed)
        return l

    def _query_ranks(self, data, session, dist=nu.l2, selected=1):
        """ Queries the rank of the given input data
        :param data: Input test data
        :param session: Tensorflow session to use
        :param dist: Distance to use. By default L2 distance is used
        :param selected: Number of positive instances after the reference instance
            (first instance). By default it is one
        :returns: Mean rank of the positive instances """
        outs = self.perform_test(data, np.empty((len(data))), session)
        # Copy reference instance as many times as comparisons needed
        # and compare reference vector and other instances vector
        repeated_ref = np.tile(outs[0], (len(outs) -1, 1))
        # Compute distances between first row and others
        distances, order = dist(repeated_ref, outs[1:])
        rank = nu.evaluate_test(distances, order, selected)
        return rank, order

    def _get_mean_rank(self, subset, others, session, dist=nu.l2, crag=None):
        """ Returns the mean rank for a subset of the test set
            :param subset: Number of instances from test to use
            :param others: Number of instances to compare each selected
                test instance. If crag is not None, this option is not used
            :param session: Tensorflow session to use
            :param dist: Distance to use. Default: L2
            :param crag: This is the path to the CRAG the test data comes from.
                If enabled, it will test instances only against those which are connected
                against the reference one through an assignment node. To compare with
                other random test instances, set this option to None
        """
        # Check rank of match for each test instance
        # against sections of all other instances
        positive_inds = self.get_positive_instances(self.test_ind)

        # Get only a subset in the test set
        if subset > len(positive_inds):
            raise ValueError('Cannot test more positive instances that the ones that exist')
        subset_inds = positive_inds[nu.get_random(subset, len(positive_inds))]

        ranks = []
        for i in subset_inds:

            print('Testing positive instance %d' % i)

            if crag is None:
                eval_data = self.build_evaluation_from_test(self.test_ind, i, others)
            else:
                eval_data, selected = self.build_evaluation_from_crag(i)

            # Forward pass
            selected = selected if crag is not None else 1
            rank, order = self._query_ranks(eval_data, session, dist, selected)
            ranks.append(rank)
            print('Tested instance %d, got rank %d' % (i, rank))

        mean_rank = np.mean(ranks)
        print('Mean rank is {}'.format(mean_rank))
        best = 'high' if order is True else 'low'
        print('For the measure type selected, %s are preferred' % best)
        return mean_rank

    def _track_test_rank(self, session, step, crag, test_subset=10, others_num=25):
        """ Stores the mean rank for a subset of the test data
        :param session: Tensorflow session
        :param step: Current step
        :param crag: Crag to use
        :param test_subset: Number of test instances to use
        :param others_num: Number of test examples to compare with for each test instance """
        mean_rank = self._get_mean_rank(subset=test_subset, others=others_num,
            session=session, crag=crag)
        self.mean_rank.append([step, mean_rank])

    ###################################
    ######## CORE FUNCTIONS
    ###################################

    def _initialize_graph(self, mode):
        """ Returns the initialized graph for the siamese architecture,
        defined in a global scope """
        g = tf.Graph()

        with g.as_default():

            with tf.variable_scope(self.scope_name) as scope:

                # Create shared weights in the root scope
                self._define_weights()
                self.step = nu.get_step_variable()
                scope.reuse_variables()

                # Define input placeholders to feed
                self.define_placeholders(mode)

                # Define losses and store it and
                self.outputs = self.get_output()

                # Create saver to store/restore elements
                saver = tf.train.Saver()

                if mode == NetworkMode.LOSS:

                    # Compute and minimize loss only for training
                    self.loss, self.l2 = self.get_loss(self.outputs, self.pl_labels)

                    # Minimize loss by gradient descent
                    #sgd = tf.train.AdamOptimizer(self.config.lr)
                    sgd = tf.train.AdagradOptimizer(self.config.lr)
                    self.optimizer = sgd.minimize(self.loss, global_step=self.step)
                    tf.scalar_summary('loss', self.loss)

                # Define summaries for all variables
                for var in tf.trainable_variables():
                    tf.histogram_summary(var.op.name, var)
        return g, saver

    def _define_weights(self):
        """ Defines the weights needed in he network """
        parameters = []

        # Create weights for convolutional layers
        for i, layer in enumerate(self.config.cnv_layers):
            depth = self.depth if i == 0 else self.config.cnv_layers[i - 1].maps
            w, b, bn = nu.create_conv_weights(i, layer.ksize, depth, layer.maps, layer.bn)
            parameters.append((w, b, bn))

        # Create weights for fully connected layers
        for i, layer in enumerate(self.config.full_layers):
            if i == 0:
                # Maxpooling is hardcoded to halve images at both dimensions
                nmaxp = np.power(2, 2 * self.config.maxp_pools)
                input_size = self.imh * self.imw / nmaxp \
                             * self.config.cnv_layers[-1].maps
            else:
                input_size = self.config.full_layers[i - 1].hidden
            wf1, bf1 = nu.create_fc_weights(i, input_size, layer.hidden)
            parameters.append((wf1, bf1))

        return parameters

    def model(self, images, prefix):
        """ Defines the graph model
            :param images: Input data
            :param prefix: Prefix to add to each operator to identify element in pair/triplet
        """
        # Obtain weights (already created)
        ws = self._define_weights()

        # Augment data if requested
        images = tf.cond(tf.equal(tf.size(self.pl_aug), tf.constant(0)),
                         lambda: tf.identity(images),
                         lambda: self.augment_data(images, prefix), name='augment_' + prefix)

        tf.image_summary(prefix, images, max_images=3)

        self.layers_out = []
        # Convolutional layers
        for i, l in enumerate(self.config.cnv_layers):
            with tf.variable_scope('_'.join([self.CONV_LAB, str(i), prefix])):
                prev = self.layers_out[i - 1] if i > 0 else images
                self.layers_out.append(nu.define_convolution(prev, ws[i][0], ws[i][1], i,
                                                             l.maxp, ws[i][2], prefix))

        # Flatten convolutions output
        input_size = tf.shape(self.layers_out[-1])
        flat = tf.reshape(self.layers_out[-1],
                          tf.pack([input_size[0], input_size[1] * input_size[2] * input_size[3]]))

        # Fully connected layers
        fully_offset = len(self.config.cnv_layers)
        for i, l in enumerate(self.config.full_layers):
            with tf.variable_scope('_'.join([self.FULLY_LAB, str(i), prefix])):
                prev = self.layers_out[fully_offset + i - 1] if i > 0 else flat
                fullyw = ws[fully_offset + i]
                self.layers_out.append(nu.define_fully_layer(prev, fullyw[0], fullyw[1],
                                                             i, prefix, self.pl_dropout))

        return self.layers_out[-1]

    def get_loss(self, outputs, pl_labels):
        """ Returns the loss term """
        loss_base = self.get_specific_loss(outputs, pl_labels)
        final_loss = tf.identity(loss_base)

        if self.config.l2_reg > 0:
            # If L2 regularization enabled, sum over all parameters
            # and add them to the general loss term
            ws = self._define_weights()
            with tf.variable_scope('l2_term'):
                l2_term = tf.constant(0.0)
                for (w, b) in ws:
                    l2_w = nu.l2_penalty(w, self.config.l2_reg)
                    l2_b = nu.l2_penalty(b, self.config.l2_reg)
                    l2_term = l2_w + l2_b + l2_term
                    tf.scalar_summary(l2_w.name, l2_w)
                    tf.scalar_summary(l2_b.name, l2_b)

                # Register both terms in the summary
                tf.scalar_summary('loss_base', loss_base)
                tf.scalar_summary('loss_l2_term', l2_term)
                # Add both components
                final_loss = tf.add(loss_base, l2_term)
        else:
            l2_term = tf.constant(0)

        return final_loss, l2_term

    def get_output(self):
        """" Returns the output layers of the networks for each branch """
        return [self.model(pl, name) for (pl, name) in zip(self.pls, self.pl_names)]

    def build_feed(self, batch_data, batch_labels, dropout=True, augm=True):
        """ Builds a dictionary that feeds the network with the input data.
        Dropout and data augmentation can be controlled by the input parameters """
        feed = {pl: batch_data[:, i, ...] for i, pl in enumerate(self.pls)}
        feed.update({self.pl_labels: batch_labels})

        # Only to use during training
        keep_p = self.config.keep_p if dropout is True else 1
        feed.update({self.pl_dropout: keep_p})

        # If data augmentation provided, add random instances
        # Otherwise feed None
        num = self.config.augm
        augm = nu.get_random(num, self.batch_take) \
            if num is not None and augm is True else []
        feed.update({self.pl_aug: augm})

        return feed

    def get_data(self, inds):
        """ Returns data and labels from the given indices """
        mask = nu.boolean_mask(inds, self.data['data'].shape[0])
        data = nu.reshape_data(self.data['data'][mask, ...][:])
        return data, self.get_label_batch(mask)

    def configure_sess(self):
        """ Configures Tensorflow session """
        gpu = tf.GPUOptions(per_process_gpu_memory_fraction=self.config.mem)
        config = tf.ConfigProto(gpu_options=gpu,
                                intra_op_parallelism_threads=self.config.workers)
        return config

    def append_losses(self, step, train, val):
        """ Appends training and validation loss to the network registers """
        self.train_loss.append([step] + train)
        self.val_loss.append([step] + val)

    def get_step(self, session):
        """ Returns the step at which the model is being trained """
        # Build fake data in the dictionary
        feed = self.build_feed(
            np.empty((0, self.depth, self.imh, self.imw, 3)),
            np.empty((0)))
        s = session.run([self.step], feed_dict=feed)
        return int(s[0])

    ###################################
    ######## IO FUNCTIONS
    ###################################

    def print_loss(self, loss_value, step, label, duration):
        """ Prints loss according to given step"""
        examples_per_sec = self.batch_take / duration
        total, l2 = loss_value
        base = total - l2
        format_str = ('---> %s: step %d [%s] Loss = %.2f (Base loss = %.2f, L2 loss = %.2f)' +
                      '(%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (dt.datetime.now(), step, label, total, base, l2,
                             examples_per_sec, float(duration)))

    def display_loss(self, save=None):
        """ Plots the training and the validation evolution.
        If save enabled, it is not shown but saved into the given path """
        x = np.arange(0, len(self.train_loss)) * self.config.loss_int
        plt.plot(x, [i[0] for i in self.train_loss], label='training')
        plt.plot(x, [i[1] for i in self.train_loss], label='training_l2reg')
        plt.plot(x, [i[0] for i in self.val_loss], label='validation')
        plt.plot(x, [i[1] for i in self.val_loss], label='validation_l2reg')
        fig = plt.gcf()
        plt.legend(loc=2)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        if save is not None:
            fig.savefig(save)
        else:
            plt.show()
        plt.close(fig)

    def get_rank_file(self, outp):
        """ Returns the path to the rank values given the model folder """
        return os.path.join(outp, self.rank_name)

    def get_loss_file(self, outp):
        """ Returns the path to the loss values given the model folder """
        return os.path.join(outp, self.loss_name)

    def load_net_state(self, outp, saver, session, model_file=None):
        """ Loads the network state, the losses and the mean ranks computed from
        the model in the folder. If a model file path is provided, loads model from
        the file. Otherwise, checks for the latest model in the folder """
        self.train_loss, self.val_loss = nu.read_losses(self.get_loss_file(outp))
        self.mean_rank = nu.read_ranks(self.get_rank_file(outp))
        # Initialize non-trainable data
        all_names = [i.name for i in tf.all_variables()]
        trainable = [i.name for i in tf.trainable_variables()]
        vars_to_init = list(set(tf.all_variables()) - set(tf.trainable_variables()))
        vars_to_init_names = [i.name for i in vars_to_init]
        print('All: {}'.format(all_names))
        print('Trainable: {}'.format(trainable))
        print('Variables to initialize: {}'.format(vars_to_init))
        tf.initialize_variables(vars_to_init).run(session=session)
        # Load model from folder or file accordingly
        if model_file is None:
            print('Loading model from folder {}'.format(outp))
            nu.restore_model(session, saver, outp)
        else:
            print('Loading model from {}'.format(model_file))
            saver.restore(session, model_file)

    def store_net_state(self, session, saver, step, outp, name):
        """ Stores the current state of the network: metadata, loss, mean rank,
        and checkpoint """
        nu.store_checkpoint(session, saver, step, outp, self.metadata, name)
        nu.dump_losses(self.train_loss, self.val_loss, os.path.join(outp, 'loss.dat'))
        nu.dump_rank(self.mean_rank, os.path.join(outp, 'ranks.dat'))

    def store_summary(self, batch_data, batch_labels, step, session):
        """ Stores a summary of the measures taken in the graph """
        feed = self.build_feed(batch_data, batch_labels)
        summary_str = session.run(self.summary_op, feed_dict=feed)
        self.sum_writer.add_summary(summary_str, step)


class PairedSiamese(SiameseNetwork):

    def get_label_batch(self, inds):
        return self.data['labels'][inds, ...][:]

    def get_placeholder_suffixes_specific(self):
        return ['left', 'right']

    def get_specific_loss(self, outputs, labels):
        return self._paired_loss(outputs[0], outputs[1], labels)

    def get_positive_instances(self, inds):
        subset = self.data['labels'][inds, ...][:]
        positive = np.where(subset == 1)[0]
        return inds[positive]

    def _paired_loss(self, f1, f2, labels, thao=1):
        """ Loss for a evaluation of a pair of (non)-corresponding slices
            Paired loss L:
                L = 0.5 * d(f1, f2) if positive example
                L = 0.5 * max(0 , thao - d(f1, f2)) if negative example
            Where thao is usually 1
        """
        # Must be float, convert to be sure
        dists = nu.euclidean_dist(f1, f2)

        # Masks of classes
        int_labels = tf.cast(labels, tf.int32)
        pos_ind = tf.cast(labels, tf.float32)
        neg_ind = tf.cast(tf.sub(tf.ones(tf.shape(labels), tf.int32), int_labels), tf.float32)

        # Apply positive and negative to both classes
        all_pos = dists
        hinge_const = tf.cast(tf.fill(tf.pack([tf.shape(dists)[0]]), thao, name='thao'), all_pos.dtype)
        all_neg = tf.sub(hinge_const, dists)
        all_neg = tf.maximum(tf.zeros(tf.shape(all_neg), all_neg.dtype), all_neg)

        # Apply binary mask to both so set 0's where not correct
        binary_pos = tf.mul(pos_ind, all_pos)
        binary_neg = tf.mul(neg_ind, all_neg)

        # Sum both and multiply by 1/2
        summed = tf.add(binary_pos, binary_neg)
        summed = tf.mul(summed, tf.fill(tf.pack([tf.shape(summed)[0]]), 0.5, name='0.5'))
        return tf.reduce_sum(summed)

    def get_positive_instances(self, inds):
        subset = self.data['labels'][inds, ...][:]
        positive = np.where(subset == 1)[0]
        return inds[positive]


class TripletSiamese(SiameseNetwork):

    def get_label_batch(self, inds):
        # Return whatever since it is not used
        return np.empty(sum(inds))

    def get_placeholder_suffixes_specific(self):
        return ['left', 'center', 'right']

    def get_specific_loss(self, outputs, labels):
        return self._triplet_loss(outputs[0], outputs[1], outputs[2])

    def _triplet_loss(self, f1, f2, f3, alpha=1):
        """ Loss for a evaluation of a pair of (non)-corresponding slices
            Triplet loss L:
                L= 0.5 * max(0, d(f1, f2) - d(f1, f3) + alpha)
            Where alpha is usually 1
        """
        # Distances negative and positive
        dist_pos = nu.euclidean_dist(f1, f2)
        dist_neg = nu.euclidean_dist(f1, f3)

        # Get maximum 
        subs = tf.sub(dist_pos, dist_neg)
        add_const = tf.cast(tf.fill(tf.pack([tf.shape(subs)[0]]), alpha), subs.dtype)
        added = tf.add(subs, add_const)
        loss = tf.maximum(tf.zeros(tf.shape(added), added.dtype), added)

        # Multiply by 1/2
        const = tf.fill(tf.pack([tf.shape(loss)[0]]), 0.5)
        return tf.reduce_mean(tf.mul(loss, const))

    def get_positive_instances(self, inds):
        return inds # All include a positive match in triplets

