#!/usr/bin/python

from neuralimg.training import siameseconf as cnf
from neuralimg.dataio import *

import tensorflow as tf
import numpy as np
import os
import glob
import logging


logger = logging.getLogger('training')


def create_conv_weights(i, ksize, inps, outs, bn, m=0.0, stddev=1e-1, bias=0):
    """ Define weights of a convolutional layer """
    # Weights
    w = tf.get_variable('convw' + str(i), [ksize, ksize, inps, outs],
            initializer=tf.truncated_normal_initializer(mean=m, stddev=stddev))
    # Define biases
    b = tf.get_variable('convb' + str(i), [outs],
            initializer=tf.constant_initializer(bias))
    # Define batch norm weights if requested
    if bn is True:
        scale = tf.get_variable('scale' + str(i), initializer=tf.zeros([w.get_shape()[-1]]))
        beta = tf.get_variable('beta' + str(i), initializer=tf.zeros([b.get_shape()[-1]]))
    else:
        scale, beta = None, None
    return w, b, (scale, beta)


def create_fc_weights(i, inps, hiddens, m=0.0, stddev=1e-1, bias=0):
    """ Define weights of a fully connected layer """
    # Weights
    w = tf.get_variable('fullyw' + str(i), [inps, hiddens],
            initializer = tf.truncated_normal_initializer(mean=m, stddev=stddev))
    # Define biases
    b = tf.get_variable('fullyb' + str(i) , [hiddens], 
            initializer = tf.constant_initializer(bias))
    return w, b


def get_step_variable():
    """ Returns the global step variable. First time this is called the variable
    is created """
    return tf.get_variable('step', (),
                           initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))


def summary_activations(name, x):
    """ Defines a histogram and a sparsity measure of the activations for the input function """
    tf.histogram_summary(name + '/activations', x)
    tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(x))


def define_convolution(data, w, b, i, maxp, bn, prefix):
    """ Returns a convolutional layer with the input parameters. Max pooling is restricted to have max pooling
    of size 2 and stride 2 for both a and y(reduces size to a half for both dimensions) and convolution
    stride is set to 1 for both x and y. Padding is introduced to preserve sizes.
        :param data: Input placeholder
        :param w: Input weights
        :param b: Input bias
        :param i: Identifier of the convolutional layer
        :param maxp: Whether to perform max pooling after the activation
        :param bn: Tuple containing scale and beta variables for the batch normalization. If disabled, it contains a pair of None
        :param prefix: Prefix of the layer (left/right/center)
    """
    # Convolution operation
    conv = tf.nn.conv2d(data, w,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name=prefix + '-conv' + str(i))

    if bn[0] is not None and bn[1] is not None:
        conv = batch_normalization(conv, bn[0], bn[1])

    # Perform ReLU(x) = max(0, x)
    relu = tf.nn.relu(tf.nn.bias_add(conv, b),
                      name=prefix + '-relu' + str(i))

    # Perform max pooling, if requested
    if maxp is True:
        relu = tf.nn.max_pool(relu,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool' + str(i))

    summary_activations('_'.join([prefix, 'conv', str(i)]), relu)

    return relu


def batch_normalization(inp, scale, beta):
    """ Performs batch normalization on the input batch 
    Code from: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html """
    batch_mean, batch_var = tf.nn.moments(inp, [0])
    epsilon = tf.constant(1e-3)
    hat = (inp - batch_mean) / tf.sqrt(batch_var + epsilon)
    return scale * hat + beta


def define_fully_layer(data, w, b, i, prefix, dropout, normalize_unit=False, activation=False):
    """ Returns a fully connected layer with the input parameters. Weights are initialized as truncated
    normals around 0 and stdev of 1e-3 and bias as 0.
        :param data: Input placeholder
        :param w: Input weights
        :param b: Input bias
        :param i: Identifier of the convolutional layer
        :param prefix: Prefix of the layer (left/right/center)
        :param dropout: Keep probability of the dropout (in interval [0,1])
    """
    # Perform ReLU(x) = max(0, x)
    relu = tf.nn.bias_add(tf.matmul(data, w), b)
    if activation:
        relu = tf.nn.relu(relu, name=prefix + '-relufc' + str(i))
    else:
        print 'no activation is added to fully connected layer'

    if normalize_unit:
        relu = tf.contrib.layers.layers.unit_norm(relu, 1)
        logging.info('embedding is unit normed')

    # Add dropout
    relu = tf.nn.dropout(relu, dropout)

    summary_activations('_'.join([prefix, 'relu', str(i)]), relu)
    return relu


def store_checkpoint(session, saver, step, path, metadata, name='model'):
    """ Stores a checkpoint of the model
        :param session: Tensorflow session
        :param saver: Tensorflow saver
        :param step: Training step where the model belongs to
        :param path: Folder where to store the model
        :param metadata: Metadata object with data characteristics
        :param name: Name of the checkpoint file
    """
    checkpoint_path = os.path.join(path, name + '_' + str(step) + '.ckpt')
    saver.save(session, checkpoint_path, global_step=step)
    save_pickle(os.path.join(path, 'metadata.dat'), metadata)


def load_model_data(path):
    """ Stores a checkpoint of the model
        :param path: Folder where to load model metadata from
    """
    return read_pickle(os.path.join(path, 'metadata.dat'))


def euclidean_dist(f1, f2):
    """ Computes the euclidean distance between two tensors
     ith position of result is the euclidean distance between ith
     position of f1 and the corresponding in f2"""
    # Must be float, convert to be sure
    f1 = tf.cast(f1, tf.float32)
    f2 = tf.cast(f2, tf.float32)

    # Compute euclidean distance between outputs
    subs = tf.sub(f1, f2)
    pows = tf.pow(subs, tf.fill(tf.shape(subs), 2.0))
    suma = tf.reduce_sum(pows, 1)
    return tf.sqrt(suma)


def reshape_data(x):
    """ Converts the data into the Tensorflow format with float32 type"""
    return np.moveaxis(x, 2, 4).astype(np.float32)


def boolean_mask(insts, total):
    """ Masks the list of indices into a boolean array of size 'total' where
    elements in insts are set to True and the rest are False """
    mask = np.zeros(total, dtype=bool)
    mask[insts] = True
    return mask


def excluding_permutation(exclude, inds):
    """ Generates a random permutation from the input indices,
     excluding one of them """
    exclude_pos = np.where(inds == exclude)[0]
    mask = np.ones(len(inds), dtype=bool)
    mask[exclude_pos] = False
    perm = inds[mask]
    return np.random.permutation(perm).tolist()


def to_output_data(data, labels):
    """ Reshapes data so it is ready to extract descriptors from it
        :param data: Input data
        :param labels: Label data
    """

    labels = np.empty([data.shape[0]]) if labels is None else labels

    if len(data.shape) != 4:
        raise ValueError('Data must have the form: NUM x H x W x C')
    if len(labels.shape) != 1 or labels.shape[0] != data.shape[0]:
        raise ValueError('Labels must be unidimensional and have the same' +
                         ' number of instances as the input data')
    return reshape_data(np.expand_dims(data, axis=1)), labels


def restore_model(session, saver, path):
    """ Restores model from the given path and returns the created
    saver object """
    # Get all model-related data and exclude metadata files
    all_models = glob.glob(os.path.join(path, '*ckpt*'))
    model_paths = [i for i in all_models if not str.endswith(i, '.meta')]
    # Sort paths so latest is retrieved
    model_paths.sort()
    if len(model_paths) == 0:
        raise ValueError('Model path not found in %s' % path)
    saver.restore(session, model_paths[-1])


def l2(f1, f2):
    """ Retrieves the euclidean distance between two input arrays """
    return np.sqrt(np.sum(np.power(f1 - f2, 2.0), axis=1)), False


def mod(x):
    """ Returns the module of an array """
    return np.sqrt(sum(np.power(x, 2)))


def cosine(f1, f2):
    """ Returns the cosine similarity between the two input arrays """
    return sum(f1 * f2) / (mod(f1) * mod(f2)), True


def get_random(num, up):
    """ Returns random unique numbers within the interval
    :param num: Number of indices to retrieve
    :oaram up: Maximum number in the generation """
    return  np.random.permutation(up)[0:num]


def read_ranks(outp):
    """ Reads the rank history from the model's output folder """
    with open(outp, 'r') as f:
        lines = f.readlines()
        ranks = [line.strip('\n').split('\t') for line in lines[1:]]
    return ranks 


def read_losses(outp):
    """ Reads the loss history from the model's output folder """
    with open(outp, 'r') as f:
        lines = f.readlines()
        train, val = [], []
        for line in lines[1:]:
            split = line.strip('\n').split('\t')
            train.append(split[:3])
            val.append([split[0]] + split[3:])
    return train, val 


def dump_rank(ranks, outp):
    """ Dumps mean ranks into a text file so they are stored as
    step mean_rank """
    with open(outp, 'w') as f:
        # Write header
        headers = ['Step', 'Mean_rank']
        f.write('\t'.join(headers))
        # Write lines
        for (s, r) in ranks:
            f.write('\n' + '\t'.join([str(s), str(r)]))


def dump_losses(train, val, outp):
    """ Dumps losses into a text file so they are stored as:
    step train_loss validation_loss """
    with open(outp, 'w') as f:
        # Write header
        headers = ['Steps', 'Training_loss', 'Training_L2_loss',
                   'Validation_loss', 'Validation_L2_loss']
        f.write('\t'.join(headers))
        # Write lines
        for (t, v) in zip(train, val):
            f.write('\n' + '\t'.join([str(i) for i in [v[0], t[1], t[2], v[1], v[2]]]))


def load_file_ext(folder, ext):
    """ Loads a file in the given folder with certain extension """
    confs = glob.glob(os.path.join(folder, '*' + ext))
    if len(confs) == 0:
        raise ValueError('No %s file found in folder %s' % (ext, folder))
    elif len(confs) > 1:
        logger.warn('Warning: Found several files with extension %s. Picking %s' % ext, confs[0])
    return confs[0]


def exists_conf(folder):
    """ Returns whether exists a configuration file in the given folder """
    confs = glob.glob(os.path.join(folder, '*.conf'))
    return len(confs) > 0

def load_configuration(folder):
    """ Loads a configuration file in the given folder """
    return load_file_ext(folder, '.conf')


def read_conf(path):
    """ Reads the configuration file """
    conf = cnf.NetworkConf()
    conf.read(path)
    return conf


def l2_penalty(inp, factor):
    """ Returns the L2 penalty tensor for the input parameter """
    return tf.mul(tf.constant(factor), tf.nn.l2_loss(inp))


def evaluate_test(metric, incr, selected):
    """ Test evaluation for siamese matching: It returns the rank
    of the matching distances within all distances computed.
    :param metric: Metric from pattern to other patterns.
        Distance with matching patches must be in the beginning of the list
    :param incr: Whether it refers to a similarity metric (True) or a
    	distance metric (False)
    :param select: Index index of the first non-matching pattern 
	in the metrics list """
    logger.debug('Distances {}'.format(metric))
    ranks = [i[0] for i in sorted(enumerate(metric),
                                  key=lambda x: x[1], reverse=incr)]
    logger.debug('Ranks {}'.format(ranks))
    # Compute sum of ranks for matching patterns. If they are in any position
    # within [0, selected) their rank is computed as 0
    ranks_out = [r for (i, r) in enumerate(ranks[:selected]) if r >= selected]
    # Average sum of ranks out of position by total number of matching positions
    mean_rank = sum(ranks_out)/float(selected)
    return mean_rank

def _build_matrix(slices, branches):
    """ Builds a matrix containing the input slices, in order, and extending to
    the given number of branches """
    result = np.empty((len(slices), branches) + slices[0].shape)
    for (i, sl) in  enumerate(slices):
        for j in range(branches):
            result[i, j, ...] = sl
    return result

