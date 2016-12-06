#!/usr/bin/python

import configparser
import os


class ConvLayer(object):

    """ Convolutional layer. Contains following information:
        - ksize: Size of the convolutional filter
        - maps: Number of filters in the layer
        - bn: Whether to perform batch normalisation.
        - maxp: Whether to perform maxpooling after the convolution
    """

    def __init__(self, ksize=None, maps=None, bn=None, maxp=None):
        self.ksize = ksize
        self.maps = maps
        self.bn = bn
        self.maxp = maxp

    def from_config(self, name, config):
        section = config[name]
        self.ksize = section.getint('ksize')
        self.maps = section.getint('maps')
        self.bn = section.getboolean('bn')
        self.maxp = section.getboolean('maxp')

    def store(self, name, config):
        config[name] = {}
        config[name]['ksize'] = str(self.ksize)
        config[name]['maps'] = str(self.maps)
        config[name]['bn'] = str(self.bn)
        config[name]['maxp'] = str(self.maxp)


class FullyCLayer(object):

    """ Fully connected layer. Contains following information:
        - hidden = Number of hidden units in the layer
        - norm_unit = Whether layer output should be normalized to unit length
        - activation = Whether activation (relu) should be performed
    """

    def __init__(self, hidden=None):
        self.hidden = hidden
        self.norm_unit = False
        self.activation = True

    def from_config(self, name, config):
        section = config[name]
        self.hidden = section.getint('hidden')
        self.norm_unit = section.getboolean('norm_unit')
        self.activation = section.getboolean('activation')


    def store(self, name, config):
        config[name] = {}
        config[name]['hidden'] = str(self.hidden)
        config[name]['norm_unit'] = str(self.norm_unit)
        config[name]['activation'] = str(self.activation)


class NetworkConf(object):

    """ List of parameters to configure a dataset generation

        batch_s: Batch training size
        lr: Learning rate (TODO)
        max_steps: Maximum steps to perform in training
        keep_p: Keep probability for dropout. To disable dropout, set to 1
        workers: Number of CPUs to use. Only used if GPU not enabled in host
        early: Early stopping strategy. Number of validation steps to compare backwards. If validation loss is
            higher than the one in 'early' validation steps before, we are overfitting the model and we must stop
            training. To disable it, set to None
        augm: whether to use data augmentation. It is the number of images per batch that we will
            use to create replicas, so 'augm' x 8 images will be generated per batch
        valid_batch: Number of random instances to take for validation evaluation
        loss_interval: Interval at which validation loss is stored
        summary_interval: Interval at which summary of the variables is stored
        checkpoint_interval: Interval at which a checkpoint of the model is stored
        lr_reg: L2 regularization factor. To disable it, set to negative
        mem: Ratio maximum memory utilization out of the available memory. Only used if host is GPU enabled
        min_decrease: Minimum loss decrease allowed. If the decrease is less than that, it will stop the training
            at the early stop check

        Layer parameters are declared in individual sections:
            [Conv1]
            [Conv2]
            ...
            [FullyC1]
            [FullyC2]

        Prefixes 'Conv' and 'FullyC' must be use for convolutional and fully connected layers
        respectively. Note that the identifier of the layer is not relevant but the relative orther among them
    """

    CONV_PREFIX = 'Conv'
    FULLC_PREFIX = 'FullyC'

    def __init__(self, batch_s=None, lr=None, max_steps=None, keep_p=None, workers=None, early=None,
        augm=None, valid_batch=None, loss_int=None, sum_int=None, checkp_int=None, l2=None, mem=None,
                 decrease=None, training_perc=None, dataset_size = None, conv_layers=[], full_layers=[]):
        self.cnv_layers = conv_layers
        self.full_layers = full_layers
        self.maxp_pools = 0
        self.batch_s = batch_s
        self.lr = lr
        self.steps = max_steps
        self.keep_p = keep_p
        self.workers = workers
        self.early = early
        self.augm = augm
        self.valid_batch = valid_batch
        self.loss_int = loss_int
        self.summary_int = sum_int
        self.checkpoint_int = checkp_int
        self.l2_reg = l2
        self.mem = mem
        self.min_decrease = decrease
        self.training_perc = training_perc
        self.dataset_size = dataset_size
        self.p_sec = 'Parameters'


    def read(self, path):
        """ Reads the dataset generation configuration from the file """
        config = configparser.ConfigParser()

        if not os.path.isfile(str(path)):
            raise IOError('File not found %s' % path)

        config.read(path)
        self.batch_s = config[self.p_sec].getint('training_batch_size')
        self.lr = config[self.p_sec].getfloat('learning_rate')
        self.steps = config[self.p_sec].getint('max_steps')
        self.keep_p = config[self.p_sec].getfloat('keep_probability')
        self.workers = config[self.p_sec].getint('workers')
        self.early = config[self.p_sec].getint('early_stop')
        self.augm = config[self.p_sec].getint('augm')
        self.valid_batch = config[self.p_sec].getint('validation_batch_size')
        self.loss_int = config[self.p_sec].getint('loss_interval')
        self.summary_int = config[self.p_sec].getint('summary_interval')
        self.checkpoint_int = config[self.p_sec].getint('checkpoint_interval')
        self.l2_reg = config[self.p_sec].getfloat('l2_reg')
        self.mem = config[self.p_sec].getfloat('memory')
        self.min_decrease = config[self.p_sec].getfloat('min_decrease')
        self.training_perc = config[self.p_sec].getfloat('training_perc')
        self.dataset_size = config[self.p_sec].getint('dataset_size')



        self._read_conv_layers(config)
        self._read_fullyc_layers(config)

        # Read number of max pools
        self.maxp_pools = sum([1 if i.maxp else 0 for i in self.cnv_layers])
        print('batch size %i' %self.batch_s)

        # Raise warning if big batch size has been chosen
        if self.batch_s > 500 or self.valid_batch > 500:
            print('Warning: it is recommended to use batch sizes of size {128, 256}')
        if self.keep_p < 0 or self.keep_p > 1:
            raise ValueError('Keep probability must be in interval [0,1]')
        if self.mem < 0 or self.mem > 1:
            raise ValueError('Memory occupation ratio must be in interval [0,1]')

    def _read_conv_layers(self, config):
        """ Read the convolutional layers in the configuration """
        layers = get_layers(config, self.CONV_PREFIX)
        self.cnv_layers = [read_clayer(i, config) for i in layers]

    def _read_fullyc_layers(self, config):
        """ Reads the fully connected layers in the configuration """
        layers = get_layers(config, self.FULLC_PREFIX)
        self.full_layers = [read_flayer(i, config) for i in layers]

    def write(self, path):
        """ Output the configuration into the given path """
        config = configparser.ConfigParser()

        # Parameters section
        config[self.p_sec] = {}
        paramsec = config[self.p_sec]
        config[self.p_sec]['training_batch_size'] = str(self.batch_s)
        paramsec['learning_rate'] = str(self.lr)
        paramsec['max_steps'] = str(self.steps)
        paramsec['keep_probability'] = str(self.keep_p)
        paramsec['workers'] = str(self.workers)
        paramsec['early_stop'] = str(self.early)
        paramsec['augm'] = str(self.augm)
        paramsec['validation_batch_size'] = str(self.valid_batch)
        paramsec['loss_interval'] = str(self.loss_int)
        paramsec['summary_interval'] = str(self.summary_int)
        paramsec['checkpoint_interval'] = str(self.checkpoint_int)
        paramsec['l2_reg'] = str(self.l2_reg)
        paramsec['memory'] = str(self.mem)
        paramsec['training_perc'] = str(self.training_perc)
        paramsec['dataset_size'] = str(self.dataset_size)

        # Write one section for each conv layer
        for (i, l) in enumerate(self.cnv_layers):
            l.store(self.CONV_PREFIX + str(i), config)

        # Write one section for each fully connected layer
        for (i, l) in enumerate(self.full_layers):
                l.store(self.FULLC_PREFIX + str(i), config)

        with open(path, 'w') as f:
            config.write(f)


def get_layers(config, prefix):
    layers = [i for i in config.sections() if str(i).startswith(prefix)]
    sorted(layers)
    return layers


def read_clayer(i, config):
    conv = ConvLayer()
    conv.from_config(i, config)
    return conv


def read_flayer(i, config):
    conv = FullyCLayer()
    conv.from_config(i, config)
    return conv

