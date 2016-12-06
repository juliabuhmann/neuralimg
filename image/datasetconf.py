#!/usr/bin/python

import configparser

class DatasetGenConf(object):

    """ List of parameters to configure a dataset generation 

        padding: Padding ratio around superpixel (float)
        norm: Whether to normalise the final dataset so it is in interval [0,1] (True/False)
        balance: Whether to balance dataset so each class has same number of 
                instances. This only works for paired datasets (not triplets) (True/False)
        shuffle: Whether to shuffle data after building it (recommended) (True/False)
        height: Height of the dataset samples /integer)
        width: Width of the dataset samples (integer)
        raw: Whether to include the raw channel (True/False)
        binary: Wether to include the binary channel (True/False)
        intensity: Whether to include the intensity/boundary channel (True/False)
        sets: Whether to split data into training/validation/testing (True/False)
        mode = Ways of splitting the data into sets related to the classes. Only used if sets is True. (String)
            Triplet datasets are internally forced to use random split.

            Values
                - same: each class has the same number (aprox.) of instances in 
                  each set.
                - ratio: the ratio between classes in the original set is 
                  preserved.
                - random: data is selected randomly to fill the sets.
    """

    def __init__(self, pad=None, bal=None, shuf=None, height=None, width=None, 
        binar=None, raw=None, inten=None, norm=None, sets=None, mode=None,
        ratioT=None, ratioV=None):
        self.sc = 'DatasetConf'
        self.padding = pad
        self.balance = bal
        self.shuffle = shuf
        self.height = height
        self.width = width
        self.binary = binar
        self.raw = raw
        self.intensity = inten
        self.normalise = norm
        self.sets = sets
        self.mode = mode
        self.ratioT = ratioT
        self.ratioV = ratioV


    def read(self, path):
        """ Reads the dataset generation configuration from the file """
        config = configparser.ConfigParser()
        config.read(path)
        self.padding = config[self.sc].getfloat('padding')
        self.normalise = config[self.sc].getboolean('normalise')
        self.balance = config[self.sc].getboolean('balance')
        self.shuffle = config[self.sc].getboolean('shuffle')
        self.height = config[self.sc].getint('height')
        self.width = config[self.sc].getint('width')
        self.binary = config[self.sc].getboolean('binary')
        self.raw = config[self.sc].getboolean('raw')
        self.intensity = config[self.sc].getboolean('intensity')
        self.sets = config[self.sc].getboolean('sets')
        self.mode = config[self.sc].get('mode')
        self.ratioT = config[self.sc].getfloat('ratioT')
        self.ratioV = config[self.sc].getfloat('ratioV')
        self.rank = config[self.sc].getint('rank')

        # Check arguments
        if self.width < 0 or self.height < 0:
            raise ValueError('Dimensions must be positive')
        if self.padding > 2.0:
            raise ValueError('A valid padding must be between 1 and 2. Set to < 1 for no padding')
        if self.sets is True:
            if self.ratioT < 0.0 or self.ratioT > 1.0 or self.ratioV < 0.0 or self.ratioV > 1.0:
                raise ValueError('Training and validation ratio must be in range [0,1]')
        if self.mode not in ['same', 'ratio', 'random'] and self.sets is True:
            raise ValueError('Split mode must be one of the following: same, ration, random.')

    def write(self, path):
        """ Output the configuration into the given path """
        config = configparser.ConfigParser()
        config[self.sc] = {}
        section = config[self.sc]
        section['padding'] = str(self.padding)
        section['normalise'] = str(self.normalise)
        section['balance'] = str(self.balance)
        section['shuffle'] = str(self.shuffle)
        section['height'] = str(self.height)
        section['width'] = str(self.width)
        section['binary'] = str(self.binary)
        section['raw'] = str(self.raw)
        section['intensity'] = str(self.intensity)
        section['sets'] = str(self.sets)
        section['mode'] = str(self.mode)
        section['ratioT'] = str(self.ratioT)
        section['ratioV'] = str(self.ratioV)

        with open(path, 'w') as f:
            config.write(f)


if __name__ == '__main__':

    c = DatasetGenConf()
    c.read('config/data.conf')
    c.write('data2.conf')

