# !/usr/bin/python

import mahotas as mh

import numpy
import os
import glob
import h5py
import pickle
import pylab as plt


class DatasetReader(object):

    """ Class that reads a CRAG dataset from an HDF5 or a pickle serialization """

    def __init__(self, path):
        self.path = path

    def read(self):

        if is_file(self.path):
            if valid_volume_path(self.path):
                # Path is an HDF5
                return self._read_hdf5()
            else:
                # Is a pickle file
                return read_pickle(self.path)
        else:
            raise IOError('Path {} does not exist or it is not a valid file'.format(self.path))

    def _read_hdf5(self):
        with h5py.File(self.path, 'r') as f:
           res = dict()
           for k in f.keys():
                res[k] = dict()
                for kk in f[k].keys():
                    res[k][kk] = f[k][kk][:]
        return res


class FileReader(object):

    """ Class that reads a set of images from a folder and extracts file names according
    to accepted extensions """

    def __init__(self, path, exts=['.tiff', '.tif', '.png', '.jpg', '.jpeg']):
        self.path = path
        self.exts = exts

    def extract_files(self, num=None):
        """ Extracts the file names of the images in the folder with accepted extensions
            :param num: Number of random images ot extract. Set to None for all images
        """
        names = []
        for e in self.exts:
            names.extend(glob.glob(os.path.join(self.path, '*' + e)))
        names.sort()

        # Extract random images
        if num is not None:
            if num > len(names):
                raise ValueError('Cannnot extract more images than ' +
                    'the ones that exist: {}'.format(str(len(names))))
            names = [names[i] for i in numpy.random.permutation(len(names))[0:num]]

        return names

    def read(self, num=None):
        """ Reads images from the accepted extensions in the given folder 
            :param num: Number of random images ot extract. Set to None for all images
        """

        names = self.extract_files(num)

        if len(names) == 0:
            raise Exception("No images to read in {}".format(self.path))

        first = mh.imread(names[0])
        mat = numpy.empty((len(names), first.shape[0], first.shape[1]), dtype=first.dtype)
        for i in range(len(names)):
            mat[i, :, :] = mh.imread(names[i])
        return mat


def check_folder(path):
    """ Checks whether the folder exists. If not, raises an error """
    if not os.path.isdir(path):
        raise IOError('Folder ' + path + ' does not exist')


def check_file(path):
    """ Checks whether the file exists. If not, raises an error """
    if not os.path.isfile(path):
        raise IOError('File ' + path + ' does not exist')


def check_volume(path):
    """ Checks whether path corresponds to a volume file (HDF5).
    Raises exception otherwise """
    if not valid_volume_path(path):
        raise ValueError('Path is not a valid HDF5 file')


def save_pickle(path, obj):
    """ Serializes the input object into the path """
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def read_pickle(path):
    """ Reads the input object stored in the path """
    with open(path, 'r') as f:
        return pickle.load(f)


def is_file(path):
    """ Whether the provided path belongs to an hipothetic file(true) or a folder (false)"""
    filename, file_ext = os.path.splitext(path)
    return not not file_ext


def valid_volume_path(path):
    """ Returns whether the input path has a proper HDF5 extensions """
    filename, file_ext = os.path.splitext(path)
    return file_ext.lower() == '.hdf5' or file_ext.lower() == '.h5' \
        or file_ext.lower() == '.hdf'


def get_hf_group(hf_file, group):
    """ Given path using groups separated by /, obtains final group through path"""
    data = hf_file
    for i in group.split('/'):
        if i not in data:
            raise KeyError('Key {} not found in dataset'.format(i))
        data = data[i]
        print('Accessed {}'.format(i))
    return data


def store_hf_group(hf_file, group, d):
    """ Given path using groups separated by /, creates intermediate groups and stores
    data in the last level"""
    splits = group.split('/')
    g = hf_file
    for i in splits[:-1]:
        g = g.create_group(i)
    g.create_dataset(splits[-1], data=d, compression='gzip', dtype=d.dtype)


def volume_to_folder(data, outp, ext='.tif', typ='float', min_digit=5):
    """ Outputs the given volume of data into the path. Extension is 'tif' by default
    and output type 'float' """
    create_dir(outp)
    digits = max(len(str(data.shape[0])), min_digit)
    for i in range(0, data.shape[0]):
        print('Saving image {}'.format(str(i)))
        to_save = data[i, ...].astype(typ)
        mh.imsave(os.path.join(outp, str(i).zfill(digits) + ext), to_save)


def create_dir(path):
    """ Creates directory if it does not exist """
    if not os.path.exists(path):
        os.makedirs(path)

# Debug functions

def plot_matrix(data, save_fig=None):
    """ Plots images in the input matrix that represent different slice node 
    images for each row and columns display different channels for each section.
    If path provided, saves plot into an image """
    instances, channels = data.shape[0], data.shape[1]
    print('Instances %d' % instances)
    print('Channels %d' % channels)
    f, axarr = plt.subplots(instances, channels, sharex=True, sharey=True,
                           figsize=(10, 7))
    if instances == 1:
        for j in range(channels):
            axarr[j].imshow(data[0, j, ...])
    else:
        for i in range(instances):
            for j in range(channels):
                axarr[i, j].imshow(data[i, j, ...])

    if save_fig is not None:
        plt.savefig(save_fig)
