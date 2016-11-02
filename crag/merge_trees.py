#!/usr/bin/python

import numpy as np
import os
import abc
import h5py
import shutil
import mahotas as mh
import tempfile

from neuralimg import dataio
from neuralimg.base import pyprocess as prs


class MergeTreeExtractor(object):

    def __init__(self, sp, mem, gt=None):
        self.gt = gt
        self.sp = sp
        self.mem = mem

    def extract(self, outp):
        """ Extracts the merge trees in the given folder """

        # Prepare temporary folder for extracting H5 files
        self.tmp = tempfile.mkdtemp()

        print('Processing data ...')
        self.parse_data()

        print('Extracting trees ...')
        dataio.create_dir(outp)
        self.extract_trees(outp)

        # Cleaning tmp
        shutil.rmtree(self.tmp)


    @abc.abstractmethod
    def extract_trees(self, outp):
        """ Extracts a merge tree for each input image in the destination folder """

    def parse_data(self):
        """ Processes the input data """
        self.sp = self.valid_input(self.sp)
        self.mem = self.valid_input(self.mem)
        if self.gt is not None:
            self.gt = self.valid_input(self.gt)


    @abc.abstractmethod
    def process_h5(self, inp):
        """ Processes HDF5 input and makes it compatibles with the method """

    @abc.abstractmethod
    def process_folder(self, inp):
        """ Processes input folder and makes it compatible with the method """

    def valid_input(self, d):
        """ Transforms the input into a compatible format """
        if os.path.isdir(d):
            return self.process_folder(d)
        elif dataio.valid_volume_path(d):
            return self.process_h5(d)
        else:
            raise IOError('Formats accepted are HDF5 and folders')


def get_digits(num):
    """ Number of digits to represent input number """
    return int(np.log10(num)) + 1


def is_normalized(path, g='stack'):
    with h5py.File(path) as f:
        imgs = f[g][:]
    return imgs.min() >=  0 and imgs.max() <= 1


def relabel_history(history, mapping):
    """ Given a mapping to the original superpixels relabels
    the regions """
    result = []
    max_map = mapping.max()
    for (n1, n2, n3, score) in history:
        # Map only those labels that are from the original
        # set. Newly created labels are kept the same
        # Important: we assume and know that labels are
        # creatly with monotonically increasing ids
        id1 = n1 if n1 > max_map else mapping[n1]
        id2 = n2 if n2 > max_map else mapping[n2]
        result.append((id1, id2, n3, score))
    return result


def dump_history(history, outp):
    """ Dumps a list of triplets (n3, n2, n1, score) into a file """
    with open(outp, "w") as f:
        for entry in history:
            f.write("\t".join([str(int(x)) for x in entry]))
            f.write('\n')


def read_history(path):
    """ Reads history from a text file. Lines of form:
        region_1    region_2    new_region  score 
    """
    history = []
    with open(path, 'r') as f:
        for line in f:
            str_nums = line.split('\t')
            nums = [float(i) for i in str_nums]
            history.append(nums)
    return history


def load_histories(path): 
    """
    Loads the histories contained in the input folder
    """
    hists = dataio.FileReader(path, exts=['.txt', '.dat', '.data']).extract_files()
    return [read_history(i) for i in hists]


def dump_histories(hists, folder):
    """ Dumpts the input histories in the given folder """
    digits = get_digits(len(hists))
    for (i, h) in enumerate(hists):
        dump_history(h, os.path.join(folder, str(i).zfill(digits) + '.dat'))


def read_history_scores(path):
    return [score for (n1, n2, n3, score) in read_history(path)]


def load_history_values(path):
    """
    Loads the histories contained in the input folder and returns
    the list of values
    """
    total = []
    hists = dataio.FileReader(path, exts=['.txt', '.dat', '.data']).extract_files()
    for i in hists:
        scores = read_history_scores(i)
        total = total + scores
    return total


def thresh_history(hist, thresh):
    """ Thresholds input merge history by eliminating those entries below
    the given threshold """
    return [entry for entry in hist if entry[3] > thresh]


def thresh_histories(folder, thresh, outp):
    """ Given a folder contanining merge histories, generates an updated version
    that deletes all entries whose score are below the threshold"""
    hists = load_histories(folder)
    threshs = [thresh_history(i, thresh) for i in hists]
    dump_histories(threshs, outp)


def get_subcomponents(d, s_id):
    """ Given a dictionary that contains for each new superpixel, the
    children superpixels it is composed, returns the children for a superpixel key """
    if s_id in d:
        return d[s_id]
    else:
        return [s_id]


def remove_superpixel(d, s_id):
    """ Removes superpixel in dictionary, if it exists """
    if s_id in d:
        del d[s_id]


def merge_dataset(folder, histories, thresh, outp):
    """ Thresholds images by merging all supervoxels up to a given threshold 
    :param folder: Folder containing superpixel images
    :param histories: Folder containing the corresponding merge tree histories
    :param thresh: Merge threshold up to which merge regions
    :param outp: Folder where to store resulting images 
    """ 
    dataio.create_dir(outp)
    # Read superpixels and histories
    sps = dataio.FileReader(folder).extract_files()
    hists = dataio.FileReader(histories, exts=['.txt', '.dat', '.data']).extract_files()

    for (s, h) in zip(sps, hists):
        img = merge_superpixels(s, h, thresh)
        name, ext = os.path.splitext(s)
        mh.imsave(os.path.join(outp, os.path.basename(name) + ext), img.astype(float))


def merge_superpixels(superpixels, history, threshold):
    """ Performs the merging of regions up to the given input threshold
        Args:
            superpixels: superpixel image or path to image
            history: merge history or path to merge history
            threshold: Threshold (value included) up to which we merge the regions
    """

    # Read superpixels
    if isinstance(superpixels, str):
        superpixels = mh.imread(superpixels)
    else:
        if not isinstance(superpixels, np.ndarray):
            raise TypeError('Superpixels must be provided in '
                + 'a numpy array or a string')

    # Read merges
    if isinstance(history, str):
        history = read_history(history)
    else:
        if not isinstance(history, list):
            raise TypeError('Merges have to be provided as list or as path')

    # We assume the lines are already sorted
    bins = {}
    for (n0, n1, n2, score) in history:

        # If score greater than threshold, break loop
        if score > threshold:
            break

        # Add merge as combination of two superpixels
        bins[n2] = get_subcomponents(bins, n0) + get_subcomponents(bins, n1)
        # Remove previous subcomponents
        remove_superpixel(bins, n0)
        remove_superpixel(bins, n1)

    print('Building {} superpixels from smaller superpixel'.format(len(bins.keys())))
    print('Superpixels before merging: {}'.format(len(np.unique(superpixels))))

    # Perform mergings
    for i in bins.keys():
        for k in bins[i]:
            pos = np.where(superpixels == k)
            superpixels[pos] = i

    print('Superpixels after merging: {}'.format(len(np.unique(superpixels))))

    return superpixels

