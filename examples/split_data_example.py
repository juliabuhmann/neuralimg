#!/usr/bin/python

from neuralimg.training import ml
import random
import numpy

""" Splits artificial data into training, validation and testing using different strategies """


def count_class_ratio(original, indices):
    """ Given a set of indices and the original set, computes the
    ratio of classes"""

    labels = [original[i] for i in indices]
    vals = numpy.unique(labels)
    total = float(len(labels))
    return {i: sum(labels == i)/total for i in vals}


def print_split(labels, tr, v, t):
    print('Training size: {}, Validation size: {}, Testing size: {}'.format(len(tr), len(v), len(t)))
    print('Instances per class:')
    print('---- > Training: {}'.format(count_class_ratio(labels, tr)))
    print('---- > Validation: {}'.format(count_class_ratio(labels, v)))
    print('---- > Testing: {}'.format(count_class_ratio(labels, t)))
    print('Original class ratio: {}'.format(count_class_ratio(labels, range(0, len(labels)))))


if __name__ == '__main__':

    nlabels = 5000
    nclasses = 3
    print('Creating artifitial set of {} labels and {} classes...'.format(str(nlabels), str(nclasses)))
    labels = []
    for i in range(0, nlabels):
        labels.append(random.randint(0, nclasses))

    ratio_t = 0.70
    ratio_v = 0.20

    # Testing random mode
    print('\n\n---------------------------------------------------------------')
    print('--------- Random mode \n')
    tr, v, t = ml.split(labels, ratio_t, ratio_v, 'random')
    print_split(labels, tr, v, t)

    print('\n\n---------------------------------------------------------------')
    print('--------- Same instances mode \n')
    # Testing same mode
    tr, v, t = ml.split(labels, ratio_t, ratio_v, 'same')
    print_split(labels, tr, v, t)

    print('\n\n---------------------------------------------------------------')
    print('--------- Preserve ratio mode \n')
    # Testing ratio mode
    tr, v, t = ml.split(labels, ratio_t, ratio_v, 'ratio')
    print_split(labels, tr, v, t)
