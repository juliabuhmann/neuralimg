#!/usr/bin/python

import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.manifold import TSNE


def split(labels, ratio_t, ratio_v, mode, shuffle=True):
    """ Splits the input labels into training, validation and testing
    and returns the corresponding indices. Note that the sets will be
    more consistent as the number of instances grow

    Params
    ------------
    labels: list of integers
        List of labels in the dataset. 
        If mode is random, length of the dataset.
    ratio_t: float [0,1]
        Ratio of training instances
    ratio_v: float[0, 1]
        Ratio of validation instances
    mode: string
        How to split the data:
            - random: Randomly assigns instances to each set. Uses all instances
            - same: All classes are assigned the same number of instances in
                each set. It is bounded by the less common class.
            - ratio: Preserves the ratio of the classes in the original set.
    respect_ratio: boolean
        Whether to respect the ratio of classes in the original dataset
    shuffle:boolean
        Whether to shuffle final indices at the end
    """

    _check_ratio(ratio_t)
    _check_ratio(ratio_v)

    if mode == 'same':
        tr, v, t = _split_same(_positions_map(labels), ratio_t, ratio_v)
    elif mode == 'ratio':
        tr, v, t = _split_ratio(_positions_map(labels), ratio_t, ratio_v)
    elif mode == 'random':
        labels = labels if isinstance(labels, int) else len(labels)
        tr, v, t = _split_random(labels, ratio_t, ratio_v)
    else:
        raise ValueError('Unknown split mode {}'.format(mode))

    if shuffle is True:
        random.shuffle(tr)
        random.shuffle(v)
        random.shuffle(t)

    return tr, v, t


def _positions_map(labels):
    """ Returns the mapping of positions for each label """
    vals = np.unique(labels)
    return { i: np.where(labels == i)[0] for i in vals}


def _split_random(num, ratio_t, ratio_v):
    """ Splits the data into sets in a random fashion """
    ct = int(np.floor(num * ratio_t))
    vt = int(np.floor(num * ratio_v))
    perm = np.random.permutation(num)
    return perm[0:ct], perm[ct:(ct + vt)], perm[(ct + vt):]


def _split_ratio(positions, ratio_t, ratio_v):
    """ Given the positions for each class, splits data preserving
    the class ratio in the original set
    Params
    ---------
    positions: dict
        Key is the label identifier and each entry contains
        the positions in the original dataset
    ratio_t: float
        Training ratio
    ratio_v: float
        Validation ratio
    """

    train, val, test = [[]] * 3
    for i in positions.keys():
        # Split permutation so each class is split according to 
        # the original distribution
        perm = np.random.permutation(len(positions[i]))
        ct = int(np.floor(len(perm) * ratio_t))
        cv = int(np.floor(len(perm) * ratio_v))
        cte = len(perm) - cv - ct
        # Concat results
        train = train + positions[i][perm[0:ct]].tolist()
        val = val + positions[i][perm[ct:(ct + cv)]].tolist()
        test = test + positions[i][perm[(ct + cv):(ct + cv + cte)]].tolist()
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _split_same(positions, ratio_t, ratio_v):
    """ Given the positions for each class, balances the class instances
    for training, validation and testing 
    Params
    ---------
    positions: dict
        Key is the label identifier and each entry contains
        the positions in the original dataset
    ratio_t: float
        Training ratio
    ratio_v: float
        Validation ratio
    """

    # Get minimum represented class
    counts = { i: len(positions[i]) for i in positions.keys()}
    per_class = min(counts.values())
    train, val, test = [[]] * 3
    # Split data so each class is taken 'per_class' instances
    ct = int(np.ceil(per_class * ratio_t))
    cv = int(np.ceil(per_class * ratio_v))
    cte = per_class - ct - cv
    # Iterate through classes so all classes have the same
    # number of instances in each set
    for i in positions.keys():
        perm = np.random.permutation(len(positions[i]))
        train = train + positions[i][perm[0:ct]].tolist()
        val = val + positions[i][perm[ct:(ct + cv)]].tolist()
        test = test + positions[i][perm[(ct + cv):(ct + cv + cte)]].tolist()
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _check_ratio(r):
    """ Check ratio is in interval [0,1] """
    if r < 0.0 or r > 1.0:
        raise ValueError('Ratio must be between 0 and 1, both included')


def generate_image(images, positions, index):
    """ Generates a ready-to.plot annotation box for the ith row
    given the images and positions """
    im = OffsetImage(images[index, ...], zoom=1)
    xy = positions[index, :]
    return AnnotationBbox(im, xy, pad=0)


def tsne_visualization(images, descriptors, subset):
    """ Plots the a subset of high dimensional descriptors into a 2-dimensional
    space , preserving the similarity of the inputs into the new space.
    For more information chec:
        Reference: Visualizing High-Dimensional Data Using t-SNE
        Use: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    :param images: Set of input images
    :param descriptors: Corresponding high dimensional descriptors for the images
    :param subset: Number of random descriptors to plot
    """
    # Obtain representation in new space (parameters to be explored)
    dims = 2
    model = TSNE(n_components=dims)
    new_desc = model.fit_transform(descriptors)

    # Get subset data for ease of visualization
    inds = np.random.permutation(descriptors.shape[0])[0:subset]
    subset_imgs = images[inds, ...]
    subset_desc = new_desc[inds, ...]
    limits = [[subset_desc[:, i].min(), subset_desc[:, i].max()]
                            for i in range(dims)]
    # Send plot
    fig, ax = plt.subplots()
    for i in range(subset):
        ab = generate_image(subset_imgs, new_desc, i)
        ax.add_artist(ab)
    plt.xlim(limits[1])
    plt.ylim(limits[0])
    plt.draw()
    plt.show()


def compute_outlier_edge(values):
    """ Compute value from which we can consider input values as
    outliers """
    upper_quartile = np.percentile(values, 75)
    lower_quartile = np.percentile(values, 25)
    iq_range = upper_quartile - lower_quartile
    return upper_quartile + (1.5 * iq_range)

