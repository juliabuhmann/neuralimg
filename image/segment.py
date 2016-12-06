#!/usr/bin/python

import mahotas as mh
import numpy as np
from scipy import ndimage as ndi
try:
    import cv2
    check_cv2 = True
except ImportError:
    print 'Open CV not installed or not found'
    check_cv2 = False

import abc
import sys

from skimage.segmentation import felzenszwalb, slic, random_walker, quickshift
from skimage.util import img_as_float

v = sys.version_info
if check_cv2:
    if v[0] < 3:
        CV_DIST = cv2.cv.CV_DIST_L2
    else:
        CV_DIST = cv2.DIST_L2


""" Class containing different image segmentation methods """


class Segmentation:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def segment(self, img):
        "Returns the segmented image according to the subclass algorithm"

    @abc.abstractmethod
    def get_parameters_dict(self):
        "Returns the configuration of the segmentation as a dictionary"


class Watershed(Segmentation):

    """ Segmentation algorithm that thresholds the input image and applies a distance transform
    to identify the core of the neuron images. Maxima is taken from this transform to identify
    as possible neurons as seeds. Connected components are joined and ten, watershed algorithm 
    is applied on a blurred version of  the original image using the markers found.
        Params:
            - sigma_ws: sigma to use when blurring the original image before the 
              watershed procedure
            - mask_size: mask for the computation of the local maxima  """

    def __init__(self, sigma_ws, mask_size):
        Segmentation.__init__(self)
        self.sigma_ws = sigma_ws
        self.mask_size = mask_size

    def segment(self, img):
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Distance transform on the thresholded image
        # 2 possible distances (L1 and L2) and 2 possible masks (3,5)
        # Fixed L1 due to better experiment results. For L1, mask 3 and 5 have 
        # same result (as stated in documentation)
        dist_transform = cv2.distanceTransform(thresh, CV_DIST, 5)
        # Obtain local maximas given a ms x ms mask
        maxima = mh.regmax(dist_transform, np.ones((self.mask_size, self.mask_size)))

        # Connect areas and identify seeds. Better results using full
        # connectivity
        (seeds, num_seeds) = ndi.label(maxima, structure=np.ones((3, 3)))

        # Uses watershed algorithm for segmentation (fills basins)
        # Regions with adjacent catchment basins are constructed.
        # Usually produces oversegmentation of the image. Works on the
        # gradient image
        membrane_watersheds = mh.gaussian_filter(img, self.sigma_ws)

        return mh.cwatershed(membrane_watersheds, seeds)

    def get_parameters_dict(self):
        return { 'SIGMA_WS': self.sigma_ws, 'MASK_SIZE': self.mask_size }


class Fz(Segmentation):

    """ Felzenszwalb segmentation algorithm """

    def __init__(self, scale, sigma, min_size):
        Segmentation.__init__(self)
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

    def segment(self, img):
        inp = img_as_float(img)
        segments_fz = felzenszwalb(inp, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        return segments_fz

    def get_parameters_dict(self):
        return { 'SCALE': self.scale, 'SIGMA': self.sigma, 'MIN_SIZE': self.min_size }


class RandomWalker(Segmentation):

    " Random walker segmentation algorithm """

    def __init__(self, beta, tolerance):
        Segmentation.__init__(self)
        self.beta = beta
        self.tolerance = tolerance

    def segment(self, img):
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(thresh, cv2.cv.CV_DIST_L2, 3)
        maxima = mh.regmax(dist_transform, np.ones((3, 3)))
        (markers, ret3) = ndi.label(maxima)
        return random_walker(img, markers, beta=self.tolerance, mode='cg', tol=self.tolerance)

    def get_parameters_dict(self):
        return { 'BETA': self.beta, 'TOLERANCE': self.tolerance }


class Slic(Segmentation):

    "  SLIC segmentation algorithm """

    def __init__(self, n_seg, compactness, sigma):
        Segmentation.__init__(self)
        self.n_segments = n_seg
        self.compactness = compactness
        self.sigma = sigma

    def segment(self, img):
        im = img_as_float(img)
        result = slic(im, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma,
            max_iter = 10, convert2lab=False)
        return result.astype(np.uint8)

    def get_parameters_dict(self):
        return { 'N_SEGMENTS': self.n_segments, 'COMPACTNESS': self.compactness, 'SIGMA': self.sigma }


class Quickshift(Segmentation):

    "  Quickshift segmentation algorithm (similar to meanshift) """

    def __init__(self, kernel_size, max_dist, sigma, ratio):
        Segmentation.__init__(self)
        self.k_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        self.sigma = sigma

    def segment(self, img):
        im = img_as_float(img)
        segments_quick = quickshift(im, kernel_size=self.k_size, max_dist=self.max_dist, 
            ratio=self.ratio, sigma = self.sigma, convert2lab=False)
        return segments_quick.astype(np.uint8)

    def get_parameters_dict(self):
        return { 'KERNEL_SIZE': self.k_size, 'MAX_DIST': self.max_dist, 'RATIO': self.ratio,
            'SIGMA': self.sigma }

