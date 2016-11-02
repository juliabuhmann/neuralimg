#!/usr/bin/python

from scipy import ndimage as ndi
from skimage.segmentation import relabel_sequential
import cv2

from scipy.sparse import lil_matrix

import os
import numpy as np
import h5py
from six.moves import xrange  # pylint: disable=redefined-builtin

from neuralimg.image import segment as seg
from neuralimg import dataio

# Border mask functions extracted from repository: 
# https://github.com/cremi/cremi_python

class DatasetProc(object):

    """ Class that applies preprocessing operations on datasets based on volumes """

    def __init__(self, path, group=None, exts=['.png', '.tif', '.tiff']):

        self.path = path
        self.group = group
        self.exts = exts
        self.imgs = None

    def read(self, num=None):
        """Reads image content from the dataset path. If a HDF5 provided, the group
        where images are stored must be provided. Default image extensions are PNG,
        TIF and TIFF. Others can be provided
            :param num: Number of images to read from the dataset. Set to None for reading all"""

        if not os.path.exists(self.path):
            raise ValueError('Path {} does not exist'.format(self.path))

        if os.path.isfile(self.path):
            if not dataio.valid_volume_path(self.path):
                raise ValueError('Invalid extension for file {}'.format(self.path))
            with h5py.File(self.path, 'r') as f:
                dataset = dataio.get_hf_group(f, self.group)
                if num is None:
                    self.imgs = dataset[()]
                else:
                    if num > dataset.shape[0]:
                        raise ValueError('Cannot read more images than the ones available')
                    self.imgs = dataset[0:num][:]
                # Previous code loads all images in memory. Fix this for big 
                # datasets when memory is limited
        else:
            reader = dataio.FileReader(self.path, self.exts)
            self.imgs = reader.read(num)

        if self.imgs.shape[0] == 0:
            raise ValueError('No data has been read')

    def _split_labels(self, index, max_id, diagonal=True):
        """ Splits unconnected regions belonging to same label in the image. Connectivity
        by default considers diagonal directions"""

        print('Splitting labels for image {}'.format(str(index)))

        # Get list of labels different from 0
        gt = self.imgs[index, :, :]
        sps = np.unique(gt)
        non_zero = np.where(sps > 0.0)[0]
        sps = sps[non_zero]

        # Iterate over labels
        for i in sps:

            if diagonal is True:
                (cmps, num) = ndi.label(gt == i, structure=np.ones((3, 3)))
            else:
                (cmps, num) = ndi.label(gt == i)

            # If more than one connected component, split into different labels
            # split into different labels
            if num > 1:
                # Get set of ids (excluding 0) for current label
                label_ids = np.unique(cmps)
                label_ids = label_ids[np.where(label_ids > 0)[0]]
                # Let's keep the first label and add new ones for the rest
                for j in range(1, len(label_ids)):
                    gt[cmps == label_ids[j]] = max_id + 1
                    max_id += 1

    def _check_data(self):
        """ Checks whether at least a single slice in the volume exists """
        if self.imgs is None:
            raise Exception('Image must be read before modifying it')
        if self.imgs.shape[0] == 0:
            raise Exception('No images are contained in the dataset')

    def split_labels(self):
        """ Splits the regions belonging to the same superpixel that are unconnected. 
        Ids are independent of each slice of the volume"""
        self._check_data()
        max_id = self.imgs.max()
        for i in range(0, self.imgs.shape[0]):
            self._split_labels(i, max_id)

    def grow_boundaries(self, w, backg=0):
        """ Grows a boundary between neighboring subregions so pixels between them are mapped
        into the background area 

        Params
        ---------
        w: integer or float
            Width of the boundary to grow
        backg: integer 
            Label of the background
        """
        self._check_data()
        print('Growing boundares for whole volume...')
        create_border_mask(self.imgs, self.imgs, w, backg)

    def compute_supervoxel_stats(self):
        """ Assuming the data read are supervoxels, it computes the stats regarding the
        size of them """
        sizes = []
        for i in range(0, self.imgs.shape[0]):
            sizes = sizes + get_sizes(self.imgs[i, ...])
        return sizes, np.mean(sizes), np.std(sizes)

    def segment(self, mask_size, sigma):
        """ Segments the images using a watershed approach"""
        s = seg.Watershed(sigma, mask_size)
        new_imgs = np.empty(self.imgs.shape, dtype=float)
        for i in range(0, self.imgs.shape[0]):
            new_imgs[i, ...] = s.segment(self.imgs[i, ...])
        self.imgs = new_imgs

    def enhance_edges(self):
        """ This functionality is intended for boundary images. Expands and empowers
        weak edges in order to complete boundaries and avoid merge errors """
        new_imgs = np.empty(self.imgs.shape, dtype=float)
        for i in range(0, self.imgs.shape[0]):
            enhance_edges(self.imgs[i, :, :])
        self.imgs = new_imgs

    def relabel_seq(self):
        """ Sequentially relabels each superpixel within each section """
        new_imgs = np.empty(self.imgs.shape, dtype=float)
        for i in range(self.imgs.shape[0]):
            new_imgs = relabel_sequential(self.imgs[i, :, :])[0]
        self.imgs = new_imgs

    def join_small(self, min_region_area=7):
        """ Eliminates regions in the superpixel images whose region area
        is below the given threshold
        """
        new_imgs = np.empty(self.imgs.shape, dtype=float)
        difs = []
        for i in range(0, self.imgs.shape[0]):
            new_imgs[i, ...], dif = join_small(self.imgs[i, ...], min_region_area)
            difs.append(dif)
        self.imgs = new_imgs
        # Compute and return mean of eliminated regions/image
        mean_elim = np.mean(difs)
        print('Mean regions eliminated per image: %d' % mean_elim)
        return mean_elim

    def normalize(self):
        """ Normalizes data into the interval [0,1] """
        self.imgs = self.imgs.astype(float) / 255.0

    def save_data(self, out_p, group='data/labels', min_digit=5, overwrite=True, int_data=False):
        """ Saves processed data. If path to HDF file provided, a dataset is created inside the
        given group. Otherwise, images are dumped into the folder specified in the path.

        Params:
        ---------
        out_p: string
            Output path. If it corresponds to a valid HDF5 it is stored as a HDF5 dataset.
            Otherwise it is stored in a file.
        group: string
            In case the output corresponds to a HDF5 file, it is the path
            inside the dataset where data needs to be stored. Subgroups must be separated by a /.
            Not used uf dumping into a folder.
        min_digit: integer
            Images are named, in order, with its position in the input volume. This number
            specifies the minimum amount of digits to use in the labeling if dumping data into a folder.
            Not used for HDF5 files.
        overwrite: boolean
            Whether to overwrite existing datasets in the destination path
        int_data: boolean
            By default data is stored as float. If this field is True, it is stored as unsigned integer
            in .png files. Only used if data path is a folder.
        """

        if os.path.exists(out_p) and not overwrite:
            return

        self._check_data()

        if dataio.valid_volume_path(out_p):
            with h5py.File(out_p, 'w') as f:
                dataio.store_hf_group(f, group, self.imgs)
        else:
            dataio.create_dir(out_p)
            if int_data is True:
                dataio.volume_to_folder(self.imgs, out_p, min_digit=min_digit, typ='uint8', ext='.png')
            else:
                dataio.volume_to_folder(self.imgs, out_p, min_digit=min_digit)

class Point:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, y, x, z):
        """ Create a new point at the origin """
        self.y = y
        self.x = x
        self.z = z

    def __str__(self):
        return '(' + str(self.y) + ',' + str(self.x) + ',' + str(self.z) + ')'


class BoundingBox:
    """ BoundingBox represents the minimal square a superpixel is embedded into """

    # Points are input in a clock-wise order
    def __init__(self, tl=None, br=None):
        self.tl = tl
        self.br = br

    def depth(self):
        return abs(self.tl.z - self.br.z + 1)

    def height(self):
        return abs(self.br.y - self.tl.y + 1)

    def width(self):
        return abs(self.tl.x - self.br.x + 1)

    def from_img(self, img, sp_id, depth=None):
        """ Fills bounding box info from image and corresponding cluster identifier """
        values = img == sp_id
        if values.any() is False:
            raise Exception("There is no cluster with the id {}'".format(str(sp_id)))
        pos = np.where(values)
        d = 0 if depth is None else depth
        self.tl = Point(min(pos[0]), min(pos[1]), d)
        self.br = Point(max(pos[0]), max(pos[1]), d)

    def max_bb(self, other):
        """ Given current bounding box and a second one, define maximal bounding box for both """
        min_h = min(self.tl.y, other.tl.y)
        min_w = min(self.tl.x, other.tl.x)
        min_z = min(self.tl.z, other.tl.z)
        max_z = min(self.br.z, other.br.z)
        max_h = max(self.br.y, other.br.y)
        max_w = max(self.br.x,  other.br.x)
        return BoundingBox(Point(min_h, min_w, min_z), \
                           Point(max_h, max_w, max_z))

    def _pad_offsets(self, limits_down, limits_up, padding):
        """ Returns the padded bounding box offsets. Ignores depth """

        # Get limits of the pad
        min_row, min_col = limits_down[0:2]
        max_row, max_col = limits_up[0:2]

        # Default pad is a function of the bounding box size
        left, right = [np.ceil(self.width() * (padding - 1) / 2.0)] * 2
        up, down = [np.ceil(self.height() * (padding - 1) / 2.0)] * 2

        # Compute offsets
        left = int(min(self.tl.x - min_col, left))
        right = int(min(max_col - self.br.x, right))
        up = int(min(self.tl.y - min_row, up))
        down = int(min(max_row - self.br.y, down))

        return left, right, down, up

    def preserve_ratio(self, limits_down, limits_up, im_shape):
        """ Updates the bounding box so its aspect ratio preserves the
        ratio of the input dimensions. Ignores depth.
        Params
        ----------
            limits_down: tuple
               Lower bound image limits (y, x)
            limits_up: tuple
                Upper bound image limits (y, x)
            im_shape: list or ndarray
                Target dimensions (y, x)
        """
        h, w = self.height(), self.width()
        ratio = im_shape[0] / im_shape[1]

        if h > w:
            # Apply changes to w
            offset = (h / ratio) - w
            axis = 1
        else:
            # Apply changes to h
            offset = (w * ratio) - h
            axis = 0

        half = int(offset/2)

        # Absolute new positions
        current_down = self.tl.x if axis == 1 else self.tl.y
        current_up = self.br.x if axis == 1 else self.br.y

        # Leftover as positions out of range
        leftover_down = 0 if current_down - half > limits_down[axis] \
            else half - current_down
        leftover_up = 0 if current_up + half < limits_up[axis] \
            else half - (limits_up[axis] - current_up)

        if leftover_down > 0 and leftover_up > 0:
            raise ValueError('Input volume is too small: padding' +
                ' breaks both limits')

        half_down = half - leftover_down + leftover_up
        half_up = half - leftover_up + leftover_down

        # Update positions
        if axis == 1:
            self.tl.x -= half_down
            self.br.x += half_up
        else:
            self.tl.y -= half_down
            self.br.y += half_up

    def update_bb(self, limits_down, limits_up, padding):
        """ Updates the bounding box using padding but respecting the
        limits of the input volume. Pads around x and y and preserves depth
        Params
        ----------
            :param limits_down: tuple
               Lower bound image limits (y, x)
            limits_up: tuple
                Upper bound image limits (y, x)
            padding: double from 0 to 1
                Ratio of padding that will be added around the bounding box of
                the dataset samples
        """
        left, right, down, up = self._pad_offsets(limits_down, limits_up, padding)
        self.tl = Point(self.tl.y - up, self.tl.x - left, self.tl.z)
        self.br = Point(self.br.y + down, self.br.x + right, self.br.z)

    def diff(self, other):
        """ Computes the difference between the bounding box and another
        as the shift in each direction that we should move to create the
        current bounding box from the given one """
        return other.tl.x - self.tl.x, self.br.x - other.br.x, \
            other.tl.y - self.tl.y, self.br.y - other.br.y, \
            other.tl.z - self.tl.z, self.br.z - other.br.z

    def __str__(self):
        return str(self.tl) + ';' + str(self.br)


def join_small(im, min_area, mask=3, it=1):
    """ Given an image and a region area, joins the regions whose size
    are below the minimum with the biggest adjacent region. We assume there
     are no repeated regions """
    if im.dtype is not np.int:
        print('Converting image into integer')
        im = im.astype(int)

    # Generate mask for each region and dilate it
    # Regions are assumed to be unique
    regions = np.unique(im)
    inv_map = {r: i for (i, r) in enumerate(regions)}
    kernel = np.ones((mask, mask), np.uint8)
    masks = [(im == i).astype(np.uint8) for i in regions]
    sizes = [sum(sum(i)) for i in masks]
    dilated = [cv2.dilate(m.astype(np.uint8), kernel, iterations=it) for m in masks]

    # Return context regions: subset of regions in the context
    # This is used to retrict neighboring maps and speed up
    context = _get_around(im, regions)

    # Create adjacency matrix for regions
    adj_mat = _create_adjacency_mat(dilated, context, inv_map)

    reg_before = len(regions)
    print('Regions before: %d' % reg_before)
    # Eliminate those below area threshold
    for ind, s in enumerate(sizes):
        if s < min_area:
            # Merge with bigger adjacent region
            adjacent_regions = [i for i in adj_mat.getrow(ind).nonzero()][1]
            adjacent_sizes = [sizes[i] for i in adjacent_regions]
            chosen = adjacent_regions[adjacent_sizes.index(max(adjacent_sizes))]
            im[np.where(masks[ind])] = regions[chosen]
    reg_after = len(np.unique(im))
    print('Regions after: %d' % reg_after)

    return im, reg_before - reg_after


def _get_around(img, regions, padding=1.1):
    """ Returns a dictionary with the regions each region has around """
    result = {}
    for i, r in enumerate(regions):
        bb = BoundingBox()
        bb.from_img(img, r)
        bb.update_bb([0, 0], img.shape, padding)
        result[i] = np.unique(img[bb.tl.y:bb.br.y+1, bb.tl.x:bb.br.x+1])
    return result


def _create_adjacency_mat(masks, ctx, rmap):
    """ Given the input masks, creates and adjacency matrix where:
    (i, j) = 1 is i and j are adjacent regions and 0 otherwise
    Note that i and j are indices and not the region identifier.
    Not all other regions are considered but those in the context """
    mat = lil_matrix((len(masks), len(masks)), dtype=np.bool)
    for i, m in enumerate(masks):
        context_regions = ctx[i]
        for cr in context_regions:
            j = rmap[cr]
            diferent = i != j
            adjacent = _adjacent(masks[i], masks[j])
            if diferent and adjacent:
                mat[i, j] = True
                mat[j, i] = True
    return mat


def _adjacent(mask1, mask2, thresh=1):
    """ Returns whether two regions are adjacent by thresholding
    the overlapped area after dilating them. By default it is set to 1 """
    overlap = sum(sum(np.bitwise_and(mask1, mask2)))
    return overlap > thresh


def preserve_ratio(im_shape, h, w):
    """ Changes bounding box dimensions so a cut can preserve the target
    image size without evident distortions
    :param im_shape: Shape of the original image
    :param h: Height of the bounding box
    :param w: Width of the bounding box
    :return: Offset to each side of the axis to enlarge (0: y, 1:x) """

    ratio = im_shape[0] / im_shape[1]

    if h > w:
        # Apply changes to w
        offset = (h / ratio) - w
        axis = 0
    else:
        # Apply changes to h
        offset = (w * ratio) - h
        axis = 1

    half = int(offset/2)
    return half, axis


def enhance_edges(img, low_thresh=0, high_thresh=150, kernel_size=5, dilations=1):
    """ Enhances the edges of the input image """
    canny = cv2.Canny(img, low_thresh, high_thresh)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=dilations)
    return 255 - dilated


def create_border_mask(input_data, target, max_dist, background_label, axis=0):
    """
    Overlay a border mask with background_label onto input data.
    A pixel is part of a border if one of its 4-neighbors has different label.
    This code has been extracted from the gollowing repository:
        -> https://github.com/cremi/cremi_python
    Parameters
    ----------
    input_data : h5py.Dataset or numpy.ndarray - Input data containing neuron ids
    target : h5py.Datset or numpy.ndarray - Target which input data overlayed with border mask is written into.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    background_label : int - Border mask will be overlayed using this label.
    axis : int - Axis of iteration (perpendicular to 2d images for which mask will be generated)
    """
    sl = [slice(None) for d in xrange(len(target.shape))]

    for z in xrange(target.shape[axis]):
        sl[axis] = z
        border = create_border_mask_2d(input_data[tuple(sl)], max_dist)
        target_slice = input_data[tuple(sl)] if isinstance(input_data, h5py.Dataset) else np.copy(input_data[tuple(sl)])
        target_slice[border] = background_label
        target[tuple(sl)] = target_slice


def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(image == padded[:-2, 1:-1], image == padded[2:, 1:-1]),
        np.logical_and(image == padded[1:-1, :-2], image == padded[1:-1, 2:])
    )

    distances = ndi.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
    )

    return distances <= max_dist


def get_sizes(sp):
    """ Get the list of sizes of the superpixels in the input image """
    values = np.unique(sp)
    return [sum(sum(sp == i)) for i in values]


class Contingency:

    """ Computes the contigency table between two images. Labels are treated as
    integers """ 

    def __init__(self, gt, img):
        self.gt_set = np.unique(gt)
        self.sp_set = np.unique(img)
        self.id_gt = {}
        self.id_sp = {}
        self.table = np.zeros((len(self.gt_set), len(self.sp_set)))
        self.initialize_id_maps()

    def initialize_id_maps(self):
        """ We cannot ensure that superpixels have consecutive labeling and, therefore, we must
        keep track of the pair (id, matrix index) """
        self.id_gt = { int(self.gt_set[i]): i for i in range(len(self.gt_set))}
        self.id_sp = { int(self.sp_set[i]): i for i in range(len(self.sp_set))}

    def add_overlap(self, gti, spi):
        x = self.id_gt[int(gti)]
        y = self.id_sp[int(spi)]
        self.table[x, y] += 1

    def contigency_table(self):
        return self.table


def overlap_images(gt, img, next_id, overlap_bg=False):
    """ Returns the overlapping of two images so the result is the assignation, in the
    given image, the most probable groundtruth superpixel. If overlap with background chose,
    in case an area only overlaps with the background, it results in a black area.
    Otherwise, a new unused segment is created starting from the given next identifier """

    if img.shape != gt.shape:
        raise Exception("Both images shoud have same size")

    result = np.empty((img.shape))
    overlaps = {}
    max_sov = 0
    ct = Contingency(gt, img)

    # find overlaps of superpixels with solution segments
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):

            spv = img[x,y]
            sov = gt[x,y]

            # Negative ids are mapped into 0 (does this happen?)
            max_sov = max(sov, max_sov)

            # Check if superpixel id has entry
            if not overlaps.has_key(spv):
                overlaps[spv] = {}

            # Check if groundtruth has entry in sp key
            if not overlaps[spv].has_key(sov):
                overlaps[spv][sov] = 0

            ct.add_overlap(sov, spv)
            overlaps[spv][sov] += 1

    # Overlaps -> keys: superpixel ids, values: dicts for gt that overlap + 
    # weight

    # find replace values for each superpixels (max overlap segment id)
    replace = {}
    for (spv, ov) in overlaps.iteritems():

        best_sov = None
        max_ov  = 0

        # Get maximal non-background overlapped superpixel
        for (sov, n) in ov.iteritems():
            if n >= max_ov and sov != 0: # ignore background
                max_ov = n
                best_sov = sov

        if best_sov is not None:
            replace[spv] = best_sov
        else:
            if overlap_bg is True:
                replace[spv] = 0
            else:
                replace[spv] = next_id
                next_id += 1

    # replace superpixel values
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            result[x,y] = replace[img[x,y]]

    return result, ct.contigency_table(), next_id


def overlap_set(gts_folder, sps_folder, shift, output, bg=False, threads=5):
    """ Overlaps the set of images in the folders given the groundtruth and return
    the resulting stats based on TED
        :param gts_folder: Groundtruth folder
        :param sps_folder: Superpixel folder
        :param shift: TED shift parameter
        :param output: Output folder where to store the overlapped images
        :param bg: Whether to consider background in the stats
        :param threads: Number of threads to use in TED computation
    """

    if not os.path.isdir(output):
        os.mkdir(output)

    # Read folders
    sps = FileReader(sps_folder).read()
    gts_paths = FileReader(gts_folder).extract_files()
    gts_files = FileReader(gts_folder).read()
    next_id = max([im.max() for im in gts_files]) + 1

    # Initialize stats
    stats = {'VOI split': 0, 'VOI merge': 0}

    for i in range(gts.shape[0]):

        print('Computing overlap for groundtruth ' + i + '...\n')

        overlap, cont, next_id = overlap_images(gts[i], sps[i], next_id)

        print('Next identifier is {}'.format(str(next_id)))

        # Save image into output
        ov_path = os.path.join(output, str(i).zfill(5) + '.tif')
        mh.imsave(ov_path, overlap.astype(float))

        # Compute measures
        current_stats = se.call_ted(ov_path, gts_files[i], shift, split_background=bg, threads=threads)
        stats['VOI split'] += current_stats['VOI split']
        stats['VOI merge'] += current_stats['VOI merge']

        print(stats)

    # Compute means of the stats computed
    for k in stats:
        stats[k] = stats[k] / float(len(gts_files))

    return stats
