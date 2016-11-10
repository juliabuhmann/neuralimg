#!/usr/bin/python

from scipy.misc import imresize

import mahotas as mh

import numpy as np
import os
import shutil
import tempfile

import neuralimg.dataio as dataio
from neuralimg.image import preproc as pr

from pycmc import *


""" Crag utilities class """


class DChannel(object):

    RAW = 'raw'
    BINARY = 'binary'
    INTENSITY = 'intensity'


def read_crag(crag_path, solution_name='best-effort'):
    """Reads the CRAG (Candidate Region Adjacency) data

    Parameters
    ----------
    crag_path : string
        Path where the crag is stored
    solution_name: string
        Name of the solution to retrieve. If no solution is wanted
        set to None
    Returns
    ----------
    crag: Crag
        Crag main, structure
    volumes: Set of volumes
        Volumes in the crag
    ein: Volume
        Intensity volume
    ebb: Volume
        Boundary volume
    """

    # Check extension is correct
    if not dataio.valid_volume_path(crag_path):
        raise ValueError('Wrong extension for CRAG file {}. Must be valid HDF5 extension'.format(crag_path))

    print('Retrieving CRAG ...')

    # Read crag
    store = Hdf5CragStore(crag_path)
    crag = Crag()
    store.retrieveCrag(crag)

    # Volumes references
    volume_store = Hdf5VolumeStore(crag_path)

    # Gest proposed solution, if requested
    if solution_name is None:
        crag_solution = None
    else:
        # TODO: check if name exists when wrapper bug is solved
        #solution_names = store.getSolutionNames()
        #if solution_name not in solution_names:
        #    raise ValueError('Requested {} is not in set 
        #    {}'.format(solution_name,
        #        solution_names))
        crag_solution = CragSolution(crag)
        store.retrieveSolution(crag, crag_solution, solution_name)

    # Read volumes
    volumes = CragVolumes(crag)
    store.retrieveVolumes(volumes)

    # Read each of/pad_bb the volumes stored in the CRAG
    ebb = ExplicitVolume_f()
    ein = ExplicitVolume_f()
    volume_store.retrieveBoundaries(ebb)
    volume_store.retrieveIntensities(ein)

    return crag, volumes, crag_solution, ein, ebb


def get_slices(volumes, nodes):
    """ Get slices that are connected by the input node """
    return [volumes.getVolume(i) for i in nodes]


def get_opposite(crag, node):
    """ Gets the nodes connected through the input node """
    adj_edges = [e for e in crag.adjEdges(node)]
    return [crag.oppositeNode(node, e) for e in adj_edges]


def get_ordered(nodes, vol):
    """ Returns the input nodes in ascending sorted order of depth
     given the reference input volume"""
    depths = [vol.getVolume(n).getBoundingBox().max().z() for n in nodes]
    zipped = zip(nodes, depths)
    zipped.sort(key=lambda t: t[1])
    return [n for (n, d) in zipped]


def from_wrapper(bb):
    """ Converts bounding box from wrapper into python Bounding Box """
    tl = pr.Point(bb.min().__getitem__(1), bb.min().__getitem__(0))
    br = pr.Point(bb.max().__getitem__(1), bb.max().__getitem__(0))
    bounding = pr.BoundingBox(tl, br)
    return bounding


def to_wrapper(bounding, minz, maxz):
    """ Converts python bounding box into wrapper format """
    # Build new bounding box with the updated coordinates
    new_bb = box_f3()
    new_bb.min().__setitem__(0, bounding.tl.x)
    new_bb.min().__setitem__(1, bounding.tl.y)
    new_bb.min().__setitem__(2, minz)
    new_bb.max().__setitem__(0, bounding.br.x)
    new_bb.max().__setitem__(1, bounding.br.y)
    new_bb.max().__setitem__(2, maxz)
    return new_bb


def pad_bb(volume, bb, padding):
    """ Computes a new bounding box given the input one and the padding
    using the reference volume """

    # Create bounding box from volume
    bounding = from_wrapper(bb)

    # Pad bounding box
    limits_down = [volume.getBoundingBox().min().y(), volume.getBoundingBox().min().x(),
                  volume.getBoundingBox().min().z()]
    limits_up = [volume.getBoundingBox().max().y(), volume.getBoundingBox().max().x(),
                volume.getBoundingBox().max().z()]
    bounding.update_bb(limits_down, limits_up, padding)

    # Build new bounding box with the updated coordinates
    return to_wrapper(bounding)


def preserve_ratio(bb, vol, im_shape):
    """ Changes bounding box dimensions so a cut can preserve the target
    image size without evident distortions"""
    bounding = from_wrapper(bb)
    limits_up = [vol.getBoundingBox().max().y(), vol.getBoundingBox().max().x()]
    limits_down = [vol.getBoundingBox().min().y(), vol.getBoundingBox().min().x()]
    bounding.preserve_ratio(limits_down, limits_up, im_shape)
    return to_wrapper(bounding)


def get_nassigned_nodes(crag, volumes, solution, node):
    """ Returns the list of nodes the input node is connected to through
    an assignment node that is not part of the solution """
    # Get assignment nodes connected to input which are out of the solution
    op_nodes = get_opposite(crag, node)
    assign_nodes = [n for n in op_nodes if CragNodeType.AssignmentNode == crag.type(n)]
    no_solution = [n for n in assign_nodes if not solution.selected(n)]

    # We consider section as the maximum in the bounding box
    node_z = volumes.getVolume(node).getBoundingBox().max().z()

    # Get slices connected to non-solution assignment node
    result = []
    for n in no_solution:

        # Get slices connected to non-solution assignment node
        slices = get_opposite(crag, n)

        # One of the slices is the input itself, append the other
        non_inp_slices = [volumes.getVolume(i) for i in slices
            if crag.id(i) != crag.id(node)]

        # Return volumes that are from following section only
        non_inp_slices_next = [i for i in non_inp_slices
            if i.getBoundingBox().max().z() == node_z + i.getResolution().z()]
        result = result + non_inp_slices_next

    return result


def obtain_slice(vol, bb, ev, tmpath, binary=None):
    """ Extracts a single channel image from a section using a bounding box
    Parameters
    ----------
    vol: Volume
        Volume that contains the slice
    bb: Boundinbog box
        Bounding box that specifies area to cut
    ev: Volume
        Volume where to save the cut
    tmpath: string
        Temporal folder path
    binary: BoundingBox
        If binary, original bounding box of the section (withoud padding
        or aspect ratio normalisation). If it is not a binary volume, set to None
    """

    # Cut volume
    vol.cut(bb, ev)
    outpath = os.path.join(tmpath, 'slices')
    saveVolume(ev, outpath)

    # Obtain image from generated folder
    imgpath = os.path.join(outpath, 'slice_00000000.tif')
    if not os.path.exists(imgpath):
        raise IOError('Slice has not been generated. Cannot continue')

    # Clean folder
    open_img = mh.imread(imgpath)
    img = open_img.copy()
    shutil.rmtree(outpath)

    # Add some exception for the case of binary, since wrapping do not
    # cut binary images according to bounding box
    if binary is not None:

        # Get offsets for all directions
        left, right, down, up, _, _ = diff_bb(bb, binary)

        # Reshape image according to resolution
        res_x, res_y = img.shape[1] * vol.getResolution().x(), \
            img.shape[0] * vol.getResolution().y()
        res_img = imresize(img, (int(res_y), int(res_x)), 'nearest')

        # Apply mask on resulting image
        new_img = np.zeros((int(down) + int(res_y) + int(up),
            int(left) + int(res_x) + int(right)))
        new_img[int(down):int(down) + int(res_y), int(left):int(left) + int(res_x)] = res_img
        img = new_img

    return img


def resize_slice(img, imh, imw, normalise=True):
    """ Resizes input image according provided size and normalizes it, if requested
    Parameters
    ----------
    img: ndarray
        Slice image
    imh: int
        Image height to use for resizing the input slice
    imw: int
        Image width to use for resizing the input slice
    normalise: boolean
        Whether to normalise or not the input image
    """
    res = imresize(img, (imh, imw), 'nearest')
    if normalise == True:
        if res.max() == 0:
            print('Found very blurry region, clipping to zeros to avoid error')
            res = np.zeros((res.shape[0], res.shape[1]))
        else:
            res = res / float(res.max())
    return res


def get_resized_slice(vol, bb, ev, tmpath, imh, imw, normalise=True, binary=None):
    """ Extracts a section channel from the volume and resizes it to the given size.
    If requested, normalizes the resulting image """
    sl = obtain_slice(vol, bb, ev, tmpath, binary)
    return resize_slice(sl, imh, imw, normalise)


def from_wrapper(bb):
    """ Converts bounding box from wrapper into python Bounding Box """
    tl = pr.Point(bb.min().__getitem__(1), bb.min().__getitem__(0), bb.min().__getitem__(2))
    br = pr.Point(bb.max().__getitem__(1), bb.max().__getitem__(0), bb.max().__getitem__(2))
    bounding = pr.BoundingBox(tl, br)
    return bounding


def to_wrapper(bounding):
    """ Converts python bounding box into wrapper format """
    # Build new bounding box with the updated coordinates
    new_bb = box_f3()
    new_bb.min().__setitem__(0, bounding.tl.x)
    new_bb.min().__setitem__(1, bounding.tl.y)
    new_bb.min().__setitem__(2, bounding.tl.z)
    new_bb.max().__setitem__(0, bounding.br.x)
    new_bb.max().__setitem__(1, bounding.br.y)
    new_bb.max().__setitem__(2, bounding.br.z)
    return new_bb


def diff_bb(padded_bb, src_bb):
    """ Returns the difference between two bounding boxes """
    bb1 = from_wrapper(padded_bb)
    bb2 = from_wrapper(src_bb)
    return bb1.diff(bb2)


def compute_bbs(slices, padding, imh, imw, vol):
    """ Computes padding around each slice node and returns their
    bounding boxes according to the desired image patch
    Args:
        - slices: List of slices to process
        - padding: padding around each slice. Padding is disabled if
            value is below or equal to 1
        - imh: Desired image height
        - imw: Desired image width
        - vol: Reference volume to use for considering limits
    Returns:
        Bounding boxes after padding and aspect ratio normalization
        and original bounding boxes
    """
    # Compute bounding boxes for current slices
    bbs = [bb.getBoundingBox() for bb in slices]

    # Add padding to bounding box if requested
    # We assume the bounding box will be the same in all volumes
    if padding > 1:
        bbs = [pad_bb(vol, bb, padding) for bb in bbs]

    # Modify bounding boxes to avoid distortions after resizing
    return [preserve_ratio(bb, vol, [imh, imw]) for bb in bbs]


def print_bb(bb):
    print('Bounding box:\n')
    print('----- Min: ({}, {}, {})'.format(bb.min().x(), bb.min().y(), bb.min().z()))
    print('----- Max: ({}, {}, {})'.format(bb.max().x(), bb.max().y(), bb.max().z()))
    print('----- Size: ({}, {}, {})'.format(bb.width(), bb.height(), bb.depth()))


def _get_connected(crag, node, type_n):
    """ Returns the list of nodes from a type the input node is connected to """
    op = get_opposite(crag, node)
    return [i for i in op if type_n == crag.type(i)]


def get_connected_an(crag, node):
    """ Returns the list of assignment nodes the node is connected to """
    return _get_connected(crag, node, CragNodeType.AssignmentNode)


def get_connected_nan(crag, node):
    """ Returns the list of no assignment nodes the node is connected to """
    return _get_connected(crag, node, CragNodeType.NoAssignmentNode)


def get_other_slice(crag, node, an):
    """ Given a slice node and an assignment node, returns the other slice node """
    op = get_opposite(crag, an)
    return op[0] if crag.id(node) != crag.id(op[0]) else op[1]


def get_depth(node, volumes):
    """ Returns the depth of the node """
    return volumes.getVolume(node).getBoundingBox().min().z()


def get_slice(volumes, n):
    """ Returns the slice from the slice node """
    return volumes.getVolume(n)


def get_connected_slices(crag, volumes, solution, node_id, mode='forward'):
    """ Returns a list of pairs, where:
        - First object is the section that is connected with the input node 
          through an assignment node
        - Second object is a boolean that indicates whether the connection is 
          part of the solution proposed
        :param crag: CRAG object
        :param volumes: CRAG volumes object
        :param solution: CRAG solution object
        :param node_id: Slice reference node identifier
        :param mode: Way connection is searched. Valid modes:
            - forward: Searching for connections in next section
            - backward: Searching for connections in next section
            - all: Searching any connections
    """
    if mode not in ['forward', 'backward', 'all']:
        raise ValueError('Unknown mode %s. Valid are: forward, backward and all' % mode)

    node = crag.nodeFromId(int(node_id))
    current_depth = get_depth(node, volumes)
    connected = []

    ans = get_connected_an(crag, node)
    for an in ans:
        n1, n2 = get_opposite(crag, an)
        other = get_other_slice(crag, node, an)
        other_depth = get_depth(other, volumes)
        if (other_depth < current_depth and mode == 'backward') \
            or (other_depth > current_depth and mode == 'forward') \
            or mode == 'all':
            connected.append([volumes.getVolume(other), solution.selected(an)])

    return connected


def build_image(section, ein, ebb, clabels, positions, imh, imw, padding, norm):
    """ Builds the corresponding image for a slice node in a CRAG
        :param section: Crag node
        :param ein: Intensity/raw volume
        :param ebb: Boundary volume
        :param clabels: Channels to use for the data
        :param positions: Positions (order) of the input channels
        :param imh: Height of the instance
        :param imw: Width of the 
        :param padding: Padding to use around a node
        :param norm: Whether to normalise channels into [0, 1]
    """
    # Create temporary path and create numpy result array
    tmpath = tempfile.mkdtemp()
    img = np.empty((len(clabels), imh, imw))
    for i in range(len(clabels)):

        # Iterate through desired channels
        if DChannel.INTENSITY == clabels[i]:
            vol, ev, binary = ein, ExplicitVolume_f(), None
        elif DChannel.RAW == clabels[i]:
            vol, ev, binary = ebb, ExplicitVolume_f(), None
        elif DChannel.BINARY == clabels[i]:
            vol, ev, binary = section, CragVolume(), section.getBoundingBox()
        else:
            raise ValueError('Unknown channel type %s' % clabels[i])

        # Compute bounding box and cut and resize accordingly
        bb = compute_bbs([section], padding, imh, imw, ein)[0]
        img[positions[i], ...] = get_resized_slice(vol, bb, ev, tmpath, 
            imh, imw, norm, binary)

    return img

