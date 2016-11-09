#!/usr/bin/python

import os
from neuralimg.crag.crag import *
from neuralimg.image import preproc as pr
from neuralimg.crag.merge_mc import *
from neuralimg.dataio import *

""" Crag generation for CREMI datasets  """

THREADS = 6
RESOLUTION = [4, 4, 40]
MERGES = 5


def generate_superpixels(mask, sigma, mems, sps):
    """ Generates the superpixels in the given folder according to parameters """
    proc = pr.DatasetProc(mems)
    proc.read()
    proc.segment(mask, sigma)
    proc.save_data(sps)


def generate_merge_histories(sps, mem, hists):
    """ Generates the merge histories in the given folder """
    MCTreeExtractor(sps, mem).extract(hists)

if __name__ == '__main__':

    root_output = '/DataDisk/morad/all'
    create_dir(root_output)

    # Parameters per sample
    # These are not the only optimal parameters. We have several combinations of 
    # them having similar merge and split errors. Maximum Hausdorff distances to 
    # be tested
    paramsA = {'name': 'A', 'mask': 5, 'sigma': 0.25, 'thresh': 10.5, 'haus': 500 } 
    paramsB = {'name': 'B', 'mask': 9, 'sigma': 0.5, 'thresh': 25.5, 'haus': 500 } 
    paramsC = {'name': 'C', 'mask': 7, 'sigma': 1, 'thresh': 41.0, 'haus': 500 }

    # Paths per sample: Modify them pointing to your folders
    # Recommended to use processed version for the groundtruth
    # pr.join_small()
    # that erases noisy regions (specially in sample A)
    pathsA = {
            'raw': 'data/crag/raw',
            'gt': 'data/crag/gt',
            'mem': 'data/crag/mem' }
    pathsB = {
            'raw': 'data/crag/raw',
            'gt': 'data/crag/gt',
            'mem': 'data/crag/mem' }
    pathsC = {
            'raw': 'data/crag/raw',
            'gt': 'data/crag/gt',
            'mem': 'data/crag/mem' }

    params = [paramsA, paramsB, paramsC]
    paths = [pathsA, pathsB, pathsC]

    # Iterate through samples
    for (par, pat) in zip(params, paths):

        sample_dir = os.path.join(root_output, 'sample_' + par['name'])
        create_dir(sample_dir)

        sps = os.path.join(sample_dir, 'superpixels')
        generate_superpixels(par['mask'], par['sigma'], pat['mem'], sps)

        hists = os.path.join(sample_dir, 'histories')
        generate_merge_histories(sps, pat['mem'], hists)

        # Generate crag
        cragen = CragGenerator(sample_dir)
        cragen.generate_crag(groundtruth=pat['gt'],
            superpixels=sps,
            raws=pat['raw'],
            membranes=pat['mem'],
            max_zlink=par['haus'],
            histories=hists,
            histories_thresh=par['thresh'],
            max_merges=MERGES,
            overwrite=True,
            logs=sample_dir,
            res=RESOLUTION,
            threads=THREADS)

        # Generate best effort
        best_effort_path = os.path.join(sample_dir, 'best_effort')

        cragen.extract_best_effort(loss_type=LossType.ASSIGNMENT,
            best_effort=best_effort_path, 
            logs=sample_dir, 
            overwrite=True, 
            threads=THREADS)


