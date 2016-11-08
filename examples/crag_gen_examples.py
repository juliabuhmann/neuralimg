#!/usr/bin/python

import os
from neuralimg.crag.crag import *

""" Crag generation examples from the sample data """

if __name__ == '__main__':

    # Data paths
    gt = 'data/crag/gt/'
    sps = 'data/crag/sps/'
    raws = 'data/crag/raw'
    membranes = 'data/crag/mem'
    hists = 'data/crag/sps/histories_mc'
    out_p = 'out/project'
    best_effort = os.path.join(out_p, 'best')

    # Params
    thresh = 0.06   # Set computed thresh
    maxHaus = 800   # Maximum Hausdorff distance
    max_merges = 4
    resolution = [4, 4, 40]
    nthreads = 3

    cragen = CragGenerator(out_p)

    cragen.generate_crag(gt, sps, raws, membranes, max_zlink=maxHaus, 
        histories=hists, histories_thresh=thresh, max_merges=max_merges,
        overwrite=True, logs=out_p, indexes=None, res=resolution, threads=nthreads)

    cragen.extract_best_effort(loss_type=LossType.ASSIGNMENT,
        best_effort=best_effort, logs=out_p, overwrite=True, threads=nthreads)

