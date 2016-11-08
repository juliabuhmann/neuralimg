#!/usr/bin/python

import os
from neuralimg.crag.crag import *
from neuralimg.image import preproc as pr
from neuralimg.crag.merge_mc import *

""" Crag generation example from the sample data """

if __name__ == '__main__':

    # Segmentation parameters
    mask = 5
    sigma = 0.5
    thresh = 0.060

    # Data
    gt = 'data/crag/gt/'
    raws = 'data/crag/raw'
    membranes = 'data/crag/mem'

    # Crag parameters
    maxHaus = 500   # Maximum Hausdorff distance
    res = [4, 4, 40]
    nthreads = 5
    max_merges = 4
    out_p = 'out/project'

    # Intermediate paths
    sps_folder = 'sps'
    hists = 'hists'
    mem_norm = 'mem_norm'
    best_effort_path = os.path.join(out_p, 'best_effort')

    # Segment data
    proc = pr.DatasetProc(membranes)
    proc.read()
    proc.segment(mask, sigma)
    proc.save_data(sps_folder)

    # Generate normalized membranes [0,1]
    proc = pr.DatasetProc(membranes)
    proc.read()
    proc.normalize()
    proc.save_data(mem_norm)

    # MC Extractor
    MCTreeExtractor(sps_folder, mem_norm).extract(hists)

    # Generate crag, extract features and best effort
    cragen = CragGenerator(out_p)

    cragen.generate_crag(gt, sps_folder, raws, mem_norm, max_zlink=maxHaus, 
        histories=hists, histories_thresh=thresh, max_merges=max_merges,
        overwrite=True, logs=out_p, indexes=None, res=res, threads=nthreads)

    cragen.extract_best_effort(loss_type=LossType.ASSIGNMENT,
        best_effort=best_effort_path, logs=out_p, overwrite=True, threads=nthreads)

