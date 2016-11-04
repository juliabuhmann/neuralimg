#!/usr/bin/python

from neuralimg.crag import crag as cr
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
    maxHaus = 800   # Maximum Hausdorff distance
    create_conf = 'data/crag/config/create_training_project.conf'
    features_conf = 'data/crag/config/extract_training_features.conf'
    effort_conf = 'data/crag/config/extract_best-effort.conf'
    out_p = 'out/project'

    # Intermediate paths
    sps_folder = 'sps'
    hists = 'hists'
    mem_norm = 'mem_norm'
    best_effort_path = 'best_effort'

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
    cragen = cr.CragGenerator(out_p)
    cragen.generate_crag(gt, sps_folder, raws, membranes, create_conf, 
        max_zlink=maxHaus, histories=hists, histories_thresh=thresh, overwrite=True)
    cragen.extract_best_effort(effort_conf, best_effort=best_effort_path, overwrite=True)

