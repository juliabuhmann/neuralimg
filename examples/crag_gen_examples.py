#!/usr/bin/python

from neuralimg.crag import crag as cr

""" Crag generation examples from the sample data """

if __name__ == '__main__':

    gt = 'data/crag/gt/'
    sps = 'data/crag/sps/'
    raws = 'data/crag/raw'
    hists = 'data/crag/sps/histories_mc'
    thresh = 0.06   # Set computed thresh
    maxHaus = 800   # Maximum Hausdorff distance
    membranes = 'data/crag/mem'
    out_p = 'out/project'
    create_conf = 'data/crag/config/create_training_project.conf'
    features_conf = 'data/crag/config/extract_training_features.conf'
    effort_conf = 'data/crag/config/extract_best-effort.conf'

    # Generate crag, extract features and best effort
    cragen = cr.CragGenerator(out_p)
    cragen.generate_crag(gt, sps, raws, membranes, create_conf, max_zlink=maxHaus, 
        histories=hists,histories_thresh=thresh, overwrite=True)
    cragen.extract_best_effort(effort_conf, best_effort='out/project/best', overwrite=True)

