#!/usr/bin/python

from neuralimg.crag import crag as cr

""" Crag generation examples from the sample data """

if __name__ == '__main__':

    gt = 'data/crag/gt/'
    sps = 'data/crag/sps/'
    raws = 'data/crag/raw'
    membranes = 'data/crag/mem'
    out_p = 'out/project'
    create_conf = 'data/crag/config/create_training_project.conf'
    features_conf = 'data/crag/config/extract_training_features.conf'
    effort_conf = 'data/crag/config/extract_best-effort.conf'

    # Generate crag, extract features and best effort
    cragen = cr.CragGenerator(out_p)
    cragen.generate_crag(gt, sps, raws, membranes, create_conf, 200, histories=None, overwrite=True)
    cragen.extract_features(features_conf)
    cragen.extract_best_effort(effort_conf, cr.LossType.HEURISTIC, 'out/project/best', overwrite=True)
    stats = cragen.evaluate_best_effort( 50)
    print(stats)

    print('Test 1 finished!')

    # Read crag and extract best effort
    cragen = cr.CragGenerator(out_p)
    cragen.read_crag('out/project/hdf/training_dataset.hdf')
    cragen.extract_features(features_conf)
    cragen.extract_best_effort(effort_conf, cr.LossType.HEURISTIC, 'out/project/best', overwrite=True)
    stats = cragen.evaluate_best_effort(50)
    print(stats)

    print('Test 2 finished!')

    cragen = cr.CragGenerator(out_p)
    cragen.read_crag('out/project/hdf/training_dataset.hdf')
    stats = cragen.evaluate_best_effort(50, best='out/project/best')
    print(stats)
    print('Test 3 finished!')

