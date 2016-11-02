#!/usr/bin/python

from neuralimg.evaluation import speval as ev

""" Benchmark that uses different merge and end scores and returns the Adapted Rand and VOI
of the best effort of each setting """

crag_path = 'out/project/hdf/training_dataset.h5'
effort_conf = 'data/crag/config/extract_best-effort.conf'

ev.evaluate_crag_weights(crag_path, effort_conf, [0.5], [0.5], 'weight_stats.dat')
