#!/usr/bin/python

from neuralimg.crag import crag as cr

""" Benchmark that uses different merge and end scores and returns the Adapted Rand and VOI
of the best effort of each setting """

crag_path = '$HOME/data/crags/sampleA/hdf/training_dataset.h5'
iterations = 5
ends = 1
merges = 1
folder='solution'
model = 'path/to/model'

cr.generate_solution(crag_path, merges, ends, iterations, folder, model)

