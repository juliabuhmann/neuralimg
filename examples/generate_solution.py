#!/usr/bin/python

from neuralimg.crag import crag as cr

""" Benchmark that uses different merge and end scores and returns the Adapted Rand and VOI
of the best effort of each setting """

crag_path = '/DataDisk/morad/out/hdf/training_dataset.h5'
iterations = 5
ends = 1
merges = 1
folder='solution'
threads=5
logs='.'
model = 'models'

cr.generate_solution(crag_path, merges, ends, iterations, model, folder, 
        logs=logs, nthreads=threads)

