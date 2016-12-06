#!/usr/bin/python

from neuralimg.crag import crag as cr
import os

""" Benchmark that uses different merge and end scores and returns the Adapted Rand and VOI
of the best effort of each setting """


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# crag_path = '/DataDisk/morad/out/hdf/training_dataset.h5'
# crag_path = '/raid/julia/documents/Dropbox/best_effort/heuristic_100sections/sampleA.hdf'
# crag_path = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/crag/sampleA_bestEffort.hdf'
crag_path = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/crag_best_effort/sampleA.hdf'
iterations = 5
ends = 1
merges = 1
# folder='/raid/julia/projects/fragment_embedding/nn_training/smallMLPmodel/solution/'
folder = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/crag_solution/'
threads=5
logs = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/crag_solution/'
model = '/raid/julia/projects/fragment_embedding/nn_training/sampleArank3model_II/221238/'

out = cr.generate_solution(crag_path, merges, ends, iterations, model, folder,
        logs=logs, nthreads=threads)

print(out)

