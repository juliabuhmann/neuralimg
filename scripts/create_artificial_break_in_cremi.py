import h5py
import numpy as np


filename = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/crag_best_effort/sampleA_broken.hdf'
# filename = '/raid/julia/projects/fragment_embedding/cremi_data/sampleB_broken.hdf'


f = h5py.File(filename, 'a')
data =f['volumes/groundtruth'].value

max_value = np.max(data)
print(max_value, 'max value is')
data[:50, ...] += max_value
f['volumes/groundtruth'][...] = data

f.close()


