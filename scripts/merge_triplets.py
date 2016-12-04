# Script to merge several triplets files into one
import h5py
import numpy as np


number_of_examples = 10
files = ['/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/sampleA.h5',
         '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/sampleB.h5',
         '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/sampleC.h5']
data_collection = []
for file in files:
    f = h5py.File(file, 'r')
    data = f['data'].value
    f.close()
    data_collection.append(data)
    print(data.shape)


data_collection = np.concatenate(data_collection, axis=0)
print(data_collection.shape)
np.random.shuffle(data_collection)
outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/merged.h5'

# f = h5py.File(outputfile, 'w')
# f.create_dataset('data', data=new_array)
# f.close()

dummy_file = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/sampleA_backup.h5'

f = h5py.File(dummy_file, 'a')
del f['data']
f.create_dataset('data', data=data_collection)
f.close()
# labels_type = h5py.special_dtype(vlen=str)
# labels_data = np.asarray(list(['binary', 'intensity', 'raw']), dtype=object)
# f.create_dataset('clabels', data=labels_data, dtype=labels_type)
# f.create_dataset('cpositions', data=np.asarray(list(self.channel_map.values())))

