# Script to merge several triplets files into one
import h5py
import numpy as np
import random



MERGEDATA = True
SHUFFLE = True

dummy_file = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/dummy2.h5'
if MERGEDATA:
    validation_percentage = 0.8
    # Open the dummy file to which the data is written out

    new_f = h5py.File(dummy_file, 'a')
    # del f['data']
    # f.create_dataset('data', compression='gzip')
    # new_f .close()
    del new_f['train']
    del new_f['test']
    new_f.create_dataset('train', shape=(0, 3, 3, 128, 128), maxshape=(None, 3, 3, 128, 128), compression='gzip', chunks=True)
    new_f.create_dataset('test', shape=(0, 3, 3, 128, 128), maxshape=(None, 3, 3, 128, 128), compression='gzip', chunks=True)

    # files = ['/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank7/sampleA.h5',
    #          '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank7/sampleB.h5',
    #          '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank7/sampleC.h5']
    files = ['/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank5/sampleA.h5']
    # files = ['/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets_rank3/sampleA.h5']
    # max_shape=None
    # self.h5_file.create_dataset(DATA_TAG, data_shape, compression='gzip', chunks=True, maxshape=max_shape)
    # current = self.h5_file[DATA_TAG].shape[0]
    # self.h5_file[DATA_TAG].resize(current + num, axis=0)
    # data_shape = (0, self.get_sections(), self.dims, self.conf.height, self.conf.width)
    # max_shape = (None, self.get_sections(), self.dims, self.conf.height, self.conf.width)

    data_collection = []
    for file in files:
        f = h5py.File(file, 'r')
        ref_ids = f['ref_ids'].value
        # data = f['data'].value

        validation_cut = int(len(ref_ids)*validation_percentage)
        test_size = len(ref_ids)-validation_cut
        current = new_f['train'].shape[0]
        print('size of train', validation_cut)
        print('size of test', test_size)
        new_f['train'].resize(current + validation_cut, axis = 0)
        new_f['train'][current:current+validation_cut, ...] = f['data'][:validation_cut, ...]

        current = new_f['test'].shape[0]
        new_f['test'].resize(current + test_size, axis = 0)
        new_f['test'][current:current+test_size, ...] = f['data'][validation_cut:, ...]
        f.close()

    print('final train shape', new_f['train'].shape[0])
    print('final train shape', new_f['test'].shape[0])
    new_f.close()



chunksize = 250
dataset_name = 'train'
# Shuffle data


def shuffle_data(new_f, dataset_name, starts=None):

    start1 = starts[0]
    start2 = starts[1]
    print(start1, start2)
    first_array = new_f[dataset_name][start1:start1 + chunksize].copy()
    sec_array = new_f[dataset_name][start2:start2 + chunksize].copy()

    np.random.shuffle(first_array)
    np.random.shuffle(sec_array)
    new_f[dataset_name][start1:start1 + chunksize, ...] = sec_array.copy()
    new_f[dataset_name][start2:start2 + chunksize, ...] = first_array.copy()


def shuffle_dataset_and_write(dataset_name, chunksize=300, number_of_iterations=2):
    data_shape = new_f[dataset_name].shape[0]

    # Make sure that every part is shuffled at least twice
    shuffle_list = range(0, data_shape - chunksize, chunksize - 15)
    shuffle_list.append(data_shape - chunksize)

    shuffle_list_com = []
    for num_it in range(number_of_iterations):
        shuffle_list_com.extend(shuffle_list)
    print(shuffle_list_com)
    random.shuffle(shuffle_list_com)

    # shuffle_data(new_f, dataset_name, data_shape, chunksize, starts=(0,))

    for ii, start1 in enumerate(shuffle_list):
        if ii% 10 ==0:
            print('%i of %i' %(ii, len(shuffle_list)))
        start2 = random.sample(shuffle_list, 1)[0]
        shuffle_data(new_f, dataset_name, starts=(start1, start2))


if SHUFFLE:
    new_f = h5py.File(dummy_file, 'a')

    shuffle_dataset_and_write('train', number_of_iterations=2)
    shuffle_dataset_and_write('test', number_of_iterations=2)
    new_f.close()






    # data_collection.append(data)
    # print(data.shape)


# data_collection = np.concatenate(data_collection, axis=0)
# print(data_collection.shape)
# np.random.shuffle(data_collection)
# outputfile = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics100/triplets/merged.h5'

# f = h5py.File(outputfile, 'w')
# f.create_dataset('data', data=new_array)
# f.close()

# dummy_file = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/sampleA_backup.h5'



# labels_type = h5py.special_dtype(vlen=str)
# labels_data = np.asarray(list(['binary', 'intensity', 'raw']), dtype=object)
# f.create_dataset('clabels', data=labels_data, dtype=labels_type)
# f.create_dataset('cpositions', data=np.asarray(list(self.channel_map.values())))

