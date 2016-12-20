import h5py
import numpy as np

def print_distance_statistics(dataset, distanceset):
    f_triplets = h5py.File(dataset, 'r')
    ref_ids = list(f_triplets['ref_ids'].value)
    ref_ids = [int(ref_id) for ref_id in ref_ids]
    f_triplets.close()

    f = h5py.File(distanceset, 'r')
    # This section validates whether for each slice node the right partner would have been picked.
    picked_collection = []
    for slice_node in np.unique(ref_ids):

        ids = list(np.where(np.array(ref_ids) == slice_node)[0])
        was_picked = True

        for id in ids:
            pos_dist = f['positive_dist'][id]
            neg_dist = f['negative_dist'][id]
            if neg_dist < pos_dist:
                was_picked = False
        picked_collection.append(was_picked)

    pos_dist_all = f['positive_dist'].value
    neg_dist_all = f['negative_dist'].value
    f.close()
    print('perc of correct', np.sum(picked_collection)/float(len(picked_collection)))
    print('# correct / #all:  %d /%d' %(np.sum(picked_collection), len(picked_collection)))
    print('positive distance mean %0.3f std %0.3f' %(np.mean(pos_dist_all), np.std(pos_dist_all)))
    print('negative distance mean %0.3f std %0.3f' %(np.mean(neg_dist_all), np.std(neg_dist_all)))
    print('mean margin %0.3f std %0.3f' %(np.mean(neg_dist_all-pos_dist_all), np.std(neg_dist_all-pos_dist_all)))
    print('whole percentage', np.sum(pos_dist_all < neg_dist_all)/float(pos_dist_all.shape[0]))


dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/triplets_rank3/sampleA.h5'
distanceset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank1models/smallMLPmodel/distances_sampleA.h5'


# print_distance_statistics(dataset, distanceset)

distanceset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/distances/rank3models/sampleArank3model_II/distances_sampleA.h5'
print_distance_statistics(dataset, distanceset)

