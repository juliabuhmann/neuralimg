
from scipy import signal
import h5py
import matplotlib.pyplot as plt
import numpy as np

dataset = '/Users/juliabuhmann/Dropbox/Doktorarbeit/2016/NIPS/data/heur25triplet3.h5'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/heuristics25/triplets/sampleB.h5'

# Data: [crag_id, anchor/pos/neg, mask/grey/probmap, 128, 128]
example = 0732
trip_type1 = 0
trip_type2 = 1
channel = 1

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

def normalize_image(image):
    image = image - np.mean(image)
    image = image/np.std(image)
    #print np.min(image), np.max(image)
    return image


def calculate_cross(dataset, channel, distance_outputfile, normalize=True):
    cross_cor_pos_col = []
    cross_cor_neg_col = []
    crossmax_pos = []
    crossmax_neg = []
    f = h5py.File(dataset, 'r')
    num_of_exam = f['data'].shape[0]
    num_of_exam = 100

    cor_dimx = f['data'].shape[-2]
    cor_dimy = f['data'].shape[-1]

    for example in range(0, num_of_exam):
        anchor = f['data'][example, 0, channel]
        pos = f['data'][example, 1, channel]
        neg = f['data'][example, 2, channel]
        if normalize:
            anchor = normalize_image(anchor)
            pos = normalize_image(pos)
            neg = normalize_image(neg)

        # pos_cross = signal.correlate2d(anchor, pos, mode='valid')
        pos_cross = signal.correlate2d(anchor, pos)
        neg_cross = signal.correlate2d(anchor, neg)

        # print pos_cross.shape, 'shape'
        print pos_cross[cor_dimx, cor_dimy], neg_cross[cor_dimx, cor_dimy]
        cross_cor_pos_col.append(pos_cross[cor_dimx, cor_dimy])
        cross_cor_neg_col.append(neg_cross[cor_dimx, cor_dimy])
        crossmax_neg.append(np.max(neg_cross))
        crossmax_pos.append(np.max(pos_cross))

    # f = h5py.File(distance_outputfile, )

    cross_cor_pos_col = np.array(cross_cor_pos_col)
    cross_cor_neg_col = np.array(cross_cor_neg_col)

    crossmax_neg = np.array(crossmax_neg)
    crossmax_pos = np.array(crossmax_pos)

    print cross_cor_pos_col, cross_cor_neg_col
    print 'percentage %0.5f' %(np.sum(cross_cor_pos_col > cross_cor_neg_col)/float(num_of_exam))
    print 'percentage of max %0.5f' %(np.sum(crossmax_pos > crossmax_neg)/float(num_of_exam))





def plot_a_pair(example, trip_type1=0, trip_type2=1, channel=1, normalize=True):
    f = h5py.File(dataset, 'r')
    image1 = f['data'][example, trip_type1, channel]
    image2 = f['data'][example, trip_type2, channel]
    f.close()

    if normalize:
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        print np.min(image1), np.max(image2)

    correlation = signal.correlate2d(image1, image2)
    # correlation = image1 - image2
    cor_dimx = correlation.shape[0]//2
    cor_dimy = correlation.shape[1]//2
    f, axarr = plt.subplots(3)

    plt.axis('off')


    axarr[0].imshow(image1, cmap='gray')
    axarr[1].imshow(image2, cmap='gray')
    axarr[2].imshow(correlation, cmap='gray')
    for ax in list(axarr.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])

    axarr[2].set_title('max correlation: %0.3f \n correlation of middle: %0.3f' %(np.max(correlation), correlation[cor_dimx, cor_dimy]))
    # axarr[2, 0].set_title('Negative Example Dist %0.4f' %neg_dist)
    plt.tight_layout()
    plt.show()

calculate_cross(dataset, 0, 'dummy', normalize=True)
# plot_a_pair(1, channel=0)