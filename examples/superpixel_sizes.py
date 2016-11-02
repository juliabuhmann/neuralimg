# !/usr/bin/python

from neuralimg.image import preproc as pr
import matplotlib.pyplot as plt

""" Generates superpixels given the membrane probabilities """

proc = pr.DatasetProc('/DataDisk/morad/cremi/superpixels/sampleA/mask3_sigma1_ted25')
proc.read()
sizes, mean, std = proc.compute_supervoxel_stats()

print('Mean: %f, Standard deviation: %f' % (mean, std))

plt.hist(sizes, 50, normed=1, histtype='stepfilled')
plt.show()

