# !/usr/bin/python

import numpy as np
from neuralimg.training import ml

""" TSNE visualization: maps high dimensional data into low dimensional so it can
be easily plotted """

# Generate random images and descriptors and visualize them
imgs = np.random.random((30, 32, 32, 3))
desc = np.random.random((30, 200))
# imgs should correspond to input images and descriptors to network outputs
plot_num = 25
ml.tsne_visualization(imgs, desc, plot_num)

