#!/usr/bin/python

from neuralimg.crag.crag_costs import *

ef = 1.0
ew = 1.0
nw = 1.0

crag_path = '/DataDisk/morad/out/project/hdf/training_dataset.h5'
costs = CragCostManager(crag_path)
costs.update_node_features('models')
costs.update_edge_features(ef)
costs.update_edge_weights(ew)
costs.update_node_weights(nw)
costs.save()
