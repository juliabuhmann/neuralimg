#!/usr/bin/python

from neuralimg.evaluation import speval as ev
import os

""" Evaluates CRAGs on sample data using different maximum Haussdorff distances
(merge history to be created using extract_merge_trees) """

gts = 'data/crag/gt'
sps = 'data/crag/sps'
raws = 'data/crag/raw'
histories_mc = os.path.join(sps, 'histories_mc')
histories_gala = os.path.join(sps, 'histories_gala')
membranes = 'data/crag/mem_norm'

create_conf = 'data/crag/config/create_training_project.conf'
features_conf = 'data/crag/config/extract_training_features.conf'
effort_conf = 'data/crag/config/extract_best-effort.conf'

dists = [150, 250, 350]

# Computed best thresh for example data in MC: 0.1652
ev.evaluate_crags(sps, gts, raws, membranes, histories_mc, create_conf, features_conf, 
    effort_conf, dists, nworkers=3, thresh=0.1652, outp='stats_mc_test.dat')

# Computed best thresh for example data in Gala: 0.3267
ev.evaluate_crags(sps, gts, raws, membranes, histories_gala, create_conf, features_conf, 
    effort_conf, dists, nworkers=3, thresh=0.3267, outp='stats_gala_test.dat')
