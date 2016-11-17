#!/usr/bin/python

from neuralimg.evaluation import speval as ev
import os

""" Evaluates CRAGs on sample data using different maximum Haussdorff distances
(merge history to be created using extract_merge_trees) """

gts = 'data/crag/gt'
sps = 'data/crag/sps'
raws = 'data/crag/raw'
histories_mc = os.path.join(sps, 'histories_mc')
thresh = 0.15  # example
membranes = 'data/crag/mem_norm'
tmp_path = '/DataDisk/morad/tmp'    # use a temporary path with enough space

dists = [350]

######### We can evaluate a single CRAG
stats = ev.evaluate_crag(sps=sps,
                    gts=gts, 
                    raws=raws, 
                    mems=membranes, 
                    hists=histories_mc, 
                    dist=dists[0], 
                    thresh=thresh,
                    max_merges=5,
                    res=[4,4,40],
                    threads=3,
                    indexes=None,   # Can use a set of indices here [init, end]
                    tmp=tmp_path)
print(stats)

######### Or we can evaluate CRAGS given a set of Hausdorff distances
otuput_file = 'stats_crag.dat'
stats = ev.evaluate_crags(sps=sps,
                    gts=gts, 
                    raws=raws, 
                    mems=membranes, 
                    hists=histories_mc, 
                    max_merges=5,
                    dists=dists, 
                    res=[4,4,40],
                    threads=3,
                    indexes=None,   # Can use a set of indices here [init, end]
                    nworkers=3,
                    thresh=thresh,
                    outp='stats_crag.dat',
                    tmp=tmp_path)
print(stats)

