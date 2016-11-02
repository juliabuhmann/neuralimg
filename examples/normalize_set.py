# ! /usr/bin/python

from neuralimg.image import preproc as pr

""" Normalizes the membrane probabilities so they are in [0,1] interval """

proc = pr.DatasetProc('data/crag/mem')
proc.read()
proc.normalize()
proc.save_data('data/crag/mem_norm')

