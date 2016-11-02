#!/usr/bin/python

import numpy as np
import os
import h5py
import tempfile
import sys

from neuralimg.crag.merge_trees import *
from neuralimg import dataio

req_version = (3,5)
cur_version = sys.version_info

if cur_version < req_version:
    print('WARNING: Gala tree extraction is disabled for Python versions below 3.5')
else:
    from gala import imio, classify, features, agglo, evaluate as ev


####################################################################################
#
# This file presents a class for extracting merge trees: GalaTreeExtractor. It 
# uses already built projects that are specified in their headers. Note that 
# Gala is muh slower than MergeTreeExtractor due to overhead and basically due 
# to the learning step
# 
# Merge history entries have the form:
#    region_old_1   region_old_2    new_region  score
#####################################################################################


class GalaTreeExtractor(MergeTreeExtractor):

    # Gala requires input data to be normalized
    # Output score is always between 0 and 1

    def __init__(self, sp, mem, gt, model=None, min_epochs=5):
        """ If path for model provided, it will be stored. If model exists, it will be loaded
        and therefore we avoid computing it from scratch """
        MergeTreeExtractor.__init__(self, sp, mem, gt=gt)
        self.min_ep = min_epochs
        self.model_p = model

    def process_h5(self, inp):
        return inp

    def process_folder(self, inp):
        print('Processing ' + inp)
        files = dataio.FileReader(inp).read()
        out = tempfile.mkstemp(dir=self.tmp, suffix='.h5')[1]
        print('Storing into ' + out)
        with h5py.File(out) as f:
            f.create_dataset('stack', data=files, compression='gzip')
        return out

    def extract_trees(self, outp):

        dataio.check_volume(self.sp)
        dataio.check_volume(self.mem)

        # Train model before merging
        self.train_model()

        # Read stack
        with h5py.File(self.sp) as s, h5py.File(self.mem) as m:
            try:
                sp_stack = s['stack']
                mem_stack = m['stack']
            except KeyError as e:
                raise KeyError('Datasets must have a stack of images in key "stack"')

            # Iteratively create trees
            digits = get_digits(sp_stack.shape[0])
            for i in range(sp_stack.shape[0]):
                output_file = os.path.join(outp, str(i).zfill(digits) + ".txt")
                history = self._extract_specific(sp_stack[i, ...], mem_stack[i, ...], output_file)
                print('Created merge history {}'.format(history))

    def train_model(self):

        # Check probabilities are normalized
        if not is_normalized(self.mem):
            raise ValueError('Gala input probabilities must be normalized')

        if self.model_p is None:
            self._train_model()
        else:
            dataio.check_volume(self.model_p)
            self._train_model(self.model_p)

    def _train_model(self, model_file=None):

        print("Creating GALA feature manager...")
        fm = features.moments.Manager()
        fh = features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]) # Recommended numbers in the repo
        fg = features.graph.Manager()
        fc = features.contact.Manager()
        self.fm = features.base.Composite(children=[fm, fh, fg, fc])

        if model_file is not None and os.path.isfile(model_file):
            print('Loading model from path ...')
            rf = classify.load_classifier(model_file)
        else:

            gt, pr, sv = (map(imio.read_h5_stack, [self.gt, self.mem, self.sp]))

            print("Creating training RAG...")
            g_train = agglo.Rag(sv, pr, feature_manager=self.fm)

            print("Learning agglomeration...")
            (X, y, w, merges) = g_train.learn_agglomerate(gt, self.fm, learning_mode='permissive',
                min_num_epochs=self.min_ep)[0]
            y = y[:, 0]

            rf = classify.DefaultRandomForest().fit(X, y)

            # Save if path requested
            if model_file is not None:
                classify.save_classifier(rf, model_file)

        self.model = agglo.classifier_probability(self.fm, rf)

    def _extract_specific(self, sp, mem, path):
        """ Generates merge tree given a pair of matrices representing superpixels
        and membrane views of an image """
        def file_to_h5(img, suffix):
            tmp_file = os.path.join(self.tmp, suffix + '.h5')
            with h5py.File(tmp_file) as h5:
                h5.create_dataset('stack', data=np.expand_dims(img, 0), compression='gzip')
            return tmp_file

        # Map images into H5 sets
        iden = os.path.splitext(os.path.basename(path))[0]
        sp_path = file_to_h5(sp, iden + '_superpixels.h5')
        mem_path = file_to_h5(mem, iden + '_membranes.h5')
        sv, pr = (map(imio.read_h5_stack, [sp_path, mem_path]))

        # Create test rag given pair of images
        g_test = agglo.Rag(sv, pr, feature_manager=self.fm, merge_priority_function=self.model)

        # We use a threshold value > 1.0 (maximum) to retrieve the complete 
        # merge tree
        history, _, _ = g_test.agglomerate(5.0, save_history=True)

        # Gala internally relabels the regions. Use inverse mapping to get to 
        # original labels. If mapping equals original, no need to relabel (gets 
        # erros, indeed)
        if not g_test.is_sequential:
            mapping = g_test.inverse_watershed_map
            history = relabel_history(history, mapping)
        dump_history(history, path)

        # Clean auxiliar
        os.remove(sp_path)
        os.remove(mem_path)

        return path

