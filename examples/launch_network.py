# ! /usr/bin/python

from neuralimg.training import siamese as si
import os
import sys
from ml_utils import machine_learning_utils as ml


""" Launches a Triplet Siamese network using the given data and stores the model and the logs of the network
so they can be further visualized. More precisely, it stores:

    - 'models' folder contains the following information:
        -> Checkpoint information so most recent model can be retrieved in case of failure.
        -> loss.jpg as the plot with training and validation evolution
        -> metadata.dat as information related to the data used to train the network
        -> Configuration of the training
        -> When it finished, stores the final model and the list of losses for bot training and validation

    - 'logs' folder contains:
        -> Contains a file that gathers the summary and shows the evolution of the network state over time.

            This can be visualized thanks to the Tensorboard visualization tool like this:
                $ tensorboard --logdir=path/to/logs

            Then, we can access the visualization tool by browsing the following URL in a browser:
                http://localhost:6006/

            In case the tensorboard is executed remotely, we can access the GUI by typing the following:

                ssh -NL localhost:6007:localhost:6006 user@server
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# dataset = '/raid/julia/projects/fragment_embedding/dani_crag/dataA.h5'
# dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets/sampleA.h5'
# dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/sampleA.h5'
# dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/sampleA_backup.h5'
dataset = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/triplets_toy/dummy2.h5'
models = '/raid/julia/projects/fragment_embedding/nn_training/models'


# logs = 'logs'
# logs = '/raid/julia/projects/fragment_embedding/nn_training/logs'
base_dir_session = '/raid/julia/projects/fragment_embedding/nn_training/'
session_dir, tensorboard_dir = ml.folder_structure_for_tensorflow(base_dir_session)

os.mkdir(tensorboard_dir + '/test')
os.mkdir(tensorboard_dir + '/train')

print(tensorboard_dir)
# crag_path = '/raid/julia/projects/fragment_embedding/vanessa_crag/cremi_besteffort_van/sampleA_woFeatures.hdf'
crag_path = ''
crag_path = None
retrain = False

if retrain:
    tensorboard_dir = '/raid/julia/projects/fragment_embedding/nn_training/toy_model/'


siamese = si.TripletSiamese(dataset, crag_path, traindataname='train', testdataname='test')
conf_path = '/raid/julia/projects/fragment_embedding/nn_training/configs/config.conf'

#conf_path = None
siamese.train(tensorboard_dir, tensorboard_dir, conf_path, retrain=retrain)

