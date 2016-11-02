# ! /usr/bin/python

from neuralimg.training import siamese as si
import os

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

dataset = '/DataDisk/morad/out/triplet.h5'
models = 'models'
logs = 'logs'
crag_path = '/DataDisk/morad/out/project/hdf/training_dataset.h5'

siamese = si.TripletSiamese(dataset, crag_path)
conf_path = '../training/config/network.conf'
#conf_path = None
siamese.train(models, logs, conf_path, retrain=True)

