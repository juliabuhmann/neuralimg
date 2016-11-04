## Information

This repository contains utilities to deal with a neural reconstruction problem.

## Install requirements

It requires [conda](http://conda.pydata.org/miniconda.html) to be installed.

Requirements to install:

- Beforehand, the following utilities must be installed:
  - [Candidate_mc](https://github.com/DaniUPC/candidate_mc)
  - [TED](https://github.com/DaniUPC/ted)

We assume both projects have been installed in the HOME directory.
We need out PATH variable to point at the build/binaries folder within each of the previous folders.:

```
export PATH=$PATH:$HOME/candidate_mc/build/binaries:$HOME/ted/build/binaries/
```

And PYTHONPATH variable must point to the build/python folder:

```
export PYTHONPATH=$PYTHONPATH:$HOME/candidate_mc/build/python
```

Finally, to install this package:

```
git clone https://github.com/DaniUPC/neuralimg
```

Make sure the PYTHONPATH variable points to the cloned project as well.

The python libraries used to execute the main functionalities have been embedded into a Conda environment. To import it:

```
conda env create -f envs/default.yml
```

This will create an environment called 'neural'. To activate it, call:

```
source activate neural
```

And to disable it, call:

```
source deactivate
```

Finally, we need to install Tensorflow. It is only available for CPU in the conda repositories. So, if you need 
to execute it on GPU (which is probably the case), install it using Pip following the instructions [here](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation).

## Network reconstruction pipeline - Main use cases:

1. CRAG generation and best effort extraction can be performed using [this script](https://github.com/DaniUPC/neuralimg/blob/master/examples/segment_and_crag.py).
2. To extract datasets from a CRAG solution we can use [this example](https://github.com/DaniUPC/neuralimg/blob/master/examples/create_dataset.py). Then, resulting data can be visualized using [this notebook](https://github.com/DaniUPC/neuralimg/blob/master/notebooks/dataset_visualization.ipynb)
3. To train a network to extract descriptors from it, we can use [this script](https://github.com/DaniUPC/neuralimg/blob/master/examples/launch_network.py).
4. Once we have a network trained properly, we can save the descriptors to the crag by using [code here](https://github.com/DaniUPC/neuralimg/blob/master/examples/test_features.py).

## Other use cases

In order to explore parametrizations for the CRAG creation, we can use the following scripts:

- Important: groundtruth sometimes can be preprocessed for better evaluation. We have seen that some samples have small regions that represent nothing but noise. Moreover, unconnected regions belonging to the same neuron can affect the performance of the evaluation in the supervoxel computation from steps 1 to 3 (when we just consider pairwise relations between sections). A version of the groundtruth solving this issues can be obtained [here](https://github.com/DaniUPC/neuralimg/blob/master/examples/unconn_and_remove_holes.py).

1. [Evaluation of superpixel segmentations](https://github.com/DaniUPC/neuralimg/blob/master/examples/evaluate_grid.py). Results can be explored in this [notebook](https://github.com/DaniUPC/neuralimg/blob/master/notebooks/grid_analysis.ipynb).
2. Given some desired parametrization, we can [obtain our superpixels and merge trees](https://github.com/DaniUPC/neuralimg/blob/master/examples/segment_data.py).
3. Given segmentations, we can evaluate [up to which threshold we must merge adjacent candidates](https://github.com/DaniUPC/neuralimg/blob/master/examples/evaluate_merges.py).
- In order to create CRAGS, we can use examples [here](https://github.com/DaniUPC/neuralimg/blob/master/examples/crag_gen_examples.py).


