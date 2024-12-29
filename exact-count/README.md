
# RoMA - Exact Count

The following instructions are for installing and executing the Exact Count alogithm on a linux/ubuntu machine with a GPU properly configured.

This tutorial assumes you have the CUDA driver 11.7 installed.

### Setting up the environment

1. Create a virtual environment using conda:

`conda create -n exact-count python=3.10`

2. Activate the environment:

`conda activate exact-count`

5. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

#### Exact-count on simple models
`python ver_basic_models_new_algorithm.py`

#### Exact-count on acas-xu models
`python ver_acas_new_algorithm.py`

Good Luck!
