
# RoMA - Exact Count

The following instructions are for installing and executing the Exact Count vs. RoMA algorithms on a linux/ubuntu machine.

### Setting up the environment

1. Create a virtual environment using conda:

`conda create -n exact-count python=3.10`

2. Activate the environment:

`conda activate exact-count`

3. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

#### Exact-count on simple models
`python ver_basic_models_new_algorithm.py`

#### Exact-count on acas-xu models
`python ver_acas_new_algorithm.py`

#### RoMA on simple models

Set the model name and relevant folder at the begining of roma_vs_exact_count.py file (lines 13-21), and then execute.

Example - basic models:

`model_name = "model_2_20.h5"`

`model_name_for_log = model_name[:-3]`  

`str_model_tf = f"models/{model_name}"`

Example - AcasXU models:

`model_name = "ACASXU_run2a_2_7_batch_2000.h5"`  

``model_name_for_log = model_name[:-3]``

``str_model_tf = f"acas_models_h5/{model_name}"``

 - Run the roma algorithm 

`python roma_vs_exact_count.py`

### Experiments results

The experiments results can be found in the following [link](https://drive.google.com/drive/folders/1KOEkhwcs-tjPOB1uDQnLD_4iQKPC62nJ?usp=drive_link).

Good Luck!
