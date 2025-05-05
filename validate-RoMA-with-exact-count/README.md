# RoMA vs. Exact Count

The following instructions are for installing and executing the RoMA vs. Exact Count algorithms on a linux/ubuntu machine.

### Setting up the environment

1. Create a virtual environment using conda:

`conda create -n roma-vs-exact-count python=3.10`

2. Activate the environment:

`conda activate roma-vs-exact-count`

3. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

#### Exact-Count on simple models

 1. First you need to choose the model you wish to execute exact_count on.
 2. For example, if you want to execute exact count on the model `model_2_20.onnx`, you need to set the following vairables accordingly (this basic model has 2 input variables):
 3. `NETWORK_FILENAME = "onnx_models/model_2_20.onnx"`
 4. `NUM_OF_INTERVALS = 2 # Should match the number in the input_file name` 
 5. Then you can execute: 
 6. `python ver_basic_models_new_algorithm.py`
 7. The results should appear in a `verification_model_2_20_*.log` file.

*Note that you can control the epsilon and depth parameters at the `parameter_sweep` method.* 

#### RoMA on simple models

In this example, we continue with the previous `model_2_20` model.

 1. Set the model path at the begining of roma_vs_exact_count.py file (uncomment lines 14-16).
 2. for instance, `model_name = "model_2_20.h5"`(note that the ending for the models here is `.h5`).
 3. Set the `roma_mode = 'simple_model'` on line 136.
 4. Execute the RoMA algorithm:
 5. `python roma_vs_exact_count.py`
 6. The results should appear in the `logfile_model_2_20_*.log` and at the `stats_model_2_20_*.csv` files.

*Note that you can control the number of sampled points at the `sample_size` variable.* 

#### Exact-Count on acas-xu models

 1. First you need to choose the model you wish to execute exact_count on.
 2. For example, if you want to execute exact count on the model `ACASXU_experimental_v2a_2_7.onnx`, you need to set the following vairables accordingly:
 3. `NETWORK_FILENAME = "acasxu/ACASXU_experimental_v2a_2_7.onnx"`
 5. Then you can execute: 
 6. `python ver_acas_new_algorithm.py`
 7. The results should appear in a `verification_ACASXU_experimental_v2a_2_7_*.log` file.

*Note that you can control the epsilon and depth parameters at the `parameter_sweep` method.* 

#### RoMA on acas-xu models

In this example, we continue with the previous `ACASXU_experimental_v2a_2_7` model.

 1. Set the model path at the begining of roma_vs_exact_count.py file (uncomment lines 19-21).
 2. `model_name = "ACASXU_run2a_2_7_batch_2000.h5"`(note that the ending for the models here is `.h5`).
 3. Set the `roma_mode = 'acas_model'` on line 136.
 4. Execute the RoMA algorithm:
 5. `python roma_vs_exact_count.py`
 6. The results should appear in the `logfile_ACASXU_run2a_2_7_batch_2000_*.log` and at the `stats_ACASXU_run2a_2_7_batch_2000_*.csv` files.

*Note that you can control the number of sampled points at the `sample_size` variable.* 

Good Luck!
