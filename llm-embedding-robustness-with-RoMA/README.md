

# Measuring LLM Embedding Robustness with RoMA

The following instructions are for installing and executing the roma-llm case study on a linux/ubuntu machine with a GPU properly configured.

### Setting up the environment
1. Create a virtual environment using conda:

`conda create -n roma-llm python=3.10`

2. Activate the environment:

`conda activate roma-llm`

3. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

**Pre-requisite** - you should download the GLUE dataset to a folder named data.

#### 1. Creating the dataset

`python create_full_sentence_dataset.py`

#### 2. Running prediction on the dataset

`python predict_sentence_dataset_full.py`

#### 3. Running RoMA analysis on the predictions

`python analysis_full_prediction.py`

Good Luck!
