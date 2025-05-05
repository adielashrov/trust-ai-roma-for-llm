# Measuring LLM Typo Robustness with RoMA

The following instructions are for installing and executing the roma-llm typo robustness case study on a Linux/Ubuntu machine with a properly configured GPU.

### Setting Up the Environment

1. Create a virtual environment using conda:

`conda create -n roma-llm python=3.10`

2. Activate the environment:

`conda activate roma-llm`

3. Install the requirements file

`pip install -r requirements.txt`

### Executing the experiment

#### 1. Creating the Dataset

`python create_full_sentence_dataset.py`

This should create the `full_modified_sentences` folder with the perturbed sentences. 

#### 2. Running Predictions on the Dataset

**Recommendation**: Execute the prediction code using a GPU.
##### Step 2.1 - Download a Model into the Project Directory

 1. Create a folder named `models` inside the project directory.
 2. Download a model folder (best_model or final_model) from the [models_folder](https://drive.google.com/drive/folders/1HijpyTd5HhTYV1qqxHhrXdZaw6JJ_k1N?usp=drive_link) to the created `models` directory.
 3. The selected model for execution is configured in line #10 in the file `predict_sentence_dataset_full.py`.

##### Step 2.2 - Run the Predicetion Code

`python predict_sentence_dataset_full.py`

*Note:* You should see a log file created, in which the status of the currently processed file is documented.

The output of this step is a folder named `full_sentiment_predictions`.

#### 3. Running RoMA Analysis on the Predictions

Execute the following command to run the RoMA analysis:

`python analysis_full_prediction.py`

The outputs of this step are  the `total_analysis*.csv` and `analysis_timing*.log` files.

Good Luck!
