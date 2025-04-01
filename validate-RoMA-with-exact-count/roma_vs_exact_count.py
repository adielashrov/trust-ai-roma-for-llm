import tensorflow as tf
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import datetime
import logging
from tensorflow.keras.models import load_model
import timeit

# Set random seed for reproducibility
np.random.seed(0)

# Model constants for basic models
# model_name = "model_2_20.h5"
# model_name_for_log = model_name[:-3]
# str_model_tf = f"models/{model_name}"

# Model constants for Acas models
model_name = "ACASXU_run2a_2_7_batch_2000.h5"
model_name_for_log = model_name[:-3]
str_model_tf = f"acas_models_h5/{model_name}"


def setup_logging():
    current_time = datetime.datetime.now().strftime('%d_%m_%H_%M_%S')
    logfile_name = f"logfile_{model_name_for_log}_{current_time}.log"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(logfile_name),
        logging.StreamHandler()
    ])


def save_results_to_file(robustness_stats):
    current_time = datetime.datetime.now().strftime('%d_%m_%H_%M_%S')
    filename = f'results/stats_{model_name_for_log}_{current_time}.csv'
    df = pd.DataFrame(robustness_stats,
                      columns=['iteration', 'robustness'])
    df.to_csv(filename, index=False)
    logging.info(f'DataFrame saved to {filename}')


def execute_roma_on_simple_model():
    model = tf.keras.models.load_model(str_model_tf, compile=False)
    statistics_at_work = []

    sample_size = 10000
    sum_of_accurate_predictions = 0

    # log the model name (without the models/ prefix):
    logging.info(f"Model name: {str_model_tf[7:]}")

    for i in range(sample_size):
        # Draw a random point from the input range
        random_point = np.random.uniform(low=0, high=1, size=2)  # for every model the size should be different
        # Calculate model output for the random point
        model_output = model.predict(np.array([random_point]))
        if model_output.item() > 0:
            sum_of_accurate_predictions += 1

        current_robustness = sum_of_accurate_predictions / (i + 1)
        logging.info(f"Current iteration: {i}")
        logging.info(f"current robustness: {current_robustness}")
        statistics_at_work.append([i, current_robustness])

    logging.info(f"Sum of accurate predictions: {sum_of_accurate_predictions}")
    logging.info(f"Robustness: {sum_of_accurate_predictions / sample_size}")

    # write the statistics  to a csv file, one csv file per label
    save_results_to_file(statistics_at_work)

    return


def sanity_check(model):
    # Reshape the input array to match the expected shape (None, 5)
    src_point = np.array([0.6, -0.25, 0.11122729905459967, 0.48432132038710424, -0.45])
    src_point = src_point.reshape(1, -1)  # Reshape to (1, 5)

    model_output = model.predict(src_point)
    max_index = np.argmax(model_output)
    if max_index != 0:  # if the index of the maximum value is not equal to 0
        logging.info(f"Property holds for src_point {src_point}")


def execute_roma_on_acas_model():
    model = tf.keras.models.load_model(str_model_tf, compile=False)
    statistics_at_work = []
    property = {
        "type": "decision",
        "P": [
            [0.600000, 0.679858],
            [-0.500000, 0.500000],
            [-0.500000, 0.500000],
            [0.450000, 0.500000],
            [-0.500000, -0.450000]
        ],
        "A": 0
    }

    sample_size = 1000
    sum_of_accurate_predictions = 0
    range_array = np.array(property["P"])

    # log the model name (without the models/ prefix):
    logging.info(f"Model name: {str_model_tf[11:]}")

    for i in range(0, sample_size):
        # Draw a random point from the input range
        random_point = np.random.uniform(range_array[:, 0], range_array[:, 1], range_array.shape[0])

        # Calculate model output for the random point
        model_output = model.predict(np.array([random_point]))

        # Find the index of the maximum value in the model output
        max_index = np.argmax(model_output)
        if max_index != property["A"]:  # if the index of the maximum value is not equal to 0
            sum_of_accurate_predictions += 1

        current_robustness = sum_of_accurate_predictions / (i + 1)
        logging.info(f"Current iteration: {i}")
        logging.info(f"current robustness: {current_robustness}")
        statistics_at_work.append([i, current_robustness])

    logging.info(f"Sum of accurate predictions: {sum_of_accurate_predictions}")
    logging.info(f"Robustness: {sum_of_accurate_predictions / sample_size}")

    # write the statistics  to a csv file, one csv file per label
    save_results_to_file(statistics_at_work)

    return


if __name__ == "__main__":

    roma_mode = 'acas_model'  # or 'simple_model'
    setup_logging()

    if roma_mode == 'simple_model':
        execution_time_simple_model = timeit.timeit(execute_roma_on_simple_model, globals=globals(), number=1)
        logging.info(f"Execution time for execute_roma_on_simple_model: {execution_time_simple_model}")
    if roma_mode == 'acas_model':
        execution_time_acas_model = timeit.timeit(execute_roma_on_acas_model, globals=globals(), number=1)
        logging.info(f"Execution time for execute_roma_on_acas_model: {execution_time_acas_model}")
