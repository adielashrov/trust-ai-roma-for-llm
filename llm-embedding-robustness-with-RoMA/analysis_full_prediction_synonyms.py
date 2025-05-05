import os
import csv
import time
from scipy import stats
from scipy.stats import anderson
from scipy.special import boxcox
from sklearn import preprocessing
import numpy as np
from scipy import stats
import math
import logging
from datetime import datetime

model_name = "final_model"  # Change this to the model name you are analyzing
mode = "full_sentiment_predictions_synonyms" # Change this to the mode you are analyzing_final_model
# Get the current timestamp
current_time = datetime.now().strftime('%d-%m_%H-%M-%S')
log_filename = f'statistical_analysis_{mode}_{model_name}_{current_time}.log'

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# number of samples
samples = 1821
threshold = 0.5

def extract_index(filename: str) -> float:
    return float(filename.split("_")[1].split(".")[0])

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        logging.error(f"Failed to cast {value} to float")
        return False


def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if f.endswith('.csv')])


def process_csv_files(directory):
    samples = count_files_in_directory(directory)
    results = []
    finale_table = np.zeros((samples, 12))

    counter = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            # Start the timer for this file
            start_time = time.time()

            try:
                with open(file_path, 'r') as file:
                    csv_reader = csv.reader(file)
                    rows = list(csv_reader)
                    logging.info(f"Processing file: {filename}")
                    if len(rows) < 2 or len(rows[0]) < 2:
                        logging.warning(f"Skipping {filename}: Insufficient data")
                        continue

                    col1 = [row[-2] for row in rows[1:]]  # Skip header row
                    col2 = [row[-1] for row in rows[1:]]  # Skip header row

                    start_row = 0
                    for i in range(len(rows)):
                        if is_float(col1[i]) and is_float(col2[i]):
                            start_row = i
                            break

                    if start_row == len(rows) - 1:
                        logging.warning(f"Skipping {filename}: No numeric data")
                        continue

                    if start_row != 0:
                        logging.warning(f"{filename}, the original prediction results are not in the first row")

                    true_label = None
                    # First, check if the negative probability is below the threshold,
                    # if so, the sentence is Positive, and we want to analyize the negative proablities
                    # Then, check if the positive probability is below the threshold,
                    # If so, the sentence is negative, and we want to analyze the positive probabilities
                    if float(col1[start_row]) < threshold:
                        process_col = np.array([float(x) for x in col1[start_row:]])
                        true_label = 1 # 1 for positive
                    elif float(col2[start_row]) <= threshold:
                        process_col = np.array([float(x) for x in col2[start_row:]])
                        true_label = 0 # 0 for negative
                    else:
                        logging.warning(f"Skipping {filename}: No values below threshold")
                        continue

                    # Check normality of current distribution
                    pred_anderson = anderson(process_col, dist='norm')
                    logging.info(f"Initial Anderson test statistic: {pred_anderson.statistic}")
                    logging.info(f"Critical values: {pred_anderson.critical_values}")

                    if pred_anderson.statistic < pred_anderson.critical_values[3]:  # 95% significance level
                        normalize_Z_score = (threshold - np.mean(process_col)) / np.std(process_col)
                        p_value = stats.norm.cdf(abs(normalize_Z_score))
                        statistic = pred_anderson.statistic
                        status = 1
                    else:
                        # Check if data needs transformation
                        if np.all(process_col == process_col[0]):  # All values are the same
                            normalize_Z_score = np.nan
                            p_value = np.nan
                            statistic = np.nan
                            status = 5
                        else:
                            # Prepare data for Box-Cox
                            min_val = np.min(process_col)
                            if min_val <= 0:
                                # Shift data to be positive
                                offset = abs(min_val) + 1
                                logging.info(f"Shifting data in {filename} to be positive by {offset}")
                                process_col_shifted = process_col + offset
                            else:
                                process_col_shifted = process_col

                            try:
                                shaped_cox_box, fitted_lambda = stats.boxcox(process_col_shifted)
                                coxbox_anderson = anderson(shaped_cox_box, dist='norm')
                                logging.info(f"Box-Cox Anderson test statistic: {coxbox_anderson.statistic}")

                                if coxbox_anderson.statistic < coxbox_anderson.critical_values[3]:
                                    if fitted_lambda == 0:
                                        Z_score = math.log(threshold)
                                        status = 4
                                    else:
                                        Z_score = ((threshold ** fitted_lambda) - 1) / fitted_lambda
                                        status = 2
                                        normalize_Z_score = (Z_score - np.mean(shaped_cox_box)) / np.std(shaped_cox_box)
                                        p_value = stats.norm.cdf(abs(normalize_Z_score))
                                        statistic = coxbox_anderson.statistic
                                else:
                                    normalize_Z_score = np.nan
                                    p_value = np.nan
                                    statistic = np.nan
                                    status = 3
                            except Exception as e:
                                logging.error(f"Error in Box-Cox transformation: {str(e)}")
                                normalize_Z_score = np.nan
                                p_value = np.nan
                                statistic = np.nan
                                status = 5

                    # Calculate process time in seconds
                    process_time = time.time() - start_time

                    # Update the storage array with one more column for processing time
                    finale_table[counter, 0] = extract_index(filename)
                    finale_table[counter, 1] = process_time  # Store the processing time
                    finale_table[counter, 2] = np.nan
                    finale_table[counter, 3] = true_label  # the category of the sentence
                    finale_table[counter, 4] = fitted_lambda if status in [2, 4] else np.nan
                    finale_table[counter, 5] = statistic
                    finale_table[counter, 8] = np.nan
                    finale_table[counter, 9] = status
                    finale_table[counter, 10] = p_value
                    finale_table[counter, 11] = normalize_Z_score
                    counter += 1

            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                continue

    # Save only the rows we actually processed
    finale_table = finale_table[:counter]
    os.makedirs("results", exist_ok=True)
    np.savetxt(f"results/final_report_{mode}_{model_name}_{current_time}.csv", finale_table, delimiter=",", fmt='%s',
               header="sentence_index,process_time,favorite_index,test_set_label,lambda,statistic,critical_val,"
                      "sig_level,nan,status,P_value,Z")

    return finale_table  # Changed to return finale_table instead of empty results list

def write_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Filename', 'Count < 0.50', 'Total Records'])
        csv_writer.writerows(results)


# Main execution
if __name__ == "__main__":
    directory_path = mode + "_" + model_name
    logging.info(f"Starting analysis...in directory: {directory_path}")
    # output_file_path = f"total_analysis_synonyms_{model_name}.csv"

    results = process_csv_files(directory_path)
    # write_results_to_csv(results, output_file_path)

    # print(f"Processing complete. Results written to {output_file_path}")