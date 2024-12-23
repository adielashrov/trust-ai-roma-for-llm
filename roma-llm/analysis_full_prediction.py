import os
import csv
import time
import logging
from datetime import datetime


def setup_logging(model_name):
    current_time = datetime.now().strftime('%d-%m_%H-%M-%S')
    log_filename = f'analysis_timing_{model_name}_{current_time}.log'

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    return current_time


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def process_csv_files(directory, model_name):
    results = []

    # Setup logging and get current time
    current_time = setup_logging(model_name)
    logging.info(f"Starting analysis for model: {model_name}")
    logging.info(f"Reading files from directory: {directory}")

    total_processed = 0
    total_time = 0

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            # Start timing for this file
            file_start_time = time.time()
            logging.info(f"Starting to process {filename}")

            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

                if len(rows) < 2 or len(rows[0]) < 2:
                    logging.warning(f"Skipping {filename}: Insufficient rows or columns")
                    continue

                # Get the two rightmost columns
                col1 = [row[-2] for row in rows]
                col2 = [row[-1] for row in rows]

                # Find the first row with numeric data
                start_row = 0
                for i in range(len(rows)):
                    if is_float(col1[i]) and is_float(col2[i]):
                        start_row = i
                        break

                # If no numeric data found, skip this file
                if start_row == len(rows) - 1:
                    logging.warning(f"Skipping {filename}: No numeric data found")
                    continue

                # Determine which column to process
                if float(col1[start_row]) > 0.50:
                    process_col = col1[start_row:]
                elif float(col2[start_row]) > 0.50:
                    process_col = col2[start_row:]
                else:
                    logging.warning(f"Skipping {filename}: No column with first value > 0.50")
                    continue

                # Count values less than 0.50 and total records
                count_less_than_50 = sum(1 for value in process_col[1:] if float(value) < 0.50)
                total_records = len(process_col) - 1  # Exclude the first row of numeric data

                # Calculate processing time for this file
                file_processing_time = time.time() - file_start_time

                # Add all results including processing time to the results list
                results.append([filename, count_less_than_50, total_records, file_processing_time])

                total_processed += 1
                total_time += file_processing_time

                logging.info(f"Processed {filename} in {file_processing_time} seconds")


    # Log summary statistics
    if total_processed > 0:
        avg_time = total_time / total_processed
        logging.info("Processing Summary:")
        logging.info(f"Total files processed: {total_processed}")
        logging.info(f"Total processing time: {total_time} seconds")
        logging.info(f"Average processing time per file: {avg_time} seconds")
    else:
        logging.warning("No files were processed")

    return results, current_time


def write_results_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Filename', 'Count < 0.50', 'Total Records', 'Processing Time (s)'])
        csv_writer.writerows(results)
    logging.info(f"Results written to {output_file}")


# Main execution
if __name__ == "__main__":
    directory_path = "full_sentiment_predictions"
    model_name = "best_model"  # You can modify this to match your model name

    # Time the entire process
    total_start_time = time.time()

    # Get results and current_time from process_csv_files
    results, current_time = process_csv_files(directory_path, model_name)

    # Create output filename with model name and timestamp
    output_file_path = f"total_analysis_{model_name}_{current_time}.csv"

    write_results_to_csv(results, output_file_path)

    total_time = time.time() - total_start_time
    logging.info(f"Total script execution time: {total_time} seconds")
    logging.info("Analysis complete")