import os
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import time
from datetime import datetime

# Load the model and tokenizer
model_path = "./models/best_model"

# Get the current timestamp
current_time = datetime.now().strftime('%d-%m_%H-%M-%S')

# Create the log filename
log_filename = f'{model_path[9:]}_{current_time}.log'
# Create the timing CSV filename
timing_csv = f'timing_{model_path[9:]}_{current_time}.csv'

# Configure logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create timing CSV with headers
with open(timing_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Input_file_name', 'runtime_to_process'])

logging.info("=========================================")
logging.info("Starting predict_sentence_dataset_full.py...")

# Debugging: logging.info the current working directory and model path
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Model path: {model_path}")

start_time = time.time()

if not os.path.exists(model_path):
    logging.error(f"The model file does not exist at the specified path: {model_path}")
    raise FileNotFoundError(f"The model file does not exist at the specified path: {model_path}")

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check if CUDA is available and move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device: {}".format(device))
model.to(device)



# Set the model to evaluation mode
model.eval()

def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the GPU
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][0].item(), probs[0][1].item()  # negative, positive

def process_file(input_file, output_file):
    file_start_time = time.time()

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Sentence', 'Negative Probability', 'Positive Probability'])

        for line in infile:
            if line.startswith(("Original:", "Modified")):
                sentence = line.split(": ", 1)[1].strip()
                neg_prob, pos_prob = predict_sentiment(sentence)
                csv_writer.writerow([sentence, neg_prob, pos_prob])

    # Calculate processing time for this file
    file_processing_time = time.time() - file_start_time

    # Log the timing to CSV
    with open(timing_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(input_file), file_processing_time])

    return file_processing_time


# Process all files in the modified_sentences folder
input_folder = "full_modified_sentences"
output_folder = "full_sentiment_predictions"
os.makedirs(output_folder, exist_ok=True)

total_files = 0
total_processing_time = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", ".csv"))
        logging.info(f"Processing {filename}...")

        # Process file and get processing time
        file_time = process_file(input_path, output_path)
        total_files += 1
        total_processing_time += file_time

        logging.info(f"Finished processing {filename} in {file_time:.2f} seconds")

        if total_files % 10 == 0:
            logging.info(f"Processed {total_files} files so far.")


elapsed_time = time.time() - start_time
average_time = total_processing_time / total_files if total_files > 0 else 0

logging.info(f"predict_sentence_dataset_full.py execution time: {elapsed_time} seconds for model:{model_path[9:]}")
logging.info(f"Average processing time per file: {average_time:.2f} seconds")
logging.info("All files processed. Results saved in CSV format.")