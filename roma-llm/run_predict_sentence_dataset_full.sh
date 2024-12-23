#!/bin/bash
#SBATCH --mem=128gb
#SBATCH -c4
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --killable
#SBATCH --requeue
#SBATCH --error=error_log_%j.txt
#SBATCH --output=log_%j.txt
#SBATCH --job-name=predict_sentence_dataset_full

# Change to the project directory
cd /cs/labs/guykatz/adielas/roma_llm/

# Provide the full path to the virtual environment
source /cs/labs/guykatz/adielas/roma_llm/venv/bin/activate

# Check if the virtual environment is activated
which python

# Run the Python script
python predict_sentence_dataset_full.py
