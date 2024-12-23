#!/bin/bash
#SBATCH --mem=128gb
#SBATCH -c4
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --killable
#SBATCH --requeue
#SBATCH --error=error_log_%j.txt
#SBATCH --output=log_%j.txt
#SBATCH --job-name=analysis_full_prediction


cd /cs/labs/guykatz/adielas/roma_llm/
source venv/bin/activate
python analysis_full_prediction.py
