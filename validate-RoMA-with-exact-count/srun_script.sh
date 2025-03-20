#!/bin/bash
#SBATCH --time=1-0
#SBATCH --mem=128G
#SBATCH --gres=gpu:1,vmem:10gb
#SBATCH --error=error_log_%j.txt
#SBATCH --output=log_%j.txt
#SBATCH --job-name=ver_acas_new_algorithm
#SBATCH --killable
#SBATCH --requeue

cd /cs/labs/guykatz/adielas/ver_on_marabou/
venv/bin/python3 ver_acas_new_algorithm.py
