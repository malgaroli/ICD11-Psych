#!/bin/bash
#SBATCH --job-name=llm_icd11_job
#SBATCH --nodes=1			# 1 node -> max 4 GPUs, 2 nodes -> max 8 GPUs -> adapt that in the script
#SBATCH --ntasks=1	
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:a100:3                   	# uses 2 GPUs - 1 node is sufficient
#SBATCH --output=log_files/logs/llm_predict_%j.log	
#SBATCH --error=log_files/errors/llm_predict_%j.err 

# Setup conda
eval "$(conda shell.bash hook)"
conda activate icd11_qwen3 # icd11_gpt2 #icd11_qwen3 

# Your actual job command(s) go here
python /gpfs/data/schultebrauckslab/Users/muellv01/ICD11-Psych/code/diagnosis_huggingface_ddx.py

