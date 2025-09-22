#!/bin/bash
#SBATCH --job-name=llm_icd11_job
#SBATCH --partition=a100_short            # or a100_dev/a100_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G                        # increase if you’ll offload a lot to CPU
#SBATCH --time=0-2:00:00
#SBATCH --output=log_files/logs/llm_predict_%j.log
#SBATCH --error=log_files/errors/llm_predict_%j.err

# Keep caches & offload OUT of $HOME to reduce file quota pressure
export HF_HOME=/gpfs/data/schultebrauckslab/Users/muellv01/.hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# Optional: make PyTorch allocator more fragmentation-friendly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Good NCCL defaults for single-node multi-GPU
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0

# Setup conda
eval "$(conda shell.bash hook)"
conda activate icd11_gpt2 # icd11_gpt2 #icd11_qwen3 

# Your actual job command(s) go here
python /gpfs/data/schultebrauckslab/Users/muellv01/ICD11-Psych/code/diagnosis_huggingface_ddx.py

