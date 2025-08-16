#!/bin/bash

#SBATCH --job-name=conda_env_setup_and_verify
#SBATCH --output=logs/conda_setup_and_verify_%j.log
#SBATCH --error=logs/conda_setup_and_verify_%j.err
#SBATCH --partition=gpu_normal_stud  # Partition from your slurm_pipeline.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere" # Optional: for specific GPU architectures
#SBATCH --time=23:30:00

# --- Configuration ---
ENV_NAME="disc-with-replogle"
CONDA_INIT_PATH="/home/mohammad/Downloads/nobackup/anacondaex/etc/profile.d/conda.sh"
# --- End Configuration ---


# Create a directory for logs if it doesn't exist
mkdir -p logs

echo "--- Environment Setup Job ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "GPU allocated: $(nvidia-smi -L)"
echo "-----------------------------"

# 1. Source Conda
echo "Initializing Conda from $CONDA_INIT_PATH"
source "$CONDA_INIT_PATH"
if [ $? -ne 0 ]; then
    echo "--- ERROR: Conda initialization failed. Please check the CONDA_INIT_PATH in this script. ---"
    exit 1
fi

# 2. Remove existing environment to ensure a clean installation
echo "Removing old conda environment '$ENV_NAME' if it exists..."
conda env remove -n "$ENV_NAME" -y

# 3. Create the new conda environment from the .yml file
echo "Creating new conda environment '$ENV_NAME' from environment.yml..."
conda env create -f environment.yml

# 4. Check if the creation was successful
if [ $? -ne 0 ]; then
    echo "--- ERROR: Conda environment creation failed. Check the error log. ---"
    exit 1
fi

echo "--- SUCCESS: Conda environment '$ENV_NAME' created. ---"

# 5. Activate the new environment and verify GPU support
echo ""
echo "--- Verifying PyTorch GPU Installation ---"

# Activate the environment
source "$CONDA_INIT_PATH" # Re-sourcing might be needed in some shell environments
conda activate "$ENV_NAME"

# Run the verification python script
python -c "
import torch
import sys

print(f'Python Version: {sys.version}')
print(f'PyTorch Version: {torch.__version__}')
print(f'PyTorch CUDA Version: {torch.version.cuda}')
print(f'PyTorch build with CUDA: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('\n--- VERIFICATION FAILED: PyTorch was not built with CUDA support. ---')
    sys.exit(1)

print(f'Number of GPUs available: {torch.cuda.device_count()}')

for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

print('\n--- VERIFICATION SUCCEEDED: PyTorch is correctly configured with GPU support. ---')

" 

if [ $? -eq 0 ]; then
    echo ""
    echo "The environment is ready. You can now use it in your training scripts."
else
    echo ""
    echo "Verification failed. Please check the log file: logs/conda_setup_and_verify_$SLURM_JOB_ID.log"
fi

echo "End time: $(date)"
