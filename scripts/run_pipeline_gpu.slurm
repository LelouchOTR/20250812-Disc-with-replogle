#!/usr/bin/env bash
#SBATCH --mail-user=mohammad@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=discrepancy_vae_pipeline
#SBATCH --output=/data/gidb/shared/results/tmp/replogle/outputs/%x_%j/logs/pipeline.log
#SBATCH --error=/data/gidb/shared/results/tmp/replogle/outputs/%x_%j/logs/pipeline.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_normal_stud
#SBATCH --constraint="turing|ampere"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# Source conda
source /home/mohammad/Downloads/nobackup/anacondaex/etc/profile.d/conda.sh

# Activate environment (adjust name as needed)
conda activate scvi-env

# GPU debug info - verify GPU availability
echo "===== GPU Debugging Information ====="
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "import torch; print(f'PyTorch GPU: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo "===================================="

# Set working directory to project root
cd /home/mohammad/Documents/2025_HiWi/20250812-Disc-with-replogle

# Create output directories
mkdir -p /data/gidb/shared/results/tmp/replogle/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/logs

# Run the full pipeline
CMD="python run_pipeline.py all --seed 42"

echo "Starting DiscrepancyVAE pipeline..."
echo "Command: $CMD"
$CMD

# Exit with status of last command
exit $?
