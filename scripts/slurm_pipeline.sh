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
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Source conda
source /home/mohammad/Downloads/nobackup/anacondaex/etc/profile.d/conda.sh

# Activate environment
conda activate disc-with-replogle

# Set working directory to project root
cd /home/mohammad/Documents/2025_HiWi/20250812-Disc-with-replogle

# Create output directories
mkdir -p /data/gidb/shared/results/tmp/replogle/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/logs

# Run the full pipeline
CMD="python run_pipeline.py all --seed 42"
# CMD="python run_pipeline.py process train evaluate --seed 42 --run-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250817_222444/processed"
# CMD="python scripts/04_train.py --data-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250816_000002/processed --graph-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250816_000002/graphs --output-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250816_000002/models --log-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250816_000002/logs --config model_config --seed 42 --device cuda"
# CMD=" python scripts/05_eval.py --config configs/pipeline_config.yaml --model-path /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250819_011131/models/best_model.pth --data-path /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250819_011131/processed/val_data.h5ad --graph-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250819_011131/graphs --output-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250819_011131/evaluation --log-dir /data/gidb/shared/results/tmp/replogle/discrepancy_vae_20250819_011131/logs"
echo "Starting DiscrepancyVAE pipeline..."
echo "Command: $CMD"
$CMD

# Exit with status of last command
exit $?
