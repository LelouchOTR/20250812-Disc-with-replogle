#!/usr/bin/env python3
"""
Main pipeline orchestration script for single-cell perturbation analysis.

This script provides a command-line interface for running individual pipeline
steps or the full end-to-end execution. It handles configuration validation,
error handling, logging, and seed management for reproducibility.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


def run_step(command: list):
    """Run a pipeline step as a subprocess."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(process.stdout)
        if process.stderr:
            logger.warning(process.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise PipelineError(f"Step failed: {' '.join(command)}")


def run_ingest(config: dict, seed: int):
    """Run the data ingestion step."""
    logger.info("--- Running Ingestion Step ---")
    output_dir = "/data/gidb/shared/results/tmp/replogle/raw"
    cmd = [
        "python", "01_ingest.py",
        "--output-dir", output_dir,
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info("--- Ingestion Step Complete ---")


def run_process(config: dict, seed: int):
    """Run the data processing step."""
    logger.info("--- Running Processing Step ---")
    raw_data_dir = "/data/gidb/shared/results/tmp/replogle/raw"
    # Find the main data file
    raw_data_file = ""
    for f in os.listdir(raw_data_dir):
        if f.endswith('.h5ad.gz') or f.endswith('.h5ad'):
            raw_data_file = os.path.join(raw_data_dir, f)
            break
    if not raw_data_file:
        raise PipelineError("Could not find raw data file in raw data directory.")
        
    output_dir = "/data/gidb/shared/results/tmp/replogle/processed"
    cmd = [
        "python", "02_process.py",
        "--input", raw_data_file,
        "--output-dir", output_dir,
        "--config", "data_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info("--- Processing Step Complete ---")


def run_graphs(config: dict, seed: int):
    """Run the graph generation step."""
    logger.info("--- Running Graph Generation Step ---")
    output_dir = "/data/gidb/shared/results/tmp/replogle/graphs"
    cmd = [
        "python", "03_graphs.py",
        "--output-dir", output_dir,
        "--config", "graph_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info("--- Graph Generation Step Complete ---")


def run_train(config: dict, seed: int):
    """Run the model training step."""
    logger.info("--- Running Training Step ---")
    data_dir = "/data/gidb/shared/results/tmp/replogle/processed"
    model_dir = "/data/gidb/shared/results/tmp/replogle/models"
    graph_dir = "/data/gidb/shared/results/tmp/replogle/graphs"
    
    cmd = [
        "python", "04_train.py",
        "--data-dir", data_dir,
        "--output-dir", model_dir,
        "--config", "model_config",
        "--seed", str(seed)
    ]
    if os.path.exists(graph_dir):
        cmd.extend(["--graph-dir", graph_dir])
        
    run_step(cmd)
    logger.info("--- Training Step Complete ---")


def run_evaluate(config: dict, seed: int):
    """Run the model evaluation step."""
    logger.info("--- Running Evaluation Step ---")
    model_path = "/data/gidb/shared/results/tmp/replogle/models/best_model.pth"
    data_path = "/data/gidb/shared/results/tmp/replogle/processed/test_data.h5ad"
    output_dir = "/data/gidb/shared/results/tmp/replogle/evaluation"
    
    cmd = [
        "python", "05_eval.py",
        "--model-path", model_path,
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--config", "pipeline_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info("--- Evaluation Step Complete ---")


def main():
    """Main function for pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the single-cell perturbation analysis pipeline.")
    parser.add_argument(
        "steps",
        nargs='+',
        choices=['all', 'ingest', 'process', 'graphs', 'train', 'evaluate'],
        help="Pipeline steps to run."
    )
    parser.add_argument("--config", type=str, default="pipeline_config", help="Main pipeline configuration file (without path or extension).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    config = load_config(args.config)
    
    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['ingest', 'process', 'graphs', 'train', 'evaluate']
        
    logger.info(f"Running pipeline with steps: {', '.join(steps_to_run)}")
    
    for step in steps_to_run:
        if step == 'ingest':
            run_ingest(config, args.seed)
        elif step == 'process':
            run_process(config, args.seed)
        elif step == 'graphs':
            run_graphs(config, args.seed)
        elif step == 'train':
            run_train(config, args.seed)
        elif step == 'evaluate':
            run_evaluate(config, args.seed)
            
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
