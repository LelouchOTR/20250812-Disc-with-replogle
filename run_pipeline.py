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


def create_run_dirs(config: dict, args) -> dict:
    """
    Create a timestamped run directory structure and configure a run-specific
    file logger. Returns a dict with paths for raw, processed, graphs, models,
    evaluation, logs, and cache directories (all as strings).

    Behavior:
    - Uses config.output_paths.base_output_dir and config.output_paths.* for naming.
    - Honors output_paths.timestamp_format and output_paths.use_timestamps.
    - Includes optional args.run_id when provided.
    - Writes run_metadata.json into the run root.
    """
    import json
    import subprocess
    from datetime import datetime

    # Resolve base output dir and naming parameters from config
    output_cfg = config.get('output_paths', {}) if isinstance(config, dict) else {}
    base_output_dir = Path(output_cfg.get('base_output_dir', 'outputs'))
    experiment_name = output_cfg.get('experiment_name', 'run')
    timestamp_format = output_cfg.get('timestamp_format', '%Y%m%d_%H%M%S')
    use_timestamps = bool(output_cfg.get('use_timestamps', True))

    run_id = getattr(args, 'run_id', None)

    # Build run folder name
    now = datetime.now()
    timestamp = now.strftime(timestamp_format)
    if run_id and not use_timestamps:
        run_folder_name = f"{experiment_name}_{run_id}"
    elif run_id:
        run_folder_name = f"{experiment_name}_{run_id}_{timestamp}"
    else:
        run_folder_name = f"{experiment_name}_{timestamp}" if use_timestamps else f"{experiment_name}"

    # Create run directory and canonical subdirectories
    run_root = Path(base_output_dir) / run_folder_name
    subdirs = {
        'raw': run_root / 'raw',
        'processed': run_root / 'processed',
        'graphs': run_root / 'graphs',
        'models': run_root / 'models',
        'evaluation': run_root / 'evaluation',
        'logs': run_root / 'logs',
        'cache': run_root / 'cache'
    }

    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # Add a file handler to the root logger to capture pipeline logs inside run folder
    try:
        file_handler = logging.FileHandler(str(subdirs['logs'] / 'pipeline.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create run-specific log file handler: {e}")

    # Collect metadata about the run
    metadata = {
        'run_root': str(run_root),
        'run_name': run_folder_name,
        'timestamp': now.isoformat(),
        'seed': getattr(args, 'seed', None),
        'config_file': getattr(args, 'config', None)
    }

    # Try to capture git commit if available
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                          stderr=subprocess.DEVNULL).decode().strip()
        metadata['git_sha'] = git_sha
    except Exception:
        metadata['git_sha'] = None

    # Save metadata
    try:
        with open(run_root / 'run_metadata.json', 'w') as fh:
            json.dump(metadata, fh, indent=2)
    except Exception as e:
        logger.warning(f"Unable to write run metadata to {run_root}: {e}")

    logger.info(f"Created run directory: {run_root}")

    # Return string paths for compatibility with subprocess args used below
    return {k: str(v) for k, v in subdirs.items()}


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


def run_step(command: list):
    """Run a pipeline step as a subprocess with real-time output."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()

        # Capture any remaining stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            logger.warning(stderr_output)

        # Wait for process to complete and check return code
        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    except subprocess.CalledProcessError as e:
        raise PipelineError(f"Step failed: {' '.join(command)}")
    except Exception as e:
        raise PipelineError(f"Step failed: {' '.join(command)} - {e}")


def run_ingest(config: dict, run_dirs: dict, seed: int):
    """Run the data ingestion step."""
    logger.info("--- Running Ingestion Step ---")
    output_dir = run_dirs.get('raw')
    cmd = [
        "python", "scripts/01_ingest.py",
        "--output-dir", output_dir,
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info(f"--- Ingestion Step Complete (output: {output_dir}) ---")


def run_process(config: dict, run_dirs: dict, seed: int):
    """Run the data processing step."""
    logger.info("--- Running Processing Step ---")
    raw_data_dir = run_dirs.get('raw')

    # Find the main data file - try different naming patterns
    raw_data_files = [f for f in os.listdir(raw_data_dir) 
                     if f.endswith('.h5ad.gz') or f.endswith('.h5ad') or 'h5ad' in f]
        
    if not raw_data_files:
        # If no files match patterns, try all files
        all_files = [f for f in os.listdir(raw_data_dir)]
        if len(all_files) == 1:
            raw_data_file = os.path.join(raw_data_dir, all_files[0])
            logger.warning(f"Using unexpected file as data: {all_files[0]}")
        else:
            raise PipelineError(f"Could not find raw data file in {raw_data_dir}")
    elif len(raw_data_files) > 1:
        # Prefer .h5ad files
        h5ad_files = [f for f in raw_data_files if f.endswith('.h5ad')]
        if h5ad_files:
            raw_data_file = os.path.join(raw_data_dir, h5ad_files[0])
            logger.warning(f"Multiple data files found, using: {h5ad_files[0]}")
        else:
            raw_data_file = os.path.join(raw_data_dir, raw_data_files[0])
            logger.warning(f"Multiple data files found, using: {raw_data_files[0]}")
    else:
        raw_data_file = os.path.join(raw_data_dir, raw_data_files[0])

    output_dir = run_dirs.get('processed')
    cmd = [
        "python", "scripts/02_process.py",
        "--input", raw_data_file,
        "--output-dir", output_dir,
        "--config", "data_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info(f"--- Processing Step Complete (output: {output_dir}) ---")


def run_graphs(config: dict, run_dirs: dict, seed: int):
    """Run the graph generation step."""
    logger.info("--- Running Graph Generation Step ---")
    output_dir = run_dirs.get('graphs')
    cmd = [
        "python", "scripts/03_graphs.py",
        "--output-dir", output_dir,
        "--config", "graph_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info(f"--- Graph Generation Step Complete (output: {output_dir}) ---")


def run_train(config: dict, run_dirs: dict, seed: int):
    """Run the model training step."""
    logger.info("--- Running Training Step ---")
    data_dir = run_dirs.get('processed')
    model_dir = run_dirs.get('models')
    graph_dir = run_dirs.get('graphs')

    cmd = [
        "python", "scripts/04_train.py",
        "--data-dir", data_dir,
        "--output-dir", model_dir,
        "--config", "model_config",
        "--seed", str(seed)
    ]
    if os.path.exists(graph_dir):
        cmd.extend(["--graph-dir", graph_dir])

    run_step(cmd)
    logger.info(f"--- Training Step Complete (models: {model_dir}) ---")


def run_evaluate(config: dict, run_dirs: dict, seed: int):
    """Run the model evaluation step."""
    logger.info("--- Running Evaluation Step ---")
    model_path = os.path.join(run_dirs.get('models'), "best_model.pth")
    data_path = os.path.join(run_dirs.get('processed'), "test_data.h5ad")
    output_dir = run_dirs.get('evaluation')

    cmd = [
        "python", "scripts/05_eval.py",
        "--model-path", model_path,
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--config", "pipeline_config",
        "--seed", str(seed)
    ]
    run_step(cmd)
    logger.info(f"--- Evaluation Step Complete (output: {output_dir}) ---")


def main():
    """Main function for pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the single-cell perturbation analysis pipeline.")
    parser.add_argument(
        "steps",
        nargs='+',
        choices=['all', 'ingest', 'process', 'graphs', 'train', 'evaluate'],
        help="Pipeline steps to run."
    )
    parser.add_argument("--config", type=str, default="pipeline_config",
                        help="Main pipeline configuration file (without path or extension).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--run-id", type=str, help="Optional run id to include in the run folder name.")

    args = parser.parse_args()

    set_global_seed(args.seed)

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create timestamped run directory structure and configure run logging
    run_dirs = create_run_dirs(config, args)

    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['ingest', 'process', 'graphs', 'train', 'evaluate']

    logger.info(f"Running pipeline with steps: {', '.join(steps_to_run)}")

    for step in steps_to_run:
        if step == 'ingest':
            run_ingest(config, run_dirs, args.seed)
        elif step == 'process':
            run_process(config, run_dirs, args.seed)
        elif step == 'graphs':
            run_graphs(config, run_dirs, args.seed)
        elif step == 'train':
            run_train(config, run_dirs, args.seed)
        elif step == 'evaluate':
            run_evaluate(config, run_dirs, args.seed)

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
