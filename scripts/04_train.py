#!/usr/bin/env python3
"""
Model training pipeline for DiscrepancyVAE on single-cell perturbation data.

This script loads processed dataset and optional gene adjacency graph,
initializes DiscrepancyVAE with hyperparameters from config, implements
training loop with validation monitoring, saves model checkpoints and
training logs, and stores final trained model in outputs/models/.
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import anndata as ad

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed
from src.models.discrepancy_vae import DiscrepancyVAE, get_model_summary
from torch_geometric.loader import DataLoader as PyGDataLoader
import pickle

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Custom exception for training errors."""
    pass


class ModelTrainer:
    """
    Trainer class for DiscrepancyVAE model.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path, log_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir = self.output_dir

        self.training_params = config.get('training', {})
        self.model_config = config.get('model_config', {})

        self.epochs = int(self.training_params.get('epochs', 100))
        self.batch_size = int(self.training_params.get('batch_size', 128))
        self.learning_rate = float(self.training_params.get('learning_rate', 1e-3))
        self.weight_decay = float(self.training_params.get('weight_decay', 1e-5))
        self.gradient_clip = float(self.training_params.get('gradient_clip', 1.0))

        self.validation_freq = int(self.training_params.get('validation_freq', 5))
        self.checkpoint_freq = int(self.training_params.get('checkpoint_freq', 10))
        self.early_stopping_patience = int(self.training_params.get('early_stopping_patience', 20))
        self.save_best_only = bool(self.training_params.get('save_best_only', True))

        scheduler_params = self.training_params.get('scheduler', {})
        self.use_scheduler = bool(scheduler_params.get('enabled', True))
        self.scheduler_type = str(scheduler_params.get('type', 'reduce_on_plateau'))
        self.scheduler_params = scheduler_params.get('params', {})

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.adjacency_matrix = None
        self.graph_gene_idx = None
        self.graph_gene_mask = None

        self.start_epoch = 0
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.val_epochs = []

        logger.info(f"Initialized ModelTrainer with output dir: {output_dir}")

    def load_data(self, cell_graph_path: Path) -> None:
        logger.info(f"Loading cell graphs from {cell_graph_path}...")

        if not cell_graph_path.exists():
            raise TrainingError(f"Cell graph file not found: {cell_graph_path}")

        with open(cell_graph_path, 'rb') as f:
            all_cell_graphs = pickle.load(f)

        logger.info(f"Loaded {len(all_cell_graphs)} total cell graphs.")

        # Split graphs into train and validation sets based on the 'split' attribute
        train_graphs = [g for g in all_cell_graphs if g.split == 'train']
        val_graphs = [g for g in all_cell_graphs if g.split == 'validation']

        if not train_graphs:
            raise TrainingError("No training data found in cell graphs.")
        if not val_graphs:
            logger.warning("No validation data found in cell graphs. Validation will be skipped.")

        logger.info(f"Train graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}")

        num_workers = int(self.training_params.get('num_workers', 4))

        self.train_loader = PyGDataLoader(
            train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        if val_graphs:
            self.val_loader = PyGDataLoader(
                val_graphs,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        else:
            self.val_loader = None

        # Set input_dim from the first graph
        self.input_dim = train_graphs[0].num_node_features
        self.num_genes = train_graphs[0].num_nodes

        logger.info(f"Created PyG data loaders - Input dim: {self.input_dim}, Num genes: {self.num_genes}")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader) if self.val_loader else 0}")

    def initialize_model(self) -> None:
        logger.info("Initializing DiscrepancyVAE model...")

        # The adjacency matrix is now part of the data batch, so we pass None here.
        # The model needs to be adapted to receive the graph structure from the batch.
        self.model = DiscrepancyVAE(
            input_dim=self.num_genes, # or self.input_dim depending on model design
            config=self.model_config,
            adjacency_matrix=None,
            graph_gene_idx=None
        )

        self.model.to(self.device)

        model_summary = get_model_summary(self.model)
        logger.info(f"Model architecture:\n{model_summary}")

        optimizer_type = str(self.training_params.get('optimizer', 'adam')).lower()

        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise TrainingError(f"Unsupported optimizer: {optimizer_type}")

        if self.use_scheduler:
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=float(self.scheduler_params.get('factor', 0.5)),
                    patience=int(self.scheduler_params.get('patience', 10))
                )
            elif self.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epochs,
                    eta_min=float(self.scheduler_params.get('eta_min', 1e-6))
                )
            elif self.scheduler_type == 'exponential':
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=float(self.scheduler_params.get('gamma', 0.95))
                )
            else:
                logger.warning(f"Unknown scheduler type: {self.scheduler_type}")
                self.scheduler = None

        tensorboard_dir = self.log_dir / 'tensorboard'
        tensorboard_dir.mkdir(exist_ok=True)
        if TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard writer: {e}")
                self.writer = None
        else:
            self.writer = None

        logger.info(f"Initialized model with {self.model._count_parameters():,} parameters")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        epoch_losses = defaultdict(list)

        pbar = tqdm(total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}/{self.epochs} [{datetime.now().strftime('%H:%M:%S')}]",
                    position=0,
                    leave=True,
                    dynamic_ncols=True)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            # The model now needs to handle the PyG Batch object
            x = batch.x
            is_control = batch.is_control

            # The model needs to be adapted to get perturbation info from the batch
            perturbation_labels = None # Placeholder

            self.optimizer.zero_grad()

            # The model call needs to be adapted for graph data
            model_output = self.model(batch)

            loss_dict = self.model.compute_loss(
                x, model_output, is_control, perturbation_labels
            )

            total_loss = loss_dict['total_loss']

            total_loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())

            pbar.update(1)
            if batch_idx % 10 == 0:  # Update postfix every 10 batches
                pbar.set_postfix({
                    'L': f"{total_loss.item():.0f}",
                    'RL': f"{loss_dict.get('reconstruction_loss', 0):.0f}",
                    'KL': f"{loss_dict.get('kl_loss', 0):.0f}",
                    'DL': f"{loss_dict.get('discrepancy_loss', 0):.0f}",
                    'GL': f"{loss_dict.get('graph_reg_loss', 0):.2f}"
                })

        pbar.close()

        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = float(np.mean(values))

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        epoch_losses = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', position=0, leave=True):
                batch = batch.to(self.device)
                x = batch.x
                is_control = batch.is_control
                perturbation_labels = None # Placeholder

                model_output = self.model(batch)

                loss_dict = self.model.compute_loss(
                    x, model_output, is_control, perturbation_labels
                )

                for key, value in loss_dict.items():
                    epoch_losses[key].append(value.item())

        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = float(np.mean(values))

        return epoch_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None

        metrics = {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch)
        }

        if not self.save_best_only or epoch % self.checkpoint_freq == 0:
            checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            self.model.save_checkpoint(
                checkpoint_path, epoch, optimizer_state, scheduler_state, metrics
            )

        if is_best:
            best_path = self.models_dir / 'best_model.pth'
            self.model.save_checkpoint(
                best_path, epoch, optimizer_state, scheduler_state, metrics
            )
            logger.info(f"Saved best model at epoch {epoch}")

        latest_path = self.models_dir / 'latest_model.pth'
        self.model.save_checkpoint(
            latest_path, epoch, optimizer_state, scheduler_state, metrics
        )

    def log_metrics(self, epoch: int, train_metrics: Dict[str, float],
                    val_metrics: Optional[Dict[str, float]] = None) -> None:
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key}', value, epoch)

            current_lr = float(self.optimizer.param_groups[0]['lr'])
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

        for key, value in train_metrics.items():
            self.train_history[key].append(value)

        if val_metrics:
            self.val_epochs.append(epoch)
            for key, value in val_metrics.items():
                self.val_history[key].append(value)

    def plot_training_curves(self) -> None:
        logger.info("Plotting training curves...")

        num_plots = len(self.train_history.keys())
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        fig.suptitle('Training Progress', fontsize=16)

        for i, (key, values) in enumerate(self.train_history.items()):
            ax = axes[i]
            ax.plot(values, label='Train', alpha=0.8)
            if self.val_history[key]:
                ax.plot(self.val_epochs, self.val_history[key], label='Validation', alpha=0.8, marker='o')
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.plots_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {plot_path}")

    def save_training_summary(self) -> None:
        logger.info("Saving training summary...")

        summary = {
            'training_config': self.config,
            'model_architecture': {
                'input_dim': int(self.input_dim),
                'total_parameters': int(self.model._count_parameters()),
                'model_config': self.model_config
            },
            'training_results': {
                'total_epochs': int(len(self.train_history['total_loss'])),
                'best_epoch': int(self.best_epoch),
                'best_val_loss': float(self.best_val_loss),
                'final_train_loss': float(self.train_history['total_loss'][-1]) if self.train_history[
                    'total_loss'] else None,
                'final_val_loss': float(self.val_history['total_loss'][-1]) if self.val_history['total_loss'] else None
            },
            'training_history': {
                'train': dict(self.train_history),
                'validation': dict(self.val_history)
            },
            'training_time': datetime.now().isoformat(),
            'device': str(self.device)
        }

        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.train_history:
            train_df = pd.DataFrame(self.train_history)
            train_df.index.name = 'epoch'
            train_df.to_csv(self.log_dir / 'train_history.csv')

        if self.val_history:
            val_df = pd.DataFrame(self.val_history)
            val_df.index.name = 'epoch'
            val_df.to_csv(self.log_dir / 'val_history.csv')

        logger.info(f"Saved training summary to {summary_path}")

    def resume_from_checkpoint(self, resume_path: Path) -> None:
        if not resume_path.exists():
            raise TrainingError(f"Resume checkpoint not found: {resume_path}")

        logger.info(f"Resuming training from checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise TrainingError("Checkpoint does not contain model_state_dict")

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        self.best_val_loss = float(checkpoint.get('metrics', {}).get('best_val_loss', float('inf')))
        self.best_epoch = int(checkpoint.get('metrics', {}).get('best_epoch', 0))
        self.epochs_without_improvement = int(checkpoint.get('metrics', {}).get('epochs_without_improvement', 0))

        train_history = checkpoint.get('metrics', {}).get('train_history', {})
        val_history = checkpoint.get('metrics', {}).get('val_history', {})

        for key, values in train_history.items():
            self.train_history[key].extend(values)

        for key, values in val_history.items():
            self.val_history[key].extend(values)

        logger.info(f"Resumed from epoch {self.start_epoch - 1}, best val loss: {self.best_val_loss:.4f}")

    def train(self) -> None:
        logger.info("Starting training...")

        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start_time = time.time()
                train_metrics = self.train_epoch(epoch)
                epoch_end_time = time.time()

                val_metrics = None
                if epoch % self.validation_freq == 0 or epoch == self.epochs - 1:
                    val_metrics = self.validate_epoch(epoch)

                    val_loss = val_metrics['total_loss']
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                        is_best = True
                    else:
                        self.epochs_without_improvement += self.validation_freq
                        is_best = False

                    logger.info(f"Epoch {epoch + 1}/{self.epochs} | "
                                f"Train Loss: {train_metrics['total_loss']:.2f} | "
                                f"Val Loss: {val_metrics['total_loss']:.2f} | "
                                f"Best Val Loss: {self.best_val_loss:.2f} at epoch {self.best_epoch + 1} | "
                                f"LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
                                f"Time: {epoch_end_time - epoch_start_time:.2f}s")

                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1} "
                                    f"(no improvement for {self.epochs_without_improvement} epochs)")
                        break

                    self.save_checkpoint(epoch, is_best)

                    if self.scheduler and self.scheduler_type == 'reduce_on_plateau':
                        self.scheduler.step(val_loss)

                if self.scheduler and self.scheduler_type != 'reduce_on_plateau':
                    self.scheduler.step()

                self.log_metrics(epoch, train_metrics, val_metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            if self.writer:
                self.writer.close()

            self.plot_training_curves()
            self.save_training_summary()

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Train DiscrepancyVAE model")
    parser.add_argument("--config", type=str, default="model_config", help="Configuration file name")
    parser.add_argument("--data-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/processed",
                        help="Processed data directory")
    parser.add_argument("--graph-dir", type=str, help="Gene adjacency graph directory")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/models",
                        help="Output directory")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--beta", type=float, help="Override the beta value for KL divergence weight")
    parser.add_argument("--dropout-rate", type=float, help="Override the dropout rate for the model")

    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '04_train.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    set_global_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            try:
                # Check if CUDA is actually working
                torch.cuda.init()
                device = torch.device('cuda')
            except RuntimeError as e:
                logger.warning(f"CUDA available but initialization failed: {e}")
                logger.warning("Falling back to CPU.")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

    logger.info(f"Using device: {device}")

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Override config with command-line arguments if provided
    if args.beta is not None:
        config['model_config']['loss_params']['beta'] = args.beta
        logger.info(f"Overriding beta with command-line value: {args.beta}")
    if args.dropout_rate is not None:
        config['model_config']['model_params']['dropout_rate'] = args.dropout_rate
        logger.info(f"Overriding dropout_rate with command-line value: {args.dropout_rate}")

    logger.info("Starting DiscrepancyVAE training pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        trainer = ModelTrainer(config, args.output_dir, args.log_dir, device)

        trainer.load_data(args.data_dir)

        if args.graph_dir:
            trainer.load_graph(args.graph_dir)
        
        trainer._harmonize_genes()

        trainer.initialize_model()

        if args.resume:
            trainer.resume_from_checkpoint(Path(args.resume))

        trainer.train()

        logger.info("Training completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())