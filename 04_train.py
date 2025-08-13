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

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed
from src.models.discrepancy_vae import (
    DiscrepancyVAE, SingleCellDataset, create_data_loaders, get_model_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Custom exception for training errors."""
    pass


class ModelTrainer:
    """
    Trainer class for DiscrepancyVAE model.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path, device: torch.device):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration dictionary
            output_dir: Output directory for models and logs
            device: Device to train on
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.models_dir = self.output_dir / 'models'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.models_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract training parameters
        self.training_params = config.get('training', {})
        self.model_config = config.get('model_config', {})
        
        # Training hyperparameters - ensure proper types
        self.epochs = int(self.training_params.get('epochs', 100))
        self.batch_size = int(self.training_params.get('batch_size', 128))
        self.learning_rate = float(self.training_params.get('learning_rate', 1e-3))
        self.weight_decay = float(self.training_params.get('weight_decay', 1e-5))
        self.gradient_clip = float(self.training_params.get('gradient_clip', 1.0))
        
        # Validation and checkpointing - ensure proper types
        self.validation_freq = int(self.training_params.get('validation_freq', 5))
        self.checkpoint_freq = int(self.training_params.get('checkpoint_freq', 10))
        self.early_stopping_patience = int(self.training_params.get('early_stopping_patience', 20))
        self.save_best_only = bool(self.training_params.get('save_best_only', True))
        
        # Scheduler parameters
        scheduler_params = self.training_params.get('scheduler', {})
        self.use_scheduler = bool(scheduler_params.get('enabled', True))
        self.scheduler_type = str(scheduler_params.get('type', 'reduce_on_plateau'))
        self.scheduler_params = scheduler_params.get('params', {})
        
        # Initialize tracking variables
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.adjacency_matrix = None
        
        # Training history
        self.start_epoch = 0
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        logger.info(f"Initialized ModelTrainer with output dir: {output_dir}")
    
    def load_data(self, data_dir: Path, use_raw: bool = False) -> None:
        """
        Load processed training and validation data.
        
        Args:
            data_dir: Directory containing processed data
            use_raw: Whether to use raw counts
        """
        logger.info("Loading processed data...")
        
        data_dir = Path(data_dir)
        
        # Load training data
        train_file = data_dir / 'train_data.h5ad'
        if not train_file.exists():
            raise TrainingError(f"Training data not found: {train_file}")
        
        adata_train = ad.read_h5ad(train_file)
        logger.info(f"Loaded training data: {adata_train.shape}")
        
        # Load validation data - check for both possible filenames
        val_file = data_dir / 'validation_data.h5ad'
        alt_val_file = data_dir / 'val_data.h5ad'
        
        if val_file.exists():
            adata_val = ad.read_h5ad(val_file)
        elif alt_val_file.exists():
            adata_val = ad.read_h5ad(alt_val_file)
            logger.info(f"Loaded validation data (alternative filename): {adata_val.shape}")
        else:
            raise TrainingError(f"Validation data not found. Checked both {val_file} and {alt_val_file}")
        
        logger.info(f"Loaded validation data: {adata_val.shape}")
        
        # Create data loaders
        num_workers = int(self.training_params.get('num_workers', 4))
        self.train_loader, self.val_loader = create_data_loaders(
            adata_train, adata_val,
            batch_size=self.batch_size,
            num_workers=num_workers,
            use_raw=use_raw
        )
        
        # Store input dimension
        self.input_dim = adata_train.n_vars
        
        logger.info(f"Created data loaders - Train batches: {len(self.train_loader)}, "
                   f"Val batches: {len(self.val_loader)}")
    
    def load_graph(self, graph_dir: Optional[Path] = None) -> Optional[torch.Tensor]:
        """
        Load gene adjacency graph if available.
        
        Args:
            graph_dir: Directory containing graph files
            
        Returns:
            Adjacency matrix tensor or None
        """
        if graph_dir is None:
            logger.info("No graph directory specified, skipping graph loading")
            return None
        
        graph_dir = Path(graph_dir)
        
        # Try to load adjacency matrix
        adj_file = graph_dir / 'adjacency_matrix.npz'
        nodes_file = graph_dir / 'node_mapping.json'
        
        if adj_file.exists() and nodes_file.exists():
            logger.info("Loading gene adjacency graph...")
            
            # Load adjacency matrix
            from scipy import sparse
            adj_matrix = sparse.load_npz(adj_file)
            
            # Load node mapping
            with open(nodes_file, 'r') as f:
                node_mapping = json.load(f)
            
            # Convert to dense tensor
            adj_tensor = torch.from_numpy(adj_matrix.toarray()).float()
            
            logger.info(f"Loaded adjacency matrix: {adj_tensor.shape}")
            return adj_tensor.to(self.device)
        
        else:
            logger.warning(f"Graph files not found in {graph_dir}, proceeding without graph")
            return None
    
    def initialize_model(self) -> None:
        """Initialize DiscrepancyVAE model."""
        logger.info("Initializing DiscrepancyVAE model...")
        
        # Create model
        self.model = DiscrepancyVAE(
            input_dim=self.input_dim,
            config=self.model_config
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Print model summary
        model_summary = get_model_summary(self.model)
        logger.info(f"Model architecture:\n{model_summary}")
        
        # Initialize optimizer
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
        
        # Initialize scheduler
        if self.use_scheduler:
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=float(self.scheduler_params.get('factor', 0.5)),
                    patience=int(self.scheduler_params.get('patience', 10)),
                    verbose=True
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
        
        # Initialize tensorboard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.logs_dir / 'tensorboard')
        else:
            self.writer = None
        
        logger.info(f"Initialized model with {self.model._count_parameters():,} parameters")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = defaultdict(list)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            x = batch['x'].to(self.device)
            is_control = batch['is_control'].to(self.device)
            perturbation_labels = batch['perturbation_label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            model_output = self.model(x)
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                x, model_output, is_control, perturbation_labels, self.adjacency_matrix
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'KL': f"{loss_dict['kl_loss'].item():.4f}",
                'Disc': f"{loss_dict['discrepancy_loss'].item():.4f}"
            })
        
        # Compute epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = float(np.mean(values))
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                x = batch['x'].to(self.device)
                is_control = batch['is_control'].to(self.device)
                perturbation_labels = batch['perturbation_label'].to(self.device)
                
                # Forward pass
                model_output = self.model(x)
                
                # Compute loss
                loss_dict = self.model.compute_loss(
                    x, model_output, is_control, perturbation_labels, self.adjacency_matrix
                )
                
                # Record losses
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value.item())
        
        # Compute epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = float(np.mean(values))
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Prepare checkpoint data
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        
        metrics = {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch)
        }
        
        # Save regular checkpoint
        if not self.save_best_only or epoch % self.checkpoint_freq == 0:
            checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            self.model.save_checkpoint(
                checkpoint_path, epoch, optimizer_state, scheduler_state, metrics
            )
        
        # Save best model
        if is_best:
            best_path = self.models_dir / 'best_model.pth'
            self.model.save_checkpoint(
                best_path, epoch, optimizer_state, scheduler_state, metrics
            )
            logger.info(f"Saved best model at epoch {epoch}")
        
        # Save latest model
        latest_path = self.models_dir / 'latest_model.pth'
        self.model.save_checkpoint(
            latest_path, epoch, optimizer_state, scheduler_state, metrics
        )
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float],
                   val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log learning rate
        current_lr = float(self.optimizer.param_groups[0]['lr'])
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Store in history
        for key, value in train_metrics.items():
            self.train_history[key].append(value)
        
        if val_metrics:
            for key, value in val_metrics.items():
                self.val_history[key].append(value)
        
        # Print metrics
        train_loss = train_metrics.get('total_loss', 0)
        log_msg = f"Epoch {epoch+1:3d}/{self.epochs} - Train Loss: {train_loss:.4f}"
        
        if val_metrics:
            val_loss = val_metrics.get('total_loss', 0)
            log_msg += f" - Val Loss: {val_loss:.4f}"
        
        log_msg += f" - LR: {current_lr:.2e}"
        logger.info(log_msg)
    
    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        logger.info("Plotting training curves...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot total loss
        axes[0, 0].plot(self.train_history['total_loss'], label='Train', alpha=0.8)
        if self.val_history['total_loss']:
            axes[0, 0].plot(self.val_history['total_loss'], label='Validation', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot reconstruction loss
        axes[0, 1].plot(self.train_history['reconstruction_loss'], label='Train', alpha=0.8)
        if self.val_history['reconstruction_loss']:
            axes[0, 1].plot(self.val_history['reconstruction_loss'], label='Validation', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot KL loss
        axes[1, 0].plot(self.train_history['kl_loss'], label='Train', alpha=0.8)
        if self.val_history['kl_loss']:
            axes[1, 0].plot(self.val_history['kl_loss'], label='Validation', alpha=0.8)
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot discrepancy loss
        axes[1, 1].plot(self.train_history['discrepancy_loss'], label='Train', alpha=0.8)
        if self.val_history['discrepancy_loss']:
            axes[1, 1].plot(self.val_history['discrepancy_loss'], label='Validation', alpha=0.8)
        axes[1, 1].set_title('Discrepancy Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {plot_path}")
    
    def save_training_summary(self) -> None:
        """Save training summary and metrics."""
        logger.info("Saving training summary...")
        
        # Prepare summary data
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
                'final_train_loss': float(self.train_history['total_loss'][-1]) if self.train_history['total_loss'] else None,
                'final_val_loss': float(self.val_history['total_loss'][-1]) if self.val_history['total_loss'] else None
            },
            'training_history': {
                'train': dict(self.train_history),
                'validation': dict(self.val_history)
            },
            'training_time': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Save as JSON
        summary_path = self.logs_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save training history as CSV
        if self.train_history:
            train_df = pd.DataFrame(self.train_history)
            train_df.index.name = 'epoch'
            train_df.to_csv(self.logs_dir / 'train_history.csv')
        
        if self.val_history:
            val_df = pd.DataFrame(self.val_history)
            val_df.index.name = 'epoch'
            val_df.to_csv(self.logs_dir / 'val_history.csv')
        
        logger.info(f"Saved training summary to {summary_path}")
    
    def resume_from_checkpoint(self, resume_path: Path) -> None:
        """
        Resume training from a checkpoint.
        
        Args:
            resume_path: Path to the checkpoint file
        """
        if not resume_path.exists():
            raise TrainingError(f"Resume checkpoint not found: {resume_path}")
        
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        
        # Load checkpoint
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        # Restore model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise TrainingError("Checkpoint does not contain model_state_dict")
        
        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        self.best_val_loss = float(checkpoint.get('metrics', {}).get('best_val_loss', float('inf')))
        self.best_epoch = int(checkpoint.get('metrics', {}).get('best_epoch', 0))
        self.epochs_without_improvement = int(checkpoint.get('metrics', {}).get('epochs_without_improvement', 0))
        
        # Restore training history
        train_history = checkpoint.get('metrics', {}).get('train_history', {})
        val_history = checkpoint.get('metrics', {}).get('val_history', {})
        
        for key, values in train_history.items():
            self.train_history[key].extend(values)
            
        for key, values in val_history.items():
            self.val_history[key].extend(values)
        
        logger.info(f"Resumed from epoch {self.start_epoch - 1}, best val loss: {self.best_val_loss:.4f}")

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.epochs):
                # Training phase
                train_metrics = self.train_epoch(epoch)
                
                # Validation phase
                val_metrics = None
                if epoch % self.validation_freq == 0 or epoch == self.epochs - 1:
                    val_metrics = self.validate_epoch(epoch)
                    
                    # Check for improvement
                    val_loss = val_metrics['total_loss']
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                        is_best = True
                    else:
                        self.epochs_without_improvement += self.validation_freq
                        is_best = False
                    
                    # Early stopping check
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1} "
                                  f"(no improvement for {self.epochs_without_improvement} epochs)")
                        break
                    
                    # Save checkpoint
                    self.save_checkpoint(epoch, is_best)
                    
                    # Update scheduler
                    if self.scheduler and self.scheduler_type == 'reduce_on_plateau':
                        self.scheduler.step(val_loss)
                
                # Update scheduler (non-plateau schedulers)
                if self.scheduler and self.scheduler_type != 'reduce_on_plateau':
                    self.scheduler.step()
                
                # Log metrics
                self.log_metrics(epoch, train_metrics, val_metrics)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Clean up
            if self.writer:
                self.writer.close()
            
            # Save final results
            self.plot_training_curves()
            self.save_training_summary()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")


def main():
    """Main function for model training pipeline."""
    parser = argparse.ArgumentParser(description="Train DiscrepancyVAE model")
    parser.add_argument("--config", type=str, default="model_config", help="Configuration file name")
    parser.add_argument("--data-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/processed", help="Processed data directory")
    parser.add_argument("--graph-dir", type=str, help="Gene adjacency graph directory")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/models", help="Output directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set random seed
    set_global_seed(args.seed)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    logger.info("Starting DiscrepancyVAE training pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config, args.output_dir, device)
        
        # Load data
        trainer.load_data(args.data_dir)
        
        # Load graph (optional)
        if args.graph_dir:
            trainer.adjacency_matrix = trainer.load_graph(args.graph_dir)
        
        # Initialize model
        trainer.initialize_model()
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.resume_from_checkpoint(Path(args.resume))
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
