#!/usr/bin/env python3
"""
DiscrepancyVAE model architecture for single-cell perturbation analysis.

This module implements a Variational Autoencoder specifically designed to model
discrepancies between control and perturbed conditions in single-cell data.
The model learns latent representations that capture perturbation effects
and can reconstruct gene expression profiles.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy import sparse
import anndata as ad
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscrepancyVAEError(Exception):
    """Custom exception for DiscrepancyVAE errors."""
    pass


class GraphEncoder(nn.Module):
    """
    Graph Convolutional Encoder network for DiscrepancyVAE.
    
    Maps gene expression data to latent space parameters (mean and log variance)
    using graph convolutions and global pooling.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 dropout_rate: float = 0.1, batch_norm: bool = True,
                 activation: str = 'relu'):
        """
        Initialize graph encoder network.
        
        Args:
            input_dim: Input dimension (number of features per gene, usually 1)
            hidden_dims: List of hidden layer dimensions for GCN
            latent_dim: Latent space dimension
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'elu', 'leaky_relu')
        """
        super(GraphEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.2)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation

        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        
        self.dropout = nn.Dropout(dropout_rate)

        # The input to the linear layers is now the output of the pooling layer
        self.mu_layer = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[1], latent_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GCNConv):
                nn.init.xavier_uniform_(module.lin.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph encoder using PyG batching.
        
        Args:
            x: Node features for the batch [num_nodes_in_batch, num_node_features]
            edge_index: Edge index tensor for the batch graph [2, num_edges_in_batch]
            batch: Batch vector for nodes [num_nodes_in_batch]
            
        Returns:
            Tuple of (mu, logvar, gene_embeddings) tensors
        """
        # Apply graph convolutions
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        gene_embeddings = self.conv2(h, edge_index)

        # Pool features across genes for each graph in the batch
        pooled_output = global_mean_pool(self.activation(gene_embeddings), batch) # [batch_size, hidden_dims[1]]

        # Latent space parameters
        mu = self.mu_layer(pooled_output)
        logvar = self.logvar_layer(pooled_output)

        return mu, logvar, gene_embeddings


class Decoder(nn.Module):
    """
    Decoder network for DiscrepancyVAE.
    
    Maps latent space representations back to gene expression space.
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout_rate: float = 0.1, batch_norm: bool = True,
                 activation: str = 'relu', output_activation: str = 'linear'):
        """
        Initialize decoder network.
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions (in reverse order)
            output_dim: Output dimension (number of genes)
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Choose activation function
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.2)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        
        # Choose output activation
        if isinstance(output_activation, str):
            if output_activation == 'linear':
                self.output_activation = nn.Identity()
            elif output_activation == 'sigmoid':
                self.output_activation = nn.Sigmoid()
            elif output_activation == 'softplus':
                self.output_activation = nn.Softplus()
            else:
                raise ValueError(f"Unsupported output activation: {output_activation}")
        else:
            self.output_activation = output_activation
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed output [batch_size, output_dim]
        """
        h = self.decoder_layers(z)
        output = self.output_layer(h)
        output = self.output_activation(output)
        
        return output


class DiscrepancyVAE(nn.Module):
    """
    Discrepancy Variational Autoencoder for single-cell perturbation analysis.
    
    This model learns to encode single-cell gene expression data into a latent space
    and can compute discrepancies between control and perturbed conditions.
    """
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        Initialize DiscrepancyVAE model.
        
        Args:
            input_dim: Input dimension (number of genes)
            config: Model configuration dictionary
        """
        super(DiscrepancyVAE, self).__init__()
        
        self.input_dim = input_dim
        self.config = config
        
        # Extract model parameters
        model_params = config.get('model_params', {})
        self.latent_dim = model_params.get('latent_dim', 32)
        self.hidden_dims = model_params.get('hidden_dims', [512, 256])
        self.dropout_rate = model_params.get('dropout_rate', 0.1)
        self.batch_norm = model_params.get('batch_norm', True)
        self.activation = model_params.get('activation', 'relu')
        self.output_activation = model_params.get('output_activation', 'linear')
        
        # Loss parameters
        loss_params = config.get('loss_params', {})
        self.beta = loss_params.get('beta', 1.0)  # KL divergence weight
        self.discrepancy_weight = loss_params.get('discrepancy_weight', 1.0)
        self.reconstruction_loss_type = loss_params.get('reconstruction_loss', 'mse')
        self.discrepancy_loss_type = loss_params.get('discrepancy_loss_type', 'l2_mean')
        self.mmd_gamma = loss_params.get('mmd_gamma', 1.0)

        # Build encoder and decoder
        self.encoder = GraphEncoder(
            input_dim=1, # Each gene's expression is a single feature
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        
        # Decoder hidden dims are reversed
        decoder_hidden_dims = self.hidden_dims[::-1]
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            activation=self.activation,
            output_activation=self.output_activation
        )

        logger.info(f"Initialized DiscrepancyVAE with {self._count_parameters():,} parameters")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Node features for the batch [num_nodes_in_batch, num_node_features]
            edge_index: Edge index for the batch [2, num_edges_in_batch]
            batch: Batch vector for nodes [num_nodes_in_batch]
            
        Returns:
            Tuple of (mu, logvar, gene_embeddings) tensors
        """
        mu, logvar, h_batch = self.encoder(x, edge_index, batch)
        return mu, logvar, h_batch

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean tensor [batch_size, latent_dim]
            logvar: Log variance tensor [batch_size, latent_dim]
            
        Returns:
            Sampled latent tensor [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output space.
        
        Args:
            z: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed output [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, data: Any) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric Batch object
            
        Returns:
            Dictionary containing model outputs
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode
        mu, logvar, gene_embeddings = self.encode(x, edge_index, batch)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        # The decoder expects one vector per cell, so we use z
        x_recon_flat = self.decode(z)
        
        # The output is of shape [batch_size, num_genes], which is what we need for the loss
        x_recon = x_recon_flat

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'gene_embeddings': gene_embeddings
        }
    
    def compute_reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x: Original input [batch_size, input_dim]
            x_recon: Reconstructed input [batch_size, input_dim]
            
        Returns:
            Reconstruction loss tensor
        """
        if self.reconstruction_loss_type == 'mse':
            return F.mse_loss(x_recon, x, reduction='mean')
        elif self.reconstruction_loss_type == 'bce':
            return F.binary_cross_entropy(x_recon, x, reduction='mean')
        elif self.reconstruction_loss_type == 'poisson':
            # Poisson loss for count data
            return F.poisson_nll_loss(x_recon, x, log_input=False, reduction='mean')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {self.reconstruction_loss_type}")
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            mu: Mean tensor [batch_size, latent_dim]
            logvar: Log variance tensor [batch_size, latent_dim]
            
        Returns:
            KL divergence loss tensor
        """
        # KL divergence between q(z|x) and p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def _rbf_kernel(self, x, y, gamma):
        """Compute RBF kernel between x and y."""
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-gamma * dist_sq)

    def _mmd_loss(self, x, y, gamma):
        """Compute MMD loss between x and y."""
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=x.device)
        
        xx = self._rbf_kernel(x, x, gamma)
        yy = self._rbf_kernel(y, y, gamma)
        xy = self._rbf_kernel(x, y, gamma)

        # Use biased MMD estimator
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd

    def compute_discrepancy_loss(self, control_z: torch.Tensor, perturbed_z: torch.Tensor,
                               perturbation_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute discrepancy loss between control and perturbed conditions.
        
        Args:
            control_z: Latent representations of control cells [n_control, latent_dim]
            perturbed_z: Latent representations of perturbed cells [n_perturbed, latent_dim]
            perturbation_labels: Labels indicating perturbation type [n_perturbed]
            
        Returns:
            Discrepancy loss tensor
        """
        unique_perturbations = torch.unique(perturbation_labels)
        n_perturbations = len(unique_perturbations)

        if n_perturbations == 0 or control_z.size(0) == 0:
            return torch.tensor(0.0, device=control_z.device)

        discrepancy_loss = 0.0
        
        for pert_label in unique_perturbations:
            pert_mask = perturbation_labels == pert_label
            pert_z = perturbed_z[pert_mask]
            
            if pert_z.size(0) > 0:
                if self.discrepancy_loss_type == 'l2_mean':
                    control_mean = torch.mean(control_z, dim=0, keepdim=True)
                    pert_mean = torch.mean(pert_z, dim=0, keepdim=True)
                    discrepancy = torch.sum((pert_mean - control_mean) ** 2)
                    discrepancy_loss += discrepancy
                elif self.discrepancy_loss_type == 'mmd':
                    # MMD loss is maximized, so we negate it for minimization
                    discrepancy = -self._mmd_loss(control_z, pert_z, self.mmd_gamma)
                    discrepancy_loss += discrepancy
                else:
                    raise ValueError(f"Unsupported discrepancy loss type: {self.discrepancy_loss_type}")

        return discrepancy_loss / n_perturbations
    
    def compute_loss(self, data: Any, model_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for the model.
        
        Args:
            data: PyTorch Geometric Batch object
            model_output: Dictionary containing model outputs
            
        Returns:
            Dictionary containing loss components
        """
        x_recon = model_output['x_recon']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        
        # Original input needs to be reshaped to match decoder output for loss calculation
        x_dense, _ = to_dense_batch(data.x, data.batch)
        x_dense = x_dense.squeeze(-1) # Shape: [batch_size, num_genes]

        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(x_dense, x_recon)
        
        # KL divergence loss
        kl_loss = self.compute_kl_loss(mu, logvar)
        
        # Graph regularization is now implicit in the GCN encoder.
        graph_reg_loss = torch.tensor(0.0, device=x_dense.device)

        # Discrepancy loss
        discrepancy_loss = torch.tensor(0.0, device=x_dense.device)

        is_control = data.is_control.bool()
        control_mask = is_control
        perturbed_mask = ~control_mask

        if torch.sum(control_mask) > 0 and torch.sum(perturbed_mask) > 0:
            control_z = z[control_mask]
            perturbed_z = z[perturbed_mask]
            
            # This needs to be adapted based on how perturbation info is stored
            perturbation_labels = getattr(data, 'perturbation', None)
            if perturbation_labels is not None:
                # This assumes perturbation is a numeric label. If it's a string, it needs mapping.
                # For now, we'll assume it can be handled.
                pert_labels_numeric = pd.Categorical(np.array(perturbation_labels)[perturbed_mask]).codes
                perturbed_labels_tensor = torch.tensor(pert_labels_numeric, device=z.device)

                discrepancy_loss = self.compute_discrepancy_loss(
                    control_z, perturbed_z, perturbed_labels_tensor
                )
        
        # Total loss
        total_loss = (
                      recon_loss +
                      self.beta * kl_loss +
                      self.discrepancy_weight * discrepancy_loss
                     )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'discrepancy_loss': discrepancy_loss,
            'graph_reg_loss': graph_reg_loss # Keep for logging consistency
        }
    
    def get_latent_representation(self, data: Any) -> torch.Tensor:
        """
        Get latent representation for input data.
        
        Args:
            data: PyG Batch object
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            mu, _, _ = self.encode(x, edge_index, batch)
            # Use mean for deterministic representation
            return mu
    
    def generate_samples(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples [n_samples, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim, device=device)
            
            # Decode
            samples = self.decode(z)
            
            return samples
    
    def save_checkpoint(self, filepath: Union[str, Path], epoch: int, optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None, metrics: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Training metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'input_dim': self.input_dim,
            'model_class': self.__class__.__name__
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath.name}")
    
    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: torch.device = None) -> Tuple['DiscrepancyVAE', Dict]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model
        model = cls(
            input_dim=checkpoint['input_dim'],
            config=checkpoint['model_config']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        
        # Extract checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict')
        }
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return model, checkpoint_info


def get_model_summary(model: DiscrepancyVAE) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: DiscrepancyVAE model
        
    Returns:
        Model summary string
    """
    summary_lines = []
    summary_lines.append("DiscrepancyVAE Model Summary")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Input dimension: {model.input_dim}")
    summary_lines.append(f"Latent dimension: {model.latent_dim}")
    summary_lines.append(f"Hidden dimensions: {model.hidden_dims}")
    summary_lines.append(f"Dropout rate: {model.dropout_rate}")
    summary_lines.append(f"Batch normalization: {model.batch_norm}")
    summary_lines.append(f"Activation: {model.activation}")
    summary_lines.append(f"Total parameters: {model._count_parameters():,}")
    summary_lines.append("")
    
    # Encoder summary
    summary_lines.append("Encoder:")
    for name, module in model.encoder.named_children():
        if hasattr(module, '__len__'):
            summary_lines.append(f"  {name}: {len(module)} layers")
        else:
            summary_lines.append(f"  {name}: {module}")
    
    summary_lines.append("")
    
    # Decoder summary
    summary_lines.append("Decoder:")
    for name, module in model.decoder.named_children():
        if hasattr(module, '__len__'):
            summary_lines.append(f"  {name}: {len(module)} layers")
        else:
            summary_lines.append(f"  {name}: {module}")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Example configuration
    config = {
        'model_params': {
            'latent_dim': 32,
            'hidden_dims': [512, 256],
            'dropout_rate': 0.1,
            'batch_norm': True,
            'activation': 'relu',
            'output_activation': 'linear'
        },
        'loss_params': {
            'beta': 1.0,
            'discrepancy_weight': 1.0,
            'reconstruction_loss': 'mse',
            'graph_reg_weight': 0.1
        }
    }
    
    # Create model
    input_dim = 2000  # Example: 2000 genes
    adj = torch.ones(input_dim, input_dim) # Dummy adjacency matrix
    model = DiscrepancyVAE(input_dim=input_dim, config=config, adjacency_matrix=adj)
    
    # Print model summary
    print(get_model_summary(model))
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"\nTest forward pass:")
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {output['x_recon'].shape}")
        print(f"Latent shape: {output['z'].shape}")