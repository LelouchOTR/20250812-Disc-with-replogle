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
            input_dim: Input dimension (number of genes)
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
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv1 = GCNConv(1, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph encoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            edge_index: Edge index tensor for graph [2, num_edges]
            
        Returns:
            Tuple of (mu, logvar, gene_embeddings) tensors
        """
        batch_size, num_genes = x.shape

        # Process each sample in the batch individually to avoid large tensors
        batch_mu = []
        batch_logvar = []
        batch_gene_embeddings = []

        for i in range(batch_size):
            # Get the features for the current sample (cell)
            x_sample = x[i].unsqueeze(-1)  # Shape: [num_genes, 1]

            # Apply graph convolutions
            h = self.conv1(x_sample, edge_index)
            h = self.activation(h)
            h = self.conv2(h, edge_index)
            gene_embeddings = self.activation(h)  # Shape: [num_genes, hidden_dims[1]]

            # Pool features across genes
            pooled_output = torch.mean(gene_embeddings, dim=0) # Shape: [hidden_dims[1]]

            # Latent space parameters
            mu = self.mu_layer(pooled_output)
            logvar = self.logvar_layer(pooled_output)

            batch_mu.append(mu)
            batch_logvar.append(logvar)
            batch_gene_embeddings.append(gene_embeddings)

        # Stack results into tensors
        mu = torch.stack(batch_mu)
        logvar = torch.stack(batch_logvar)
        gene_embeddings_reshaped = torch.stack(batch_gene_embeddings)

        return mu, logvar, gene_embeddings_reshaped


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
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Choose output activation
        if output_activation == 'linear':
            self.output_activation = nn.Identity()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softplus':
            self.output_activation = nn.Softplus()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        
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
    
    def __init__(self, input_dim: int, config: Dict[str, Any], adjacency_matrix: Optional[torch.Tensor] = None):
        """
        Initialize DiscrepancyVAE model.
        
        Args:
            input_dim: Input dimension (number of genes)
            config: Model configuration dictionary
            adjacency_matrix: Optional adjacency matrix for graph-based regularization
        """
        super(DiscrepancyVAE, self).__init__()
        
        self.input_dim = input_dim
        self.config = config
        self.adjacency_matrix = adjacency_matrix
        
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
        self.graph_reg_weight = loss_params.get('graph_reg_weight', 0.1)

        # Build encoder and decoder
        self.encoder = GraphEncoder(
            input_dim=input_dim,
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
        
        if self.adjacency_matrix is not None:
            if self.adjacency_matrix.is_sparse:
                edge_index = self.adjacency_matrix._indices()
            else:
                # For backward compatibility with dense matrices
                edge_index = self.adjacency_matrix.to_sparse()._indices()
        else:
            # If no graph, create a fully connected graph as a placeholder
            logger.warning("Adjacency matrix not provided. Using a fully connected graph for GCN.")
            adj = torch.ones(input_dim, input_dim)
            edge_index = adj.to_sparse()._indices()

        self.register_buffer('edge_index', edge_index)

        logger.info(f"Initialized DiscrepancyVAE with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mu, logvar, gene_embeddings) tensors
        """
        mu, logvar, h_batch = self.encoder(x, self.edge_index)
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode
        mu, logvar, gene_embeddings = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
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
            return F.mse_loss(x_recon, x, reduction='sum')
        elif self.reconstruction_loss_type == 'bce':
            return F.binary_cross_entropy(x_recon, x, reduction='sum')
        elif self.reconstruction_loss_type == 'poisson':
            # Poisson loss for count data
            return torch.sum(x_recon - x * torch.log(x_recon + 1e-8))
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
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def compute_graph_laplacian_regularization(self, z_genes: torch.Tensor) -> torch.Tensor:
        """
        Computes the graph Laplacian regularization loss for a batch.
        This loss encourages connected nodes in the graph to have similar latent representations.
        z_genes shape: [batch_size, num_nodes, num_features]
        """
        if self.adjacency_matrix is None:
            return torch.tensor(0.0, device=z_genes.device)

        edge_index = self.edge_index
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        batch_size = z_genes.shape[0]
        total_graph_reg = 0.0
        
        edge_chunk_size = 100000 # Process 100k edges at a time

        for i in range(batch_size):
            z_sample = z_genes[i] # [num_nodes, num_features]
            
            for j in range(0, num_edges, edge_chunk_size):
                chunk_end = min(j + edge_chunk_size, num_edges)
                src_chunk = src_nodes[j:chunk_end]
                dst_chunk = dst_nodes[j:chunk_end]

                z_src = z_sample[src_chunk, :] # [chunk_size, num_features]
                z_dst = z_sample[dst_chunk, :] # [chunk_size, num_features]
                
                dist_sq = torch.sum((z_src - z_dst)**2, dim=1) # [chunk_size]
                total_graph_reg += torch.sum(dist_sq)

        graph_reg = total_graph_reg / batch_size
        
        return graph_reg

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
        # Compute mean control representation
        control_mean = torch.mean(control_z, dim=0, keepdim=True)  # [1, latent_dim]
        
        # Compute discrepancy for each perturbation
        discrepancy_loss = 0.0
        unique_perturbations = torch.unique(perturbation_labels)
        
        for pert_label in unique_perturbations:
            # Get cells with this perturbation
            pert_mask = perturbation_labels == pert_label
            pert_z = perturbed_z[pert_mask]  # [n_pert_cells, latent_dim]
            
            if pert_z.size(0) > 0:
                # Compute mean perturbation representation
                pert_mean = torch.mean(pert_z, dim=0, keepdim=True)  # [1, latent_dim]
                
                # Compute discrepancy (L2 distance)
                discrepancy = torch.sum((pert_mean - control_mean) ** 2)
                discrepancy_loss += discrepancy
        
        return discrepancy_loss
    
    def compute_loss(self, x: torch.Tensor, model_output: Dict[str, torch.Tensor],
                    is_control: torch.Tensor, perturbation_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for the model.
        
        Args:
            x: Original input [batch_size, input_dim]
            model_output: Dictionary containing model outputs
            is_control: Boolean tensor indicating control cells [batch_size]
            perturbation_labels: Perturbation labels for non-control cells [batch_size]
            
        Returns:
            Dictionary containing loss components
        """
        x_recon = model_output['x_recon']
        mu = model_output['mu']
        logvar = model_output['logvar']
        z = model_output['z']
        gene_embeddings = model_output['gene_embeddings']
        
        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(x, x_recon)
        
        # KL divergence loss
        kl_loss = self.compute_kl_loss(mu, logvar)
        
        # Graph Laplacian Regularization
        graph_reg_loss = self.compute_graph_laplacian_regularization(gene_embeddings)

        # Discrepancy loss
        discrepancy_loss = torch.tensor(0.0, device=x.device)
        if perturbation_labels is not None:
            control_mask = is_control.bool()
            perturbed_mask = ~control_mask
            
            if torch.sum(control_mask) > 0 and torch.sum(perturbed_mask) > 0:
                control_z = z[control_mask]
                perturbed_z = z[perturbed_mask]
                perturbed_labels = perturbation_labels[perturbed_mask]
                
                discrepancy_loss = self.compute_discrepancy_loss(
                    control_z, perturbed_z, perturbed_labels
                )
        
        # Total loss
        total_loss = (
                      recon_loss +
                      self.beta * kl_loss +
                      self.discrepancy_weight * discrepancy_loss +
                      self.graph_reg_weight * graph_reg_loss
                     )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'discrepancy_loss': discrepancy_loss,
            'graph_reg_loss': graph_reg_loss
        }
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for input data.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
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


class SingleCellDataset(Dataset):
    """
    Dataset class for single-cell data.
    """
    
    def __init__(self, adata: ad.AnnData, use_raw: bool = False):
        """
        Initialize dataset.
        
        Args:
            adata: AnnData object containing single-cell data
            use_raw: Whether to use raw counts or processed data
        """
        self.adata = adata
        self.use_raw = use_raw
        
        # Get expression data
        if use_raw and adata.raw is not None:
            self.X = adata.raw.X
        else:
            self.X = adata.X
        
        # Convert to dense if sparse
        if sparse.issparse(self.X):
            self.X = self.X.toarray()
        
        # Convert to float32
        self.X = self.X.astype(np.float32)
        
        # Get metadata
        self.obs = adata.obs.copy()
        
        # Extract perturbation information
        self.is_control = self._extract_control_labels()
        self.perturbation_labels = self._extract_perturbation_labels()
    
    def _extract_control_labels(self) -> np.ndarray:
        """Extract control cell labels."""
        if 'is_control' in self.obs.columns:
            return self.obs['is_control'].values.astype(bool)
        elif 'guide_identity' in self.obs.columns:
            # Assume control guides contain 'control' or 'non-targeting'
            guide_ids = self.obs['guide_identity'].astype(str).str.lower()
            is_control = guide_ids.str.contains('control|non-targeting|negative')
            return is_control.values
        else:
            # Default: all cells are control
            logger.warning("No control information found, assuming all cells are control")
            return np.ones(len(self.obs), dtype=bool)
    
    def _extract_perturbation_labels(self) -> np.ndarray:
        """Extract perturbation labels."""
        if 'perturbation_label' in self.obs.columns:
            # Convert to categorical codes
            labels = pd.Categorical(self.obs['perturbation_label'])
            return labels.codes.astype(np.int64)
        elif 'guide_identity' in self.obs.columns:
            # Use guide identity as perturbation label
            labels = pd.Categorical(self.obs['guide_identity'])
            return labels.codes.astype(np.int64)
        else:
            # Default: all cells have same perturbation
            return np.zeros(len(self.obs), dtype=np.int64)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing data tensors
        """
        return {
            'x': torch.from_numpy(self.X[idx]),
            'is_control': torch.tensor(self.is_control[idx], dtype=torch.bool),
            'perturbation_label': torch.tensor(self.perturbation_labels[idx], dtype=torch.long)
        }


def create_data_loaders(adata_train: ad.AnnData, adata_val: ad.AnnData,
                       batch_size: int = 128, num_workers: int = 4,
                       use_raw: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        adata_train: Training AnnData
        adata_val: Validation AnnData
        batch_size: Batch size
        num_workers: Number of worker processes
        use_raw: Whether to use raw counts
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SingleCellDataset(adata_train, use_raw=use_raw)
    val_dataset = SingleCellDataset(adata_val, use_raw=use_raw)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


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