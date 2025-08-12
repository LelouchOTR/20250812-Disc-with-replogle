#!/usr/bin/env python3
"""Test script for DiscrepancyVAE model."""

import sys
import torch
sys.path.insert(0, 'src')

from models.discrepancy_vae import DiscrepancyVAE, get_model_summary

# Example configuration
config = {
    'model_params': {
        'latent_dim': 32,
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.1,
        'batch_norm': True,
        'activation': 'relu',
        'output_activation': 'linear'
    },
    'loss_params': {
        'beta': 1.0,
        'discrepancy_weight': 1.0,
        'reconstruction_loss': 'mse'
    }
}

# Create model
input_dim = 2000  # Example: 2000 genes
model = DiscrepancyVAE(input_dim=input_dim, config=config)

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

print("\nModel implementation test successful!")
