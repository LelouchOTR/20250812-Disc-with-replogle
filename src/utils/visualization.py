"""
Visualization utilities for model diagnostics and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
import torch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)

class LatentSpaceVisualizer:
    """Class for visualizing and diagnosing latent space issues."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_latent_distribution(self, latent_embeddings: np.ndarray, 
                              prefix: str = ""):
        """Plot distribution of latent dimensions."""
        plt.figure(figsize=(15, 5))
        
        # Plot mean and std of each latent dimension
        latent_means = np.mean(latent_embeddings, axis=0)
        latent_stds = np.std(latent_embeddings, axis=0)
        
        dims = np.arange(latent_embeddings.shape[1])
        
        plt.subplot(121)
        plt.bar(dims, latent_means)
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.title('Mean of Latent Dimensions')
        plt.xlabel('Dimension')
        plt.ylabel('Mean')
        
        plt.subplot(122)
        plt.bar(dims, latent_stds)
        plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Ideal')
        plt.title('Std Dev of Latent Dimensions')
        plt.xlabel('Dimension')
        plt.ylabel('Standard Deviation')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}latent_distributions.png")
        plt.close()
        
        return latent_means, latent_stds
    
    def plot_latent_correlation(self, latent_embeddings: np.ndarray,
                             prefix: str = ""):
        """Plot correlation matrix of latent dimensions."""
        corr = np.corrcoef(latent_embeddings, rowvar=False)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, square=True)
        plt.title('Latent Dimension Correlations')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}latent_correlations.png")
        plt.close()
        
        return corr
    
    def plot_perturbation_effects(self, results_df: pd.DataFrame,
                               top_n: int = 20,
                               prefix: str = ""):
        """Plot effect sizes of top perturbations."""
        # Sort by absolute effect size
        df = results_df.copy()
        df['abs_magnitude'] = df['magnitude'].abs()
        df = df.sort_values('abs_magnitude', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df.head(top_n), x='magnitude', y='guide',
                   palette='viridis')
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.title(f'Top {top_n} Perturbation Effects')
        plt.xlabel('Effect Magnitude')
        plt.ylabel('Guide')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}perturbation_effects.png")
        plt.close()
    
    def plot_umap_quality(self, adata, color_by: str = 'guide_identity',
                        prefix: str = ""):
        """Generate UMAP quality control plots."""
        # Plot UMAP by guide
        plt.figure(figsize=(12, 10))
        sc.pl.umap(adata, color=color_by, show=False, size=50, 
                  title=f'UMAP by {color_by}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}umap_by_{color_by}.png")
        plt.close()
        
        # Plot UMAP by control vs perturbed
        adata.obs['is_perturbed'] = ~adata.obs['is_control']
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        sc.pl.umap(adata, color='is_perturbed', show=False, 
                  title='Control vs Perturbed', size=50)
        
        # Plot just controls
        plt.subplot(122)
        sc.pl.umap(adata[adata.obs['is_control']], 
                  color='guide_identity', show=False,
                  title='Controls Only', size=100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}umap_controls_vs_perturbed.png")
        plt.close()
        
        return adata
