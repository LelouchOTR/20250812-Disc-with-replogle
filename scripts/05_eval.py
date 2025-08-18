#!/usr/bin/env python3
"""
Model evaluation pipeline for DiscrepancyVAE on single-cell perturbation data.

This script loads a trained model and test data, computes various evaluation
metrics, generates visualization plots, and creates a final evaluation report.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats
import torch
import anndata as ad
import scanpy as sc
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import joblib
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union

# Import visualization utilities
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from utils.visualization import LatentSpaceVisualizer

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed
from src.models.discrepancy_vae import DiscrepancyVAE

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


class ModelEvaluator:
    """
    Evaluator class for the DiscrepancyVAE model with enhanced diagnostics.
    """

    def __init__(self, config: dict, output_dir: Path, log_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = LatentSpaceVisualizer(self.plots_dir)
        
        # Set up logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up file logging for diagnostics."""
        log_file = self.log_dir / 'evaluation_diagnostics.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        
        # Initialize instance variables
        self.model = None
        self.adata_test = None
        self.latent_embeddings = None
        self.metrics = {}
        self.perturbation_results = None
        self.adjacency_matrix = None
        self.node_mapping = None
        self.scaler = None
        
        logger.info(f"Initialized ModelEvaluator with output dir: {self.output_dir}")
        logger.info(f"Using device: {self.device}")
    
    def log_diagnostics(self, message: str, level: str = 'info'):
        """Log diagnostic message with consistent formatting."""
        getattr(logger, level.lower())(f"[DIAGNOSTIC] {message}")

    def load_graph(self, graph_dir: Path):
        if not graph_dir or not graph_dir.exists():
            logger.warning("Graph directory not specified or does not exist. Proceeding without graph.")
            self.adjacency_matrix = None
            self.node_mapping = None
            return

        adj_file = graph_dir / 'adjacency_matrix.npz'
        nodes_file = graph_dir / 'node_mapping.json'

        if adj_file.exists() and nodes_file.exists():
            logger.info("Loading gene adjacency graph...")
            self.adjacency_matrix = sparse.load_npz(adj_file)
            with open(nodes_file, 'r') as f:
                self.node_mapping = json.load(f)
            logger.info(f"Loaded adjacency matrix: {self.adjacency_matrix.shape}")
        else:
            logger.warning(f"Graph files not found in {graph_dir}, proceeding without graph")
            self.adjacency_matrix = None
            self.node_mapping = None

    def _harmonize_genes(self):
        if self.adjacency_matrix is None or self.node_mapping is None:
            logger.info("No graph specified, skipping gene harmonization.")
            return

        logger.info("Harmonizing genes between data and graph...")

        graph_genes = set(self.node_mapping.values())
        data_genes = set(self.adata_test.var_names.tolist())

        common_genes = sorted(list(graph_genes & data_genes))
        
        if not common_genes:
            raise EvaluationError("No common genes found between data and graph.")

        data_unique_genes = data_genes - graph_genes
        graph_unique_genes = graph_genes - data_genes
        
        data_drop_pct = (len(data_unique_genes) / len(data_genes)) * 100
        graph_drop_pct = (len(graph_unique_genes) / len(graph_genes)) * 100

        logger.info(f"Initial data genes: {len(data_genes)}")
        logger.info(f"Initial graph genes: {len(graph_genes)}")
        logger.info(f"Found {len(common_genes)} common genes.")
        logger.info(f"{len(data_unique_genes)} genes ({data_drop_pct:.2f}%) are unique to the data and will be dropped.")
        logger.info(f"{len(graph_unique_genes)} genes ({graph_drop_pct:.2f}%) are unique to the graph and will be dropped.")

        if data_drop_pct > 20:
            logger.warning(f"More than 20% of genes were dropped from the data. The gene interaction graph may not be well-matched to this dataset.")

        # Filter AnnData object
        self.adata_test = self.adata_test[:, common_genes].copy()

        # Filter adjacency matrix
        graph_gene_to_idx = {gene: i for i, gene in enumerate(self.node_mapping.values())}
        indices = [graph_gene_to_idx[gene] for gene in common_genes]
        
        self.adjacency_matrix = self.adjacency_matrix[indices, :][:, indices]

        # Update node mapping to reflect new order and filtering
        self.node_mapping = {i: gene for i, gene in enumerate(common_genes)}

        # Convert to torch sparse tensor
        coo = self.adjacency_matrix.tocoo()
        indices_tensor = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values_tensor = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        self.adjacency_matrix = torch.sparse_coo_tensor(indices_tensor, values_tensor, shape).to(self.device)

        logger.info(f"Successfully harmonized data and graph to {self.adata_test.n_vars} common genes.")
        logger.info(f"Final data shape: {self.adata_test.shape}")
        logger.info(f"Final adjacency matrix shape: {self.adjacency_matrix.shape}")

    def load_model_and_data(self, model_path: Path, data_path: Path, graph_dir: Path, scaler_path: Path):
        logger.info(f"Loading test data from {data_path}")
        if not data_path.exists():
            raise EvaluationError(f"Test data file not found: {data_path}")

        self.adata_test = ad.read_h5ad(data_path)
        logger.info(f"Loaded test data: {self.adata_test.shape}")

        if scaler_path and scaler_path.exists():
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        else:
            logger.warning("Scaler file not found. Proceeding without inverse scaling.")
            self.scaler = None

        self.load_graph(graph_dir)
        self._harmonize_genes()

        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise EvaluationError(f"Model file not found: {model_path}")

        self.model, _ = DiscrepancyVAE.load_checkpoint(
            model_path, 
            device=self.device, 
            adjacency_matrix=self.adjacency_matrix
        )
        self.model.eval()

        self.X_data_type = type(self.adata_test.X)

    def compute_latent_embeddings(self):
        """Compute latent embeddings with enhanced diagnostics."""
        self.log_diagnostics("Starting latent embedding computation...")

        if sparse.issparse(self.adata_test.X):
            X_test_np = self.adata_test.X.toarray()
        else:
            X_test_np = self.adata_test.X
        
        # Log input statistics
        self.log_diagnostics(f"Input data stats - Mean: {X_test_np.mean():.4f}, Std: {X_test_np.std():.4f}")
        self.log_diagnostics(f"Input data range - Min: {X_test_np.min():.4f}, Max: {X_test_np.max():.4f}")
        
        X_test = torch.from_numpy(X_test_np).float()
        batch_size = self.config.get('training', {}).get('batch_size', 128)
        eval_dataset = TensorDataset(X_test)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        latent_embeddings_list = []
        self.model.eval()
        
        with torch.no_grad():
            for (batch_X,) in tqdm(eval_loader, desc="Computing latent embeddings"):
                batch_X = batch_X.to(self.device)
                latent_batch = self.model.get_latent_representation(batch_X)
                latent_embeddings_list.append(latent_batch.cpu().numpy())

        self.latent_embeddings = np.vstack(latent_embeddings_list)
        
        # Log latent space statistics
        self.log_diagnostics(f"Latent space stats - Mean: {self.latent_embeddings.mean():.4f}, "
                           f"Std: {self.latent_embeddings.std():.4f}")
        self.log_diagnostics(f"Latent space range - Min: {self.latent_embeddings.min():.4f}, "
                           f"Max: {self.latent_embeddings.max():.4f}")
        
        # Check for numerical issues
        if np.isnan(self.latent_embeddings).any() or np.isinf(self.latent_embeddings).any():
            self.log_diagnostics("WARNING: NaN or Inf values detected in latent embeddings!", 'warning')
        
        # Add to AnnData
        self.adata_test.obsm['X_latent'] = self.latent_embeddings
        
        # Generate diagnostic plots
        self.visualizer.plot_latent_distribution(self.latent_embeddings)
        self.visualizer.plot_latent_correlation(self.latent_embeddings)
        
        self.log_diagnostics(f"Completed latent embedding computation. Shape: {self.latent_embeddings.shape}")
        
        return self.latent_embeddings

    def compute_reconstruction_metrics(self):
        logger.info("Computing reconstruction metrics...")

        if sparse.issparse(self.adata_test.X):
            X_true_scaled_np = self.adata_test.X.toarray()
        else:
            X_true_scaled_np = self.adata_test.X

        X_true_scaled = torch.from_numpy(X_true_scaled_np).float()

        batch_size = self.config.get('training', {}).get('batch_size', 128)
        eval_dataset = TensorDataset(X_true_scaled)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        recon_scaled_list = []
        with torch.no_grad():
            for (batch_X,) in tqdm(eval_loader, desc="Computing reconstructions"):
                batch_X = batch_X.to(self.device)
                model_output = self.model(batch_X)
                recon_scaled_list.append(model_output['x_recon'].cpu().numpy())

        X_recon_scaled = np.vstack(recon_scaled_list)

        if self.scaler:
            logger.info("Inverse transforming true and reconstructed data for metric calculation.")
            X_true_orig = self.scaler.inverse_transform(X_true_scaled_np)
            X_recon_orig = self.scaler.inverse_transform(X_recon_scaled)
        else:
            logger.warning("No scaler found. Calculating MSE on potentially scaled data.")
            X_true_orig = X_true_scaled_np
            X_recon_orig = X_recon_scaled

        mse = mean_squared_error(X_true_orig, X_recon_orig)
        self.metrics['reconstruction_mse'] = mse
        logger.info(f"Reconstruction MSE on original scale: {mse:.4f}")

    def compute_perturbation_metrics(self):
        logger.info("Computing perturbation metrics...")

        control_mask = self.adata_test.obs['is_control']
        control_latent = self.latent_embeddings[control_mask]
        mean_control_latent = np.mean(control_latent, axis=0)

        results = []
        for guide in self.adata_test.obs['guide_identity'].unique():
            if guide in self.adata_test.obs[self.adata_test.obs['is_control']]['guide_identity'].unique():
                continue

            pert_mask = self.adata_test.obs['guide_identity'] == guide
            pert_latent = self.latent_embeddings[pert_mask]
            mean_pert_latent = np.mean(pert_latent, axis=0)

            effect_vector = mean_pert_latent - mean_control_latent
            effect_magnitude = np.linalg.norm(effect_vector)
            
            results.append({'guide': guide, 'magnitude': effect_magnitude})

        self.perturbation_results = pd.DataFrame(results)
        logger.info(f"Computed perturbation effects for {len(self.perturbation_results)} guides")

    def compute_perturbation_significance(self, n_permutations=1000, fdr_alpha=0.05):
        logger.info(f"Computing perturbation significance with {n_permutations} permutations...")

        control_mask = self.adata_test.obs['is_control']
        control_latent = self.latent_embeddings[control_mask]
        
        p_values = []

        for _, row in tqdm(self.perturbation_results.iterrows(), total=len(self.perturbation_results), desc="Permutation testing"):
            guide = row['guide']
            observed_magnitude = row['magnitude']
            
            pert_mask = self.adata_test.obs['guide_identity'] == guide
            pert_latent = self.latent_embeddings[pert_mask]
            
            n_pert = pert_latent.shape[0]
            n_control = control_latent.shape[0]

            # Pool control and perturbed cells for permutation
            # To avoid bias, sample control cells to match size of perturbation group if possible
            if n_control > n_pert:
                pooled_latent = np.vstack([pert_latent, control_latent[np.random.choice(n_control, n_pert, replace=False)]])
            else:
                pooled_latent = np.vstack([pert_latent, control_latent])
            
            null_magnitudes = []
            for _ in range(n_permutations):
                np.random.shuffle(pooled_latent)
                
                perm_pert_latent = pooled_latent[:n_pert]
                perm_control_latent = pooled_latent[n_pert:]

                mean_perm_pert = np.mean(perm_pert_latent, axis=0)
                mean_perm_control = np.mean(perm_control_latent, axis=0)
                
                null_magnitudes.append(np.linalg.norm(mean_perm_pert - mean_perm_control))
            
            p_value = (np.sum(np.array(null_magnitudes) >= observed_magnitude) + 1) / (n_permutations + 1)
            p_values.append(p_value)

        self.perturbation_results['p_value'] = p_values
        
        # FDR correction
        reject, q_values, _, _ = multipletests(p_values, alpha=fdr_alpha, method='fdr_bh')
        self.perturbation_results['q_value'] = q_values
        self.perturbation_results['significant'] = reject
        
        logger.info(f"Found {np.sum(reject)} significant perturbations at alpha={fdr_alpha}")

    def generate_visualization(self, umap_neighbors: int = 30, umap_min_dist: float = 0.5):
        """Generate UMAP visualizations with enhanced diagnostics."""
        self.log_diagnostics("Starting UMAP visualization...")
        
        try:
            # Log UMAP parameters
            self.log_diagnostics(f"UMAP parameters - n_neighbors: {umap_neighbors}, min_dist: {umap_min_dist}")
            
            # Add some basic diagnostics about the data
            n_cells = self.adata_test.shape[0]
            n_controls = sum(self.adata_test.obs['is_control'])
            n_perturbed = n_cells - n_controls
            self.log_diagnostics(f"Total cells: {n_cells} (controls: {n_controls}, perturbed: {n_perturbed})")
            
            # Check if we have perturbation results
            if self.perturbation_results is None:
                self.log_diagnostics("No perturbation results found. Computing basic metrics...")
                self.compute_perturbation_metrics()
            
            # Get UMAP coordinates
            self.log_diagnostics("Computing UMAP...")
            sc.pp.neighbors(self.adata_test, use_rep='X_latent', n_neighbors=umap_neighbors)
            sc.tl.umap(self.adata_test, min_dist=umap_min_dist)
            
            # Add UMAP coordinates to adata for visualization
            umap_coords = self.adata_test.obsm['X_umap']
            umap_x = umap_coords[:, 0]
            umap_y = umap_coords[:, 1]
            
            # Log UMAP stats
            self.log_diagnostics(f"UMAP coordinates - X range: [{umap_x.min():.2f}, {umap_x.max():.2f}], "
                              f"Y range: [{umap_y.min():.2f}, {umap_y.max():.2f}]")
            
            # Get significant guides
            significant_guides = self.perturbation_results[self.perturbation_results['significant']]
            n_sig = len(significant_guides)
            self.log_diagnostics(f"Found {n_sig} significant perturbations")
            
            # Get top guides for visualization
            top_n = min(20, n_sig) if n_sig > 0 else 0
            if top_n > 0:
                top_guides = significant_guides.sort_values(by=['q_value', 'magnitude'], 
                                                         ascending=[True, False]).head(top_n)
                top_guide_names = top_guides['guide'].tolist()
            else:
                self.log_diagnostics("No significant perturbations found for visualization")
                top_guide_names = []
                
            # Plot control vs perturbed cells
            self._plot_control_vs_perturbed(umap_x, umap_y)
            
            # Plot top perturbations if any
            if top_guide_names:
                self._plot_top_perturbations(umap_x, umap_y, top_guide_names, significant_guides)
            
            # Generate additional diagnostic plots
            self._generate_diagnostic_plots(umap_x, umap_y)
            
            self.log_diagnostics("UMAP visualization completed successfully")
            
        except Exception as e:
            self.log_diagnostics(f"Error in generate_visualization: {str(e)}", 'error')
            raise

    def _plot_control_vs_perturbed(self, umap_x, umap_y):
        """Plot control vs perturbed cells."""
        self.log_diagnostics("Generating control vs perturbed visualization...")
        
        try:
            control_mask = self.adata_test.obs['is_control']
            
            plt.figure(figsize=(10, 8))
            
            # Plot control cells
            plt.scatter(
                umap_x[control_mask], 
                umap_y[control_mask],
                color='darkgray',
                s=15,
                alpha=0.6,
                label='Control Cells',
                edgecolors='none'
            )
            
            # Plot perturbed cells
            plt.scatter(
                umap_x[~control_mask],
                umap_y[~control_mask],
                color='lightblue',
                s=25,
                alpha=0.6,
                label='Perturbed Cells',
                edgecolors='none'
            )
            
            plt.title('UMAP: Control vs Perturbed Cells')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            output_path = self.plots_dir / 'umap_control_vs_perturbed.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_diagnostics(f"Saved control vs perturbed plot to {output_path}")
            
        except Exception as e:
            self.log_diagnostics(f"Error in _plot_control_vs_perturbed: {str(e)}", 'error')
            raise
    
    def _plot_top_perturbations(self, umap_x, umap_y, top_guide_names, significant_guides):
        """Plot top perturbations with effect size scaling."""
        self.log_diagnostics("Generating top perturbations visualization...")
        
        try:
            control_mask = self.adata_test.obs['is_control']
            perturbed_mask = ~control_mask
            
            # Calculate point sizes based on effect magnitude
            max_magnitude = significant_guides['magnitude'].max()
            guide_sizes = {
                row['guide']: 20 + 80 * (row['magnitude'] / max_magnitude) 
                for _, row in significant_guides.iterrows()
            }
            
            plt.figure(figsize=(14, 10))
            
            # Plot control cells in background
            plt.scatter(
                umap_x[control_mask], 
                umap_y[control_mask],
                color='lightgray',
                s=10,
                alpha=0.3,
                label='Control Cells',
                edgecolors='none',
                zorder=1
            )
            
            # Plot non-significant perturbed cells
            non_sig_mask = ~self.adata_test.obs['guide_identity'].isin(top_guide_names)
            plt.scatter(
                umap_x[perturbed_mask & non_sig_mask],
                umap_y[perturbed_mask & non_sig_mask],
                color='lightblue',
                s=15,
                alpha=0.5,
                label='Perturbed (Non-Significant)',
                edgecolors='none',
                zorder=2
            )
            
            # Plot significant perturbations with colors
            palette = sns.color_palette("viridis", n_colors=len(top_guide_names))
            guide_colors = {guide: color for guide, color in zip(top_guide_names, palette)}
            
            for i, guide in enumerate(top_guide_names):
                guide_mask = self.adata_test.obs['guide_identity'] == guide
                plt.scatter(
                    umap_x[guide_mask],
                    umap_y[guide_mask],
                    color=guide_colors[guide],
                    s=guide_sizes[guide],
                    alpha=0.9,
                    edgecolors='black',
                    linewidth=1.2,
                    marker='o',
                    label=guide if i < 20 else None,  # Only label top 20 in legend
                    zorder=3
                )
            
            # Add legend for point sizes
            size_values = [0.2, 0.5, 1.0]
            legend_elements = [
                plt.scatter([], [], s=20 + 80 * size_val, color='gray', alpha=0.9, 
                          edgecolors='black', linewidth=1.2, label=label)
                for size_val, label in zip(size_values, ['Small', 'Medium', 'Large'])
            ]
            
            # Add both legends
            plt.legend(title='Perturbation Effect Size', handles=legend_elements, 
                      loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            # Add main title and labels
            plt.title('UMAP: Top Perturbations by Effect Size')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            
            # Adjust layout to prevent cutoff
            plt.tight_layout()
            
            # Save the plot
            output_path = self.plots_dir / 'umap_top_perturbations.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_diagnostics(f"Saved top perturbations plot to {output_path}")
            
        except Exception as e:
            self.log_diagnostics(f"Error in _plot_top_perturbations: {str(e)}", 'error')
            raise
    
    def _generate_diagnostic_plots(self, umap_x, umap_y):
        """Generate additional diagnostic plots."""
        self.log_diagnostics("Generating diagnostic plots...")
        
        try:
            # Plot UMAP by batch if batch information is available
            if 'batch' in self.adata_test.obs.columns:
                plt.figure(figsize=(10, 8))
                sc.pl.umap(self.adata_test, color='batch', show=False, size=20, 
                          title='UMAP by Batch')
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'umap_by_batch.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot UMAP by guide identity
            plt.figure(figsize=(12, 10))
            sc.pl.umap(self.adata_test, color='guide_identity', show=False, 
                      size=30, title='UMAP by Guide Identity')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'umap_by_guide.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot distribution of control and perturbed cells
            plt.figure(figsize=(10, 6))
            sns.kdeplot(x=umap_x, y=umap_y, hue=self.adata_test.obs['is_control'], 
                       levels=5, thresh=0.2, alpha=0.5)
            plt.title('Density Plot: Control vs Perturbed Cells')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'density_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_diagnostics("Completed generating diagnostic plots")
            
        except Exception as e:
            self.log_diagnostics(f"Error in _generate_diagnostic_plots: {str(e)}", 'error')
            # Don't raise, as this is non-critical
    
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        self.log_diagnostics("Generating evaluation report...")
        
        try:
            # Create report directory if it doesn't exist
            report_dir = self.output_dir / 'report'
            report_dir.mkdir(exist_ok=True)
            
            # Save metrics to JSON
            metrics_path = report_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            # Save perturbation results to CSV
            if self.perturbation_results is not None:
                pert_path = report_dir / 'perturbation_results.csv'
                self.perturbation_results.to_csv(pert_path, index=False)
            
            # Create a simple markdown report
            report_path = report_dir / 'evaluation_report.md'
            with open(report_path, 'w') as f:
                f.write("# DiscrepancyVAE Evaluation Report\n\n")
                
                # Model and data info
                f.write("## Model and Data Information\n")
                f.write(f"- **Model Type**: {self.model.__class__.__name__}\n")
                f.write(f"- **Input Dimensions**: {self.adata_test.n_vars} genes\n")
                f.write(f"- **Latent Dimensions**: {self.model.latent_dim}\n")
                f.write(f"- **Number of Cells**: {self.adata_test.n_obs}\n")
                f.write(f"- **Number of Control Cells**: {sum(self.adata_test.obs['is_control'])}\n\n")
                
                # Metrics section
                f.write("## Evaluation Metrics\n")
                if 'reconstruction_mse' in self.metrics:
                    f.write(f"- **Reconstruction MSE**: {self.metrics['reconstruction_mse']:.4f}\n")
                if 'perturbation_effect' in self.metrics:
                    f.write(f"- **Mean Perturbation Effect**: {self.metrics['perturbation_effect']:.4f}\n")
                if 'perturbation_std' in self.metrics:
                    f.write(f"- **Perturbation Effect Std**: {self.metrics['perturbation_std']:.4f}\n\n")
                
                # Significant perturbations
                if self.perturbation_results is not None and 'significant' in self.perturbation_results.columns:
                    sig_pert = sum(self.perturbation_results['significant'])
                    total_pert = len(self.perturbation_results)
                    f.write(f"## Significant Perturbations\n")
                    f.write(f"- **Significant Perturbations**: {sig_pert}/{total_pert} ({sig_pert/total_pert*100:.1f}%)\n\n")
                    
                    if sig_pert > 0:
                        top5 = self.perturbation_results[self.perturbation_results['significant']]\
                            .sort_values('magnitude', ascending=False).head(5)
                        f.write("### Top 5 Perturbations by Effect Size\n")
                        f.write("| Guide | Effect Size | P-value | Q-value |\n")
                        f.write("|-------|-------------|---------|---------|\n")
                        for _, row in top5.iterrows():
                            f.write(f"| {row['guide']} | {row['magnitude']:.4f} | {row['p_value']:.2e} | {row['q_value']:.2e} |\n")
                        f.write("\n")
                
                # Figures section
                f.write("## Figures\n")
                for fig_name in ['umap_control_vs_perturbed.png', 'umap_top_perturbations.png', 
                               'latent_distributions.png', 'latent_correlations.png']:
                    fig_path = self.plots_dir / fig_name
                    if fig_path.exists():
                        rel_path = fig_path.relative_to(self.output_dir)
                        f.write(f"![{fig_name}]({rel_path})\n\n")
                
                # Add timestamp
                from datetime import datetime
                f.write(f"\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            self.log_diagnostics(f"Generated evaluation report at {report_path}")
            return report_path
            
        except Exception as e:
            self.log_diagnostics(f"Error generating report: {str(e)}", 'error')
            raise
        # Generate reconstruction quality plot
        self.log_diagnostics("Generating reconstruction quality plot...")
        try:
            if sparse.issparse(self.adata_test.X):
                X_test_scaled_np = self.adata_test.X.toarray()
            else:
                X_test_scaled_np = self.adata_test.X

            X_test_scaled = torch.from_numpy(X_test_scaled_np).float()

            batch_size = self.config.get('training', {}).get('batch_size', 128)
            eval_dataset = TensorDataset(X_test_scaled)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

            recon_scaled_list = []
            with torch.no_grad():
                for (batch_X,) in tqdm(eval_loader, desc="Generating reconstructions"):
                    batch_X = batch_X.to(self.device)
                    model_output = self.model(batch_X)
                    recon_scaled_list.append(model_output['x_recon'].cpu().numpy())
                    
            # Combine all batches
            X_recon_scaled = np.vstack(recon_scaled_list)
            
            # Plot reconstruction vs original for a random sample of genes
            n_genes_to_plot = min(1000, X_test_scaled_np.shape[1])
            gene_indices = np.random.choice(X_test_scaled_np.shape[1], n_genes_to_plot, replace=False)
            
            plt.figure(figsize=(10, 8))
            for i in gene_indices[:50]:  # Plot first 50 genes for visualization
                plt.scatter(X_test_scaled_np[:, i], X_recon_scaled[:, i], 
                           alpha=0.3, s=5, color='blue')
            
            # Add diagonal line for perfect reconstruction
            min_val = min(X_test_scaled_np.min(), X_recon_scaled.min())
            max_val = max(X_test_scaled_np.max(), X_recon_scaled.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Original Expression (scaled)')
            plt.ylabel('Reconstructed Expression')
            plt.title('Reconstruction Quality')
            plt.tight_layout()
            
            # Save the plot
            recon_plot_path = self.plots_dir / 'reconstruction_quality.png'
            plt.savefig(recon_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_diagnostics(f"Saved reconstruction quality plot to {recon_plot_path}")
            
        except Exception as e:
            self.log_diagnostics(f"Error generating reconstruction quality plot: {str(e)}", 'error')
            # Continue with evaluation even if this fails

        # The reconstruction quality plot is now generated in the try-except block above

        logger.info("Generating perturbation effect plot...")
        pert_effects_df = self.perturbation_results.sort_values('magnitude', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='guide', y='magnitude', data=pert_effects_df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title("Top 20 Perturbation Effects (Latent Space Magnitude)")
        ax.set_ylabel("Effect Magnitude (L2 norm)")
        plt.savefig(self.plots_dir / "perturbation_effects.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_report(self):
        logger.info("Generating evaluation report...")

        report_path = self.output_dir / "evaluation_report.md"
        
        significant_perts = self.perturbation_results[self.perturbation_results['significant']].sort_values(
            by='q_value', ascending=True
        )

        with open(report_path, 'w') as f:
            f.write("# DiscrepancyVAE Evaluation Report\n\n")
            f.write(f"Evaluation completed on: {pd.Timestamp.now().isoformat()}\n\n")
            
            f.write("## Visualization Explanation\n\n")
            f.write("The UMAP plot shows the latent space representation of cells:\n")
            f.write("- **Gray dots**: Control cells (unperturbed)\n")
            f.write("- **Light blue dots**: Perturbed cells (non-significant)\n")
            f.write("- **Colored dots with black outline**: Top 20 significant perturbations\n")
            f.write("  (Point size indicates perturbation effect strength - larger points represent stronger effects)\n\n")

            f.write("## Metrics\n\n")
            f.write(f"- **Reconstruction MSE:** {self.metrics['reconstruction_mse']:.4f}\n")
            f.write(f"- **Number of significant perturbations (FDR < 0.05):** {len(significant_perts)}\n\n")

            f.write("### Top 10 Significant Perturbations\n\n")
            f.write(significant_perts.head(10).to_markdown(index=False))

            f.write("\n\n## Plots\n\n")

            f.write("### UMAP of Latent Space\n\n")
            f.write("![UMAP](plots/umap_enhanced.png)\n\n")

            f.write("### Reconstruction Quality\n\n")
            f.write("![Reconstruction Quality](plots/reconstruction_quality.png)\n\n")

            f.write("### Top 20 Perturbation Effects (by magnitude)\n\n")
            f.write("![Perturbation Effects](plots/perturbation_effects.png)\n\n")

        logger.info(f"Saved evaluation report to {report_path}")
        
        # Save full results table
        results_csv_path = self.output_dir / "perturbation_results.csv"
        self.perturbation_results.sort_values('q_value').to_csv(results_csv_path, index=False)
        logger.info(f"Saved full perturbation results to {results_csv_path}")

    def plot_control_distribution(self):
        """Plot the distribution of control cells in latent space."""
        if 'is_control' not in self.adata_test.obs:
            self.log_diagnostics("No control cells found for plotting", 'warning')
            return
            
        # Plot control vs perturbed
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        sc.pl.umap(self.adata_test, color='is_control', show=False, 
                  title='Control vs Perturbed', frameon=False)
        
        # Plot just controls
        plt.subplot(122)
        controls = self.adata_test[self.adata_test.obs['is_control']].copy()
        sc.pl.umap(controls, color='guide_identity', 
                  title='Controls Only', frameon=False, show=False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'control_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log control stats
        n_controls = np.sum(self.adata_test.obs['is_control'])
        n_total = len(self.adata_test)
        self.log_diagnostics(f"Control cells: {n_controls}/{n_total} ({n_controls/n_total*100:.1f}%)")
    
    def evaluate(self, model_path: Path, data_path: Path, graph_dir: Path, 
                scaler_path: Path, umap_neighbors: int, umap_min_dist: float, 
                n_permutations: int):
        """Run the full evaluation pipeline with enhanced diagnostics."""
        self.log_diagnostics("===== STARTING EVALUATION PIPELINE =====")
        
        try:
            # Log evaluation parameters
            self.log_diagnostics(f"Model path: {model_path}")
            self.log_diagnostics(f"Data path: {data_path}")
            self.log_diagnostics(f"Graph dir: {graph_dir}")
            self.log_diagnostics(f"UMAP params: n_neighbors={umap_neighbors}, min_dist={umap_min_dist}")
            
            # Run evaluation steps
            self.log_diagnostics("Loading model and data...")
            self.load_model_and_data(model_path, data_path, graph_dir, scaler_path)
            
            self.log_diagnostics("Computing latent embeddings...")
            self.compute_latent_embeddings()
            
            self.log_diagnostics("Computing reconstruction metrics...")
            self.compute_reconstruction_metrics()
            
            self.log_diagnostics("Computing perturbation metrics...")
            self.compute_perturbation_metrics()
            
            self.log_diagnostics(f"Computing perturbation significance with {n_permutations} permutations...")
            self.compute_perturbation_significance(n_permutations=n_permutations)
            
            self.log_diagnostics("Generating visualizations...")
            self.generate_visualization(umap_neighbors, umap_min_dist)
            
            self.log_diagnostics("Generating report...")
            self.generate_report()
            
            self.log_diagnostics("===== EVALUATION COMPLETED SUCCESSFULLY =====")
            
        except Exception as e:
            self.log_diagnostics(f"EVALUATION FAILED: {str(e)}", 'error')
            raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiscrepancyVAE model")
    parser.add_argument("--config", type=str, default="pipeline_config", help="Configuration file name")
    parser.add_argument("--model-path", type=str,
                        default="/data/gidb/shared/results/tmp/replogle/models/best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the test data file (h5ad)")
    parser.add_argument("--scaler-path", type=str, help="Path to the fitted StandardScaler (.joblib file)")
    parser.add_argument("--graph-dir", type=str, help="Gene adjacency graph directory")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/evaluation",
                        help="Output directory")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations for significance testing")

    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '05_eval.log'
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    config = load_config(args.config)

    evaluator = ModelEvaluator(config, Path(args.output_dir), Path(args.log_dir), device)
    evaluator.evaluate(
        Path(args.model_path), 
        Path(args.data_path), 
        Path(args.graph_dir) if args.graph_dir else None,
        Path(args.scaler_path) if args.scaler_path else None,
        args.umap_neighbors,
        args.umap_min_dist,
        args.n_permutations
    )


if __name__ == "__main__":
    sys.exit(main())
