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
from scipy import sparse
import torch
import anndata as ad
import scanpy as sc
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    Evaluator class for the DiscrepancyVAE model.
    """

    def __init__(self, config: dict, output_dir: Path, log_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device
        self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.adata_test = None
        self.latent_embeddings = None
        self.metrics = {}

        logger.info(f"Initialized ModelEvaluator with output dir: {self.output_dir}")

    def load_model_and_data(self, model_path: Path, data_path: Path):
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise EvaluationError(f"Model file not found: {model_path}")

        self.model, _ = DiscrepancyVAE.load_checkpoint(model_path, device=self.device)
        self.model.eval()

        logger.info(f"Loading test data from {data_path}")
        if not data_path.exists():
            raise EvaluationError(f"Test data file not found: {data_path}")

        self.adata_test = ad.read_h5ad(data_path)
        logger.info(f"Loaded test data: {self.adata_test.shape}")
        self.X_data_type = type(self.adata_test.X)

    def compute_latent_embeddings(self):
        logger.info("Computing latent embeddings...")

        if sparse.issparse(self.adata_test.X):
            X_test = torch.from_numpy(self.adata_test.X.toarray())
        else:
            X_test = torch.from_numpy(self.adata_test.X)
        X_test = X_test.float().to(self.device)

        with torch.no_grad():
            self.latent_embeddings = self.model.get_latent_representation(X_test).cpu().numpy()

        self.adata_test.obsm['X_latent'] = self.latent_embeddings
        logger.info(f"Computed latent embeddings: {self.latent_embeddings.shape}")

    def compute_reconstruction_metrics(self):
        logger.info("Computing reconstruction metrics...")

        if sparse.issparse(self.adata_test.X):
            X_true = self.adata_test.X.toarray()
        else:
            X_true = self.adata_test.X

        X_test = torch.from_numpy(X_true).float().to(self.device)

        with torch.no_grad():
            model_output = self.model(X_test)
            X_recon = model_output['x_recon'].cpu().numpy()

        mse = mean_squared_error(X_true, X_recon)
        self.metrics['reconstruction_mse'] = mse
        logger.info(f"Reconstruction MSE: {mse:.4f}")

    def compute_perturbation_metrics(self):
        logger.info("Computing perturbation metrics...")

        control_mask = self.adata_test.obs['is_control']
        control_latent = self.latent_embeddings[control_mask]
        mean_control_latent = np.mean(control_latent, axis=0)

        perturbation_effects = {}
        for guide in self.adata_test.obs['guide_identity'].unique():
            if guide in self.adata_test.obs[self.adata_test.obs['is_control']]['guide_identity'].unique():
                continue

            pert_mask = self.adata_test.obs['guide_identity'] == guide
            pert_latent = self.latent_embeddings[pert_mask]
            mean_pert_latent = np.mean(pert_latent, axis=0)

            effect_vector = mean_pert_latent - mean_control_latent
            effect_magnitude = np.linalg.norm(effect_vector)
            perturbation_effects[guide] = effect_magnitude

        self.metrics['perturbation_effects'] = perturbation_effects
        logger.info(f"Computed perturbation effects for {len(perturbation_effects)} guides")

    def generate_visualization(self):
        logger.info("Generating visualization...")

        logger.info("Generating enhanced UMAP visualization...")
        sc.pp.neighbors(self.adata_test, use_rep='X_latent', n_neighbors=15)
        sc.tl.umap(self.adata_test)

        umap_x = self.adata_test.obsm['X_umap'][:, 0]
        umap_y = self.adata_test.obsm['X_umap'][:, 1]
        
        top_20_guides = sorted(
            self.metrics['perturbation_effects'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        max_magnitude = max(m[1] for m in top_20_guides)
        guide_sizes = {g[0]: 20 + 80 * (g[1] / max_magnitude) 
                       for g in top_20_guides}
        
        fig, ax = plt.subplots(figsize=(12, 10))

        control_mask = self.adata_test.obs['is_control']
        
        ax.scatter(
            umap_x[control_mask], 
            umap_y[control_mask],
            color='darkblue',
            s=15,
            alpha=0.7,
            label='Control Cells',
            edgecolors='none'
        )

        perturbed_mask = ~control_mask
        ax.scatter(
            umap_x[perturbed_mask],
            umap_y[perturbed_mask],
            color='lightcoral',
            s=25,
            alpha=0.6,
            label='Perturbed Cells',
            edgecolors='none'
        )

        for guide, magnitude in top_20_guides:
            guide_mask = (self.adata_test.obs['guide_identity'] == guide)
            size = guide_sizes[guide]
            ax.scatter(
                umap_x[guide_mask],
                umap_y[guide_mask],
                color='red',
                s=size,
                alpha=0.9,
                edgecolors='black',
                linewidth=1.2,
                marker='o'
            )

        size_values = [0.1, 0.5, 1.0]
        size_labels = ['Low Effect', 'Medium Effect', 'High Effect']
        legend_elements = [
            plt.scatter([], [], s=20 + 80 * size_val, color='red', alpha=0.9, edgecolors='black', linewidth=1.2)
            for size_val in size_values
        ]
        
        size_legend = ax.legend(
            handles=legend_elements,
            labels=size_labels,
            title="Perturbation Effect Magnitude",
            loc='upper right',
            fontsize=9,
            title_fontsize=10
        )
        
        main_legend = ax.legend(
            ['Control Cells', 'Perturbed Cells', 'Top Perturbations'],
            ['Control (Dark Blue)', 'Perturbed (Light Red)', 'Top 20 (Red with Black Outline)'],
            loc='upper left',
            fontsize=9
        )
        
        ax.add_artist(size_legend)

        ax.set_title("UMAP: Control vs Perturbed Cells\n(Point Size Indicates Perturbation Effect Strength)", fontsize=14)
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "umap_enhanced.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info("Generating reconstruction quality plot...")
        if sparse.issparse(self.adata_test.X):
            X_test = self.adata_test.X.toarray()
        else:
            X_test = self.adata_test.X
        with torch.no_grad():
            X_recon = self.model(torch.from_numpy(X_test).float().to(self.device))['x_recon'].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(X_test.flatten(), X_recon.flatten(), alpha=0.1, s=1)
        ax.set_xlabel("Original Expression")
        ax.set_ylabel("Reconstructed Expression")
        ax.set_title("Reconstruction Quality")
        plt.savefig(self.plots_dir / "reconstruction_quality.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info("Generating perturbation effect plot...")
        pert_effects_df = pd.DataFrame.from_dict(self.metrics['perturbation_effects'], orient='index',
                                                 columns=['magnitude'])
        pert_effects_df = pert_effects_df.sort_values('magnitude', ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=pert_effects_df.index, y=pert_effects_df['magnitude'], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title("Top 20 Perturbation Effects (Latent Space Magnitude)")
        ax.set_ylabel("Effect Magnitude (L2 norm)")
        plt.savefig(self.plots_dir / "perturbation_effects.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_report(self):
        logger.info("Generating evaluation report...")

        report_path = self.output_dir / "evaluation_report.md"

        with open(report_path, 'w') as f:
            f.write("# DiscrepancyVAE Evaluation Report\n\n")
            f.write(f"Evaluation completed on: {pd.Timestamp.now().isoformat()}\n\n")

            f.write("## Metrics\n\n")
            f.write(f"- **Reconstruction MSE:** {self.metrics['reconstruction_mse']:.4f}\n")

            top_5_perts = sorted(self.metrics['perturbation_effects'].items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("- **Top 5 Perturbation Effects:**\n")
            for guide, mag in top_5_perts:
                f.write(f"  - {guide}: {mag:.4f}\n")

            f.write("\n## Visualization\n\n")

            f.write("### UMAP of Latent Space\n\n")
            f.write("![UMAP](plots/umap_enhanced.png)\n\n")

            f.write("### Reconstruction Quality\n\n")
            f.write("![Reconstruction Quality](plots/reconstruction_quality.png)\n\n")

            f.write("### Perturbation Effects\n\n")
            f.write("![Perturbation Effects](plots/perturbation_effects.png)\n\n")

        logger.info(f"Saved evaluation report to {report_path}")

    def evaluate(self, model_path: Path, data_path: Path):
        logger.info("Starting model evaluation...")
        self.load_model_and_data(model_path, data_path)
        self.compute_latent_embeddings()
        self.compute_reconstruction_metrics()
        self.compute_perturbation_metrics()
        self.generate_visualization()
        self.generate_report()
        logger.info("Evaluation complete.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiscrepancyVAE model")
    parser.add_argument("--config", type=str, default="pipeline_config", help="Configuration file name")
    parser.add_argument("--model-path", type=str,
                        default="/data/gidb/shared/results/tmp/replogle/models/best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the test data file (h5ad)")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/evaluation",
                        help="Output directory")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
    evaluator.evaluate(Path(args.model_path), Path(args.data_path))


if __name__ == "__main__":
    sys.exit(main())
