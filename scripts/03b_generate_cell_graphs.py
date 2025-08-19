#!/usr/bin/env python3
"""
Cell-level graph generation pipeline for single-cell perturbation analysis.

This script loads a processed AnnData object and a static gene-gene graph.
It then constructs a list of PyTorch Geometric Data objects, one for each cell,
where the node features are the cell's gene expression and the graph structure
is derived from the gene-gene graph. This prepares the data for use with
Graph Neural Network models.
"""

import os
import sys
import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import scanpy as sc
import anndata as ad
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed

logger = logging.getLogger(__name__)


class CellGraphGenerationError(Exception):
    """Custom exception for cell graph generation errors."""
    pass


class CellGraphBuilder:
    """
    Builds cell-level graph objects for GNN-based analysis.
    """

    def __init__(self, config: Dict):
        self.config = config

    def load_data(self, processed_adata_path: Path, gene_graph_path: Path) -> Tuple[ad.AnnData, nx.Graph]:
        """Loads the processed AnnData object and the gene graph."""
        logger.info(f"Loading processed AnnData from: {processed_adata_path}")
        if not processed_adata_path.exists():
            raise CellGraphGenerationError(f"AnnData file not found: {processed_adata_path}")
        adata = sc.read_h5ad(processed_adata_path)
        logger.info(f"Loaded AnnData with shape: {adata.shape}")

        logger.info(f"Loading gene graph from: {gene_graph_path}")
        if not gene_graph_path.exists():
            raise CellGraphGenerationError(f"Gene graph file not found: {gene_graph_path}")
        with open(gene_graph_path, 'rb') as f:
            gene_graph = pickle.load(f)
        logger.info(f"Loaded graph with {gene_graph.number_of_nodes()} nodes and {gene_graph.number_of_edges()} edges.")

        return adata, gene_graph

    def create_cell_graphs(self, adata: ad.AnnData, gene_graph: nx.Graph) -> List[Data]:
        """Creates a list of PyG Data objects, one for each cell."""
        logger.info("Creating cell-level graph objects...")

        # Harmonize genes between AnnData and the graph
        adata_genes = set(adata.var_names)
        graph_genes = set(gene_graph.nodes())

        common_genes = sorted(list(adata_genes.intersection(graph_genes)))

        if len(common_genes) == 0:
            raise CellGraphGenerationError("No common genes between AnnData and the gene graph.")

        logger.info(f"Found {len(common_genes)} common genes to build graphs.")

        # Filter adata and graph to common genes
        adata = adata[:, common_genes].copy()
        subgraph = gene_graph.subgraph(common_genes)

        # Convert the common subgraph to a PyG edge_index.
        # This is done once and shared across all cell graphs for efficiency.
        logger.info("Converting base gene graph to PyG edge format...")
        pyg_graph = from_networkx(subgraph, group_node_attrs=['original_id'])
        edge_index = pyg_graph.edge_index
        edge_weight = pyg_graph.weight if 'weight' in pyg_graph else None

        cell_graphs = []
        logger.info(f"Iterating through {adata.n_obs} cells...")
        for i in tqdm(range(adata.n_obs), desc="Building cell graphs"):
            cell_obs = adata.obs.iloc[i]

            # Extract cell's gene expression vector
            # Ensure it's a dense tensor of shape [n_genes, 1]
            cell_expression = torch.tensor(adata.X[i].toarray().flatten(), dtype=torch.float).unsqueeze(1)

            data = Data(
                x=cell_expression,
                edge_index=edge_index,
                edge_attr=edge_weight,
                # Add other relevant info
                cell_id=cell_obs.name,
                perturbation=cell_obs.get('guide_identity', 'unknown'),
                is_control=torch.tensor(cell_obs.get('is_control', -1), dtype=torch.long),
                split=cell_obs.get('split', 'unknown')
            )
            cell_graphs.append(data)

        logger.info(f"Successfully created {len(cell_graphs)} cell graph objects.")
        return cell_graphs

    def save_cell_graphs(self, cell_graphs: List[Data], output_dir: Path) -> None:
        """Saves the list of cell graphs to a file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "cell_graphs.pkl"

        logger.info(f"Saving {len(cell_graphs)} cell graphs to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(cell_graphs, f)
        logger.info("Save complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate cell-level graphs for GNN models.")
    parser.add_argument("--processed_data", type=str, required=True, help="Path to the processed AnnData file (.h5ad).")
    parser.add_argument("--gene_graph", type=str, required=True, help="Path to the static gene-gene graph file (.pkl).")
    parser.add_argument("--config", type=str, default="cell_graph_config", help="Configuration file name.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--log-dir", type=str, required=True, help="Directory to save logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '03b_generate_cell_graphs.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    set_global_seed(args.seed)

    try:
        config = load_config(args.config) if args.config else {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    logger.info("Starting cell-level graph generation pipeline.")

    try:
        builder = CellGraphBuilder(config)

        adata, gene_graph = builder.load_data(Path(args.processed_data), Path(args.gene_graph))

        cell_graphs = builder.create_cell_graphs(adata, gene_graph)

        builder.save_cell_graphs(cell_graphs, Path(args.output_dir))

        logger.info("Cell-level graph generation completed successfully.")
        return 0

    except Exception as e:
        logger.error(f"Cell-level graph generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
