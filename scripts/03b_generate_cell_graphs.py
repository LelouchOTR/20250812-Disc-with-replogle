#!/usr/bin/env python3
"""
Cell-specific graph generation pipeline for single-cell perturbation analysis.

This script builds upon the static gene-gene graph by creating a PyTorch Geometric
(PyG) Data object for each individual cell. This aligns our data structure more
closely with the GEARS methodology, where each cell is treated as a separate graph
instance with its own node features (gene expression).

The process is as follows:
1.  Load the processed AnnData object containing the single-cell expression data.
2.  Load the pre-computed static gene adjacency graph (from `03_graphs.py`).
3.  Create a shared `edge_index` from the static graph, representing the common
    biological prior for all cells.
4.  For each cell in the AnnData object, create a `torch_geometric.data.Data` object:
    - `x`: The cell's gene expression profile (node features).
    - `edge_index`: The shared graph structure.
    - Other metadata: Perturbation, cell type, etc., are stored as attributes.
5.  Save the list of these `Data` objects to a file, ready for the training pipeline.
"""

import os
import sys
import logging
import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
import anndata
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed

logger = logging.getLogger(__name__)


class CellGraphGenerationError(Exception):
    """Custom exception for cell-graph generation errors."""
    pass


def load_processed_data(data_path: Path) -> anndata.AnnData:
    """Loads the processed AnnData object."""
    logger.info(f"Loading processed data from: {data_path}")
    if not data_path.exists():
        raise CellGraphGenerationError(f"Processed data file not found: {data_path}")
    return anndata.read_h5ad(data_path)


def load_static_graph(graph_path: Path) -> nx.Graph:
    """Loads the static gene adjacency graph."""
    logger.info(f"Loading static graph from: {graph_path}")
    if not graph_path.exists():
        raise CellGraphGenerationError(f"Static graph file not found: {graph_path}")
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def create_cell_graphs(adata: anndata.AnnData, graph: nx.Graph, config: Dict) -> Tuple[List[Data], Dict]:
    """
    Creates a list of PyG Data objects, one for each cell.
    """
    logger.info("Starting cell-specific graph generation...")

    # 1. Create a mapping from gene IDs in adata to node indices in the graph
    adata_genes = adata.var_names.tolist()

    # Ensure graph nodes are in a consistent order for mapping
    graph_nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(graph_nodes)}

    # Find the intersection of genes and their indices in adata
    common_genes = [gene for gene in adata_genes if gene in node_to_idx]
    if not common_genes:
        raise CellGraphGenerationError("No common genes found between AnnData and the graph.")

    logger.info(f"Found {len(common_genes)} common genes between dataset and graph.")

    # Filter both adata and the graph to only include common genes, preserving order
    adata_common_idx = [adata_genes.index(g) for g in common_genes]
    adata = adata[:, adata_common_idx].copy()

    subgraph = graph.subgraph(common_genes)
    logger.info(f"Subgraph created with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

    # Re-index the subgraph nodes to be 0 to N-1 based on the common_genes order
    subgraph_node_to_idx = {gene: i for i, gene in enumerate(common_genes)}

    edge_list = []
    for u, v in subgraph.edges():
        u_idx, v_idx = subgraph_node_to_idx[u], subgraph_node_to_idx[v]
        edge_list.append((u_idx, v_idx))
        edge_list.append((v_idx, u_idx))

    if not edge_list:
        logger.warning("No edges found in the subgraph of common genes. The graph will be fully disconnected.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 3. Get the gene expression data for the common genes
    if isinstance(adata.X, csr_matrix):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X

    logger.info(f"Expression data shape for common genes: {expression_data.shape}")

    # 4. Create a Data object for each cell
    cell_graphs = []

    pert_categories = adata.obs['condition'].astype('category').cat.categories
    pert_map = {pert: i for i, pert in enumerate(pert_categories)}

    cell_type_categories = adata.obs['cell_type'].astype('category').cat.categories
    cell_type_map = {ct: i for i, ct in enumerate(cell_type_categories)}

    # Get control indices
    control_key = config.get('control_key', 'ctrl')
    control_indices = torch.tensor(
        [i for i, p in enumerate(adata.obs['condition']) if control_key in p],
        dtype=torch.long
    )

    for i in tqdm(range(adata.n_obs), desc="Creating cell graphs"):
        cell_expression = torch.tensor(expression_data[i, :], dtype=torch.float)

        condition = adata.obs['condition'][i]
        cell_type = adata.obs['cell_type'][i]

        split = adata.obs['split'][i]

        data = Data(
            x=cell_expression.unsqueeze(1),
            edge_index=edge_index,
            y=torch.tensor([pert_map[condition]], dtype=torch.long),
            pert=condition,
            cell_type=cell_type,
            is_control=torch.tensor([control_key in condition], dtype=torch.bool),
            split=split
        )
        cell_graphs.append(data)

    logger.info(f"Successfully created {len(cell_graphs)} cell-specific graph objects.")

    mappings = {
        'pert_map': pert_map,
        'cell_type_map': cell_type_map,
        'gene_list': common_genes,
        'control_key': control_key,
        'num_genes': len(common_genes),
        'num_perts': len(pert_map),
        'num_cell_types': len(cell_type_map)
    }

    return cell_graphs, mappings


def save_cell_graphs(cell_graphs: List[Data], mappings: Dict, output_dir: Path):
    """Saves the list of cell graphs and mappings to disk."""
    logger.info(f"Saving cell graphs to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_file = output_dir / "cell_graphs.pkl"
    with open(graph_file, 'wb') as f:
        pickle.dump(cell_graphs, f)
    logger.info(f"Saved {len(cell_graphs)} graphs to {graph_file}")

    mappings_file = output_dir / "mappings.json"
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    logger.info(f"Saved mappings to {mappings_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate cell-specific graphs from single-cell data.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID for locating input files and saving outputs.")
    parser.add_argument("--config", type=str, default="cell_graph_config", help="Configuration file name.")
    parser.add_argument("--base-output-dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Setup directories
    run_dir = Path(args.base_output_dir) / args.run_id
    output_dir = run_dir / "cell_graphs"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = log_dir / '03b_generate_cell_graphs.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    logger.info(f"Starting cell-graph generation for run_id: {args.run_id}")
    set_global_seed(args.seed)

    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")

        processed_data_path = run_dir / "processed" / "processed_data.h5ad"
        static_graph_path = run_dir / "graphs" / "gene_adjacency_graph.pkl"

        adata = load_processed_data(processed_data_path)
        static_graph = load_static_graph(static_graph_path)

        cell_graphs, mappings = create_cell_graphs(adata, static_graph, config)

        save_cell_graphs(cell_graphs, mappings, output_dir)

        logger.info("Cell-specific graph generation completed successfully.")
        return 0

    except (CellGraphGenerationError, FileNotFoundError) as e:
        logger.error(f"A critical error occurred: {e}")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred in the pipeline.", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
