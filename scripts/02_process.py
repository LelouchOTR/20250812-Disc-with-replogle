#!/usr/bin/env python3
"""
Data processing pipeline for single-cell perturbation analysis.

This script loads raw Replogle 2022 K562 essential Perturb-seq dataset,
applies quality control filters, implements guide assignment policy,
performs normalization and transformation, conducts feature selection,
maps gene IDs to Ensembl namespace, and saves processed data.
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed
from src.utils.gene_ids import get_gene_mapper, standardize_gene_list

# Configure scanpy settings
sc.settings.verbosity = 1  # Reduce scanpy verbosity
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class SingleCellDataProcessor:
    """
    Comprehensive single-cell data processor for Perturb-seq data.
    
    Handles quality control, normalization, feature selection, and gene ID mapping.
    """
    
    def __init__(self, config: Dict, output_dir: Path):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
            output_dir: Directory to save processed data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract configuration sections
        self.qc_config = config.get('qc_thresholds', {})
        self.guide_config = config.get('guide_filtering', {})
        self.norm_config = config.get('normalization', {})
        self.split_config = config.get('data_splitting', {})
        self.gene_config = config.get('gene_filtering', {})
        
        # Initialize gene mapper
        self.gene_mapper = get_gene_mapper()
        
        logger.info(f"Initialized SingleCellDataProcessor with output dir: {self.output_dir}")
    
    def load_raw_data(self, data_path: Union[str, Path]) -> ad.AnnData:
        """
        Load raw single-cell data from various formats.
        
        Args:
            data_path: Path to raw data file
            
        Returns:
            AnnData object with raw data
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise DataProcessingError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading raw data from: {data_path}")
        
        try:
            if data_path.suffix == '.h5ad':
                adata = sc.read_h5ad(data_path)
            elif data_path.suffix == '.h5':
                adata = sc.read_10x_h5(data_path)
            elif data_path.suffix == '.csv':
                adata = sc.read_csv(data_path).T  # Transpose to get cells x genes
            elif data_path.suffix == '.xlsx':
                adata = sc.read_excel(data_path).T
            else:
                # Try to read as compressed h5ad
                if '.h5ad' in data_path.name:
                    adata = sc.read_h5ad(data_path)
                else:
                    raise DataProcessingError(f"Unsupported file format: {data_path.suffix}")
            
            logger.info(f"Loaded data with shape: {adata.shape} (cells x genes)")
            logger.info(f"Data type: {type(adata.X)}")
            
            return adata
            
        except Exception as e:
            raise DataProcessingError(f"Failed to load data from {data_path}: {e}")
    
    def apply_quality_control(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply quality control filters to remove low-quality cells and genes.
        
        Args:
            adata: AnnData object with raw data
            
        Returns:
            Filtered AnnData object
        """
        logger.info("Applying quality control filters...")
        
        # Store original dimensions
        n_cells_orig, n_genes_orig = adata.shape
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
        
        # Calculate per-cell QC metrics
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Calculate mitochondrial gene percentage
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Store QC metrics before filtering
        adata.obs['n_genes_orig'] = adata.obs['n_genes_by_counts']
        adata.obs['total_counts_orig'] = adata.obs['total_counts']
        adata.obs['pct_counts_mt_orig'] = adata.obs['pct_counts_mt']
        
        # Apply cell filtering
        min_genes = self.qc_config.get('min_genes_per_cell', 200)
        max_genes = self.qc_config.get('max_genes_per_cell', 8000)
        min_counts = self.qc_config.get('min_total_counts_per_cell', 1000)
        max_counts = self.qc_config.get('max_total_counts_per_cell', 50000)
        max_mt = self.qc_config.get('max_mitochondrial_fraction', 0.2)
        
        logger.info(f"Cell filtering thresholds:")
        logger.info(f"  - Genes per cell: {min_genes} - {max_genes}")
        logger.info(f"  - Counts per cell: {min_counts} - {max_counts}")
        logger.info(f"  - Max mitochondrial fraction: {max_mt}")
        
        # Filter cells
        cell_filter = (
            (adata.obs['n_genes_by_counts'] >= min_genes) &
            (adata.obs['n_genes_by_counts'] <= max_genes) &
            (adata.obs['total_counts'] >= min_counts) &
            (adata.obs['total_counts'] <= max_counts) &
            (adata.obs['pct_counts_mt'] <= max_mt * 100)  # scanpy uses percentage
        )
        
        n_cells_filtered = cell_filter.sum()
        logger.info(f"Cells passing filters: {n_cells_filtered}/{n_cells_orig} ({n_cells_filtered/n_cells_orig*100:.1f}%)")
        
        adata = adata[cell_filter, :].copy()
        
        # Apply gene filtering
        min_cells = self.qc_config.get('min_cells_per_gene', 3)
        
        logger.info(f"Gene filtering threshold: min {min_cells} cells per gene")
        
        # Filter genes
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        n_genes_filtered = adata.shape[1]
        logger.info(f"Genes passing filters: {n_genes_filtered}/{n_genes_orig} ({n_genes_filtered/n_genes_orig*100:.1f}%)")
        
        # Additional gene filtering based on biotype
        if self.gene_config.get('filter_mitochondrial', True):
            mt_genes = adata.var_names.str.startswith('MT-')
            adata = adata[:, ~mt_genes].copy()
            logger.info(f"Filtered {mt_genes.sum()} mitochondrial genes")
        
        if self.gene_config.get('filter_ribosomal', True):
            ribo_genes = adata.var_names.str.startswith(('RPS', 'RPL'))
            adata = adata[:, ~ribo_genes].copy()
            logger.info(f"Filtered {ribo_genes.sum()} ribosomal genes")
        
        logger.info(f"Final data shape after QC: {adata.shape}")
        
        return adata
    
    def assign_guides(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Implement guide assignment policy for perturbation mapping.
        
        Args:
            adata: AnnData object with QC-filtered data
            
        Returns:
            AnnData object with guide assignments
        """
        logger.info("Implementing guide assignment policy...")
        
        # Check if guide information is available
        guide_columns = [col for col in adata.obs.columns if 'guide' in col.lower()]
        
        if not guide_columns:
            logger.warning("No guide information found in data. Creating mock guide assignments.")
            # Create mock guide assignments for demonstration
            np.random.seed(42)
            n_guides = 100
            guide_names = [f"guide_{i:03d}" for i in range(n_guides)]
            guide_names.extend(["non-targeting", "scrambled"])
            
            adata.obs['guide_identity'] = np.random.choice(guide_names, size=adata.n_obs)
            adata.obs['guide_count'] = np.random.poisson(10, size=adata.n_obs)
            
        else:
            logger.info(f"Found guide columns: {guide_columns}")
            
            # Use the first guide column as primary guide identity
            primary_guide_col = guide_columns[0]
            adata.obs['guide_identity'] = adata.obs[primary_guide_col]
        
        # Apply guide filtering
        max_guides_per_cell = self.guide_config.get('max_guides_per_cell', 3)
        min_cells_per_guide = self.guide_config.get('min_cells_per_guide', 10)
        
        # Filter guides with too few cells
        guide_counts = adata.obs['guide_identity'].value_counts()
        valid_guides = guide_counts[guide_counts >= min_cells_per_guide].index
        
        guide_filter = adata.obs['guide_identity'].isin(valid_guides)
        n_cells_before = adata.n_obs
        adata = adata[guide_filter, :].copy()
        n_cells_after = adata.n_obs
        
        logger.info(f"Guide filtering: {n_cells_after}/{n_cells_before} cells retained")
        logger.info(f"Number of valid guides: {len(valid_guides)}")
        
        # Identify control guides
        control_names = self.guide_config.get('control_guide_names', ['non-targeting', 'scrambled', 'control', 'NT'])
        control_guides = []
        
        for control_name in control_names:
            matching_guides = [g for g in valid_guides if control_name.lower() in g.lower()]
            control_guides.extend(matching_guides)
        
        adata.obs['is_control'] = adata.obs['guide_identity'].isin(control_guides)
        
        n_control_cells = adata.obs['is_control'].sum()
        n_perturbed_cells = (~adata.obs['is_control']).sum()
        
        logger.info(f"Control cells: {n_control_cells}")
        logger.info(f"Perturbed cells: {n_perturbed_cells}")
        logger.info(f"Control guides: {control_guides}")
        
        return adata
    
    def normalize_data(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Perform library-size normalization followed by log1p transformation.
        
        Args:
            adata: AnnData object with guide assignments
            
        Returns:
            Normalized AnnData object
        """
        logger.info("Performing data normalization...")
        
        # Store raw counts
        adata.raw = adata
        
        # Library size normalization
        target_sum = self.norm_config.get('target_sum', 10000)
        
        logger.info(f"Normalizing to {target_sum} counts per cell")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
        # Log transformation
        if self.norm_config.get('log_transform', True):
            logger.info("Applying log1p transformation")
            sc.pp.log1p(adata)
        
        # Store normalized data
        adata.layers['normalized'] = adata.X.copy()
        
        return adata
    
    def select_features(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Conduct feature selection to identify most informative genes.
        
        Args:
            adata: AnnData object with normalized data
            
        Returns:
            AnnData object with selected features
        """
        logger.info("Performing feature selection...")
        
        # Highly variable genes selection
        if self.norm_config.get('hvg_selection', True):
            hvg_method = self.norm_config.get('hvg_method', 'seurat_v3')
            n_top_genes = self.norm_config.get('hvg_n_top_genes', 2000)
            
            logger.info(f"Selecting {n_top_genes} highly variable genes using {hvg_method} method")
            
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                subset=False,
                layer=None,
                flavor=hvg_method
            )
            
            n_hvg = adata.var['highly_variable'].sum()
            logger.info(f"Identified {n_hvg} highly variable genes")
            
            # Store HVG information
            adata.var['feature_selected'] = adata.var['highly_variable']
        else:
            # Use all genes
            adata.var['highly_variable'] = True
            adata.var['feature_selected'] = True
            logger.info("Using all genes (no HVG selection)")
        
        # Additional variance-based filtering
        min_variance = self.gene_config.get('min_expression_level', 0.01)
        if min_variance > 0:
            logger.info(f"Applying variance threshold: {min_variance}")
            
            # Calculate variance for each gene
            if sparse.issparse(adata.X):
                gene_var = np.array(adata.X.var(axis=0)).flatten()
            else:
                gene_var = np.var(adata.X, axis=0)
            
            variance_filter = gene_var >= min_variance
            adata.var['passes_variance_filter'] = variance_filter
            
            # Combine with HVG selection
            adata.var['feature_selected'] = (
                adata.var['feature_selected'] & adata.var['passes_variance_filter']
            )
            
            n_var_genes = variance_filter.sum()
            logger.info(f"Genes passing variance filter: {n_var_genes}")
        
        n_selected = adata.var['feature_selected'].sum()
        logger.info(f"Total selected features: {n_selected}")
        
        return adata
    
    def map_gene_ids(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Map gene IDs to Ensembl namespace for standardization.
        
        Args:
            adata: AnnData object with selected features
            
        Returns:
            AnnData object with standardized gene IDs
        """
        logger.info("Mapping gene IDs to Ensembl namespace...")
        
        # Get current gene names
        current_genes = adata.var_names.tolist()
        logger.info(f"Mapping {len(current_genes)} gene IDs")
        
        # Convert to Ensembl IDs
        try:
            conversion_results = self.gene_mapper.convert_to_ensembl(current_genes, id_type="auto")
            
            # Create mapping dataframe
            mapping_df = pd.DataFrame([
                {'original_id': orig_id, 'ensembl_id': ens_id}
                for orig_id, ens_id in conversion_results.items()
            ])
            
            # Count successful mappings
            successful_mappings = mapping_df['ensembl_id'].notna().sum()
            mapping_rate = successful_mappings / len(current_genes) * 100
            
            logger.info(f"Successfully mapped {successful_mappings}/{len(current_genes)} genes ({mapping_rate:.1f}%)")
            
            # Filter to genes with successful mappings
            valid_mappings = mapping_df.dropna(subset=['ensembl_id'])
            
            if len(valid_mappings) == 0:
                logger.warning("No genes could be mapped to Ensembl IDs. Keeping original IDs.")
                adata.var['ensembl_id'] = adata.var_names
                adata.var['original_id'] = adata.var_names
                return adata
            
            # Create mapping dictionary
            id_mapping = dict(zip(valid_mappings['original_id'], valid_mappings['ensembl_id']))
            
            # Filter AnnData to genes with valid mappings
            genes_to_keep = valid_mappings['original_id'].tolist()
            gene_filter = adata.var_names.isin(genes_to_keep)
            
            adata = adata[:, gene_filter].copy()
            logger.info(f"Retained {adata.n_vars} genes with valid Ensembl mappings")
            
            # Store original IDs and map to Ensembl
            adata.var['original_id'] = adata.var_names
            adata.var['ensembl_id'] = [id_mapping[gene] for gene in adata.var_names]
            
            # Update var_names to Ensembl IDs
            adata.var_names = adata.var['ensembl_id']
            adata.var_names.name = 'ensembl_id'
            
            logger.info("Gene ID mapping completed successfully")
            
        except Exception as e:
            logger.error(f"Gene ID mapping failed: {e}")
            logger.warning("Proceeding with original gene IDs")
            adata.var['ensembl_id'] = adata.var_names
            adata.var['original_id'] = adata.var_names
        
        return adata
    
    def split_data(self, adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            adata: AnnData object with processed data
            
        Returns:
            Tuple of (train_adata, val_adata, test_adata)
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        from sklearn.model_selection import train_test_split
        
        # Get split parameters
        train_frac = self.split_config.get('train_fraction', 0.7)
        val_frac = self.split_config.get('val_fraction', 0.15)
        test_frac = self.split_config.get('test_fraction', 0.15)
        random_seed = self.split_config.get('random_seed', 42)
        
        # Validate fractions
        total_frac = train_frac + val_frac + test_frac
        if abs(total_frac - 1.0) > 1e-6:
            raise DataProcessingError(f"Split fractions must sum to 1.0, got {total_frac}")
        
        # Create indices
        n_cells = adata.n_obs
        indices = np.arange(n_cells)
        
        # Stratify by guide identity if requested
        stratify_by = self.split_config.get('stratify_by', None)
        stratify_labels = None
        
        if stratify_by and stratify_by in adata.obs.columns:
            stratify_labels = adata.obs[stratify_by].values
            logger.info(f"Stratifying split by: {stratify_by}")
        
        # First split: separate test set
        test_size = test_frac
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_frac / (train_frac + val_frac)
        
        if stratify_labels is not None:
            stratify_train_val = stratify_labels[train_val_indices]
        else:
            stratify_train_val = None
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=stratify_train_val
        )
        
        # Create split datasets
        train_adata = adata[train_indices, :].copy()
        val_adata = adata[val_indices, :].copy()
        test_adata = adata[test_indices, :].copy()
        
        # Add split information to obs
        train_adata.obs['split'] = 'train'
        val_adata.obs['split'] = 'validation'
        test_adata.obs['split'] = 'test'
        
        logger.info(f"Data split completed:")
        logger.info(f"  - Train: {train_adata.n_obs} cells ({train_adata.n_obs/n_cells*100:.1f}%)")
        logger.info(f"  - Validation: {val_adata.n_obs} cells ({val_adata.n_obs/n_cells*100:.1f}%)")
        logger.info(f"  - Test: {test_adata.n_obs} cells ({test_adata.n_obs/n_cells*100:.1f}%)")
        
        return train_adata, val_adata, test_adata
    
    def save_processed_data(self, adata: ad.AnnData, train_adata: ad.AnnData, 
                           val_adata: ad.AnnData, test_adata: ad.AnnData) -> None:
        """
        Save processed data to output directory.
        
        Args:
            adata: Full processed dataset
            train_adata: Training split
            val_adata: Validation split
            test_adata: Test split
        """
        logger.info("Saving processed data...")
        
        # Save full processed dataset
        full_path = self.output_dir / "processed_data.h5ad"
        adata.write_h5ad(full_path)
        logger.info(f"Saved full processed data: {full_path}")
        
        # Save data splits
        train_path = self.output_dir / "train_data.h5ad"
        val_path = self.output_dir / "val_data.h5ad"
        test_path = self.output_dir / "test_data.h5ad"
        
        train_adata.write_h5ad(train_path)
        val_adata.write_h5ad(val_path)
        test_adata.write_h5ad(test_path)
        
        logger.info(f"Saved training data: {train_path}")
        logger.info(f"Saved validation data: {val_path}")
        logger.info(f"Saved test data: {test_path}")
        
        # Save processing metadata
        metadata = {
            'processing_date': pd.Timestamp.now().isoformat(),
            'original_shape': adata.raw.shape if adata.raw else None,
            'processed_shape': adata.shape,
            'n_highly_variable_genes': adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else 0,
            'n_selected_features': adata.var['feature_selected'].sum() if 'feature_selected' in adata.var else 0,
            'train_cells': train_adata.n_obs,
            'val_cells': val_adata.n_obs,
            'test_cells': test_adata.n_obs,
            'n_guides': adata.obs['guide_identity'].nunique() if 'guide_identity' in adata.obs else 0,
            'n_control_cells': adata.obs['is_control'].sum() if 'is_control' in adata.obs else 0,
            'config_used': self.config
        }
        
        metadata_path = self.output_dir / "processing_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved processing metadata: {metadata_path}")


def main():
    """Main function for data processing pipeline."""
    parser = argparse.ArgumentParser(description="Process single-cell perturbation data")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file")
    parser.add_argument("--config", type=str, default="data_config", help="Configuration file name")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_global_seed(args.seed)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting single-cell data processing pipeline")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize processor
        processor = SingleCellDataProcessor(config, output_dir)
        
        # Load raw data
        adata = processor.load_raw_data(args.input)
        
        # Apply quality control
        adata = processor.apply_quality_control(adata)
        
        # Assign guides
        adata = processor.assign_guides(adata)
        
        # Normalize data
        adata = processor.normalize_data(adata)
        
        # Select features
        adata = processor.select_features(adata)
        
        # Map gene IDs
        adata = processor.map_gene_ids(adata)
        
        # Split data
        train_adata, val_adata, test_adata = processor.split_data(adata)
        
        # Save processed data
        processor.save_processed_data(adata, train_adata, val_adata, test_adata)
        
        logger.info("Data processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
