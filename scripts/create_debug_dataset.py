#!/usr/bin/env python3
"""
Create a smaller, stratified subset of the single-cell dataset for debugging.

This script performs stratified sampling on the full AnnData object to generate
a smaller dataset that preserves the relative proportions of biological groups,
such as perturbations or cell types. This is crucial for rapid and meaningful
debugging of downstream models.
"""

import argparse
import logging
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_stratified_subset(
    adata: ad.AnnData,
    strata_key: str,
    fraction: float,
    min_samples_per_stratum: int = 1,
) -> ad.AnnData:
    """
    Generates a stratified subset of an AnnData object.

    Args:
        adata: The full AnnData object.
        strata_key: The column key in adata.obs to use for stratification.
        fraction: The fraction of data to retain (e.g., 0.1 for 10%).
        min_samples_per_stratum: The minimum number of samples to keep for each stratum.
                                 This prevents dropping very rare groups.

    Returns:
        A new AnnData object containing the stratified subset.
    """
    if strata_key not in adata.obs:
        raise ValueError(f"Stratification key '{strata_key}' not found in adata.obs.")

    # Ensure the stratification column is treated as a categorical variable
    strata_values = adata.obs[strata_key].astype('category')
    
    # Identify strata with fewer samples than the minimum required
    strata_counts = strata_values.value_counts()
    small_strata = strata_counts[strata_counts < min_samples_per_stratum].index
    if small_strata.any():
        logger.warning(
            f"The following strata have fewer than {min_samples_per_stratum} samples "
            f"and will be excluded from stratification: {small_strata.tolist()}"
        )

    # Use train_test_split to get a stratified subset of indices
    # We generate a dummy array of indices to split
    indices = np.arange(adata.n_obs)
    
    # We only need the 'train' part of the split, which will be our subset
    subset_indices, _ = train_test_split(
        indices,
        test_size=1 - fraction,
        stratify=strata_values,
        random_state=42,  # for reproducibility
    )

    logger.info(f"Selected {len(subset_indices)} cells out of {adata.n_obs} ({fraction:.1%}).")

    # Create the new AnnData object from the selected indices
    subset_adata = adata[subset_indices].copy()

    # Log the value counts to verify stratification
    original_dist = adata.obs[strata_key].value_counts(normalize=True)
    subset_dist = subset_adata.obs[strata_key].value_counts(normalize=True)
    
    comparison_df = pd.DataFrame({
        'Original_Proportion': original_dist,
        'Subset_Proportion': subset_dist
    }).fillna(0)
    
    logger.info("Distribution of strata in original vs. subset dataset:")
    logger.info("\n" + comparison_df.to_string(float_format="%.4f"))

    return subset_adata



def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Create a stratified debug dataset.")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the full input .h5ad file (e.g., train_data.h5ad).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the output subset .h5ad file.",
    )
    parser.add_argument(
        "--strata-key",
        type=str,
        default="perturbation_label",
        help="The key in .obs to use for stratification (e.g., 'perturbation_label', 'cell_type').",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="The fraction of the dataset to include in the subset (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum number of samples per stratum required for stratification.",
    )
    args = parser.parse_args()

    logger.info(f"Loading full dataset from: {args.input_file}")
    try:
        full_adata = ad.read_h5ad(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load AnnData file: {e}")
        sys.exit(1)

    logger.info("Creating stratified subset...")
    subset_adata = create_stratified_subset(
        adata=full_adata,
        strata_key=args.strata_key,
        fraction=args.fraction,
        min_samples_per_stratum=args.min_samples,
    )

    logger.info(f"Saving subset dataset to: {args.output-file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    subset_adata.write_h5ad(args.output_file)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
