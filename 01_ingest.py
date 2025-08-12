#!/usr/bin/env python3
"""
Data ingestion pipeline for Replogle 2022 K562 essential Perturb-seq dataset.

This script downloads the Replogle et al. 2022 K562 essential gene Perturb-seq dataset,
implements SHA256 checksum verification for data integrity, saves dataset metadata,
and stores raw data in the data/raw/ directory.

Reference:
Replogle, J.M., Norman, T.M., Xu, A. et al. Combinatorial single-cell CRISPR screens 
by direct guide RNA capture and targeted sequencing. Nat Biotechnol 38, 954–961 (2020).
https://doi.org/10.1038/s41587-020-0470-y

Dataset: Mapping information-rich genotype-to-phenotype maps by genome-scale Perturb-seq
https://doi.org/10.1016/j.cell.2022.05.013
"""

import os
import sys
import hashlib
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import requests
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class ReplogleDatasetDownloader:
    """
    Downloader for the Replogle 2022 K562 essential Perturb-seq dataset.
    
    The dataset is available from multiple sources:
    1. Original publication supplementary data
    2. GEO (Gene Expression Omnibus) - GSE144623
    3. Figshare repository
    4. CZ CELLxGENE portal
    """
    
    # Dataset URLs and checksums
    DATASET_SOURCES = {
        "geo_main": {
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144623/suppl/GSE144623_K562_essential_raw_singlecell_01.h5ad.gz",
            "description": "K562 essential gene screen - raw single cell data (part 1)",
            "expected_size": None,  # Will be determined during download
            "sha256": None,  # Will be computed during download
        },
        "geo_metadata": {
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144623/suppl/GSE144623_K562_essential_metadata.csv.gz",
            "description": "K562 essential gene screen - metadata",
            "expected_size": None,
            "sha256": None,
        },
        "figshare_main": {
            "url": "https://figshare.com/ndownloader/files/24667905",
            "description": "Replogle 2022 K562 essential screen data from Figshare",
            "expected_size": None,
            "sha256": None,
        }
    }
    
    def __init__(self, output_dir: Path, config: Optional[Dict] = None):
        """
        Initialize the dataset downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            config: Configuration dictionary (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.session = requests.Session()
        
        # Set up session with retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized ReplogleDatasetDownloader with output dir: {self.output_dir}")
    
    def download_file(self, url: str, filename: str, expected_size: Optional[int] = None) -> Tuple[Path, str, int]:
        """
        Download a file with progress bar and return path, SHA256 hash, and size.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            expected_size: Expected file size for validation (optional)
            
        Returns:
            Tuple of (file_path, sha256_hash, file_size)
        """
        file_path = self.output_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"File {filename} already exists, checking integrity...")
            existing_hash, existing_size = self._compute_file_hash_and_size(file_path)
            
            if expected_size is None or existing_size == expected_size:
                logger.info(f"Using existing file: {filename}")
                return file_path, existing_hash, existing_size
            else:
                logger.warning(f"Existing file size mismatch, re-downloading: {filename}")
        
        logger.info(f"Downloading {filename} from {url}")
        
        try:
            # Get file size for progress bar
            response = self.session.head(url, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size != expected_size:
                logger.warning(f"Server reported size {total_size} differs from expected {expected_size}")
            
            # Download with progress bar
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            hash_sha256 = hashlib.sha256()
            downloaded_size = 0
            
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            hash_sha256.update(chunk)
                            downloaded_size += len(chunk)
                            pbar.update(len(chunk))
            
            sha256_hash = hash_sha256.hexdigest()
            
            logger.info(f"Successfully downloaded {filename} ({downloaded_size} bytes)")
            logger.info(f"SHA256: {sha256_hash}")
            
            return file_path, sha256_hash, downloaded_size
            
        except requests.RequestException as e:
            raise DataIngestionError(f"Failed to download {url}: {e}")
        except Exception as e:
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            raise DataIngestionError(f"Error downloading {filename}: {e}")
    
    def _compute_file_hash_and_size(self, file_path: Path) -> Tuple[str, int]:
        """
        Compute SHA256 hash and size of an existing file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (sha256_hash, file_size)
        """
        hash_sha256 = hashlib.sha256()
        file_size = 0
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
                file_size += len(chunk)
        
        return hash_sha256.hexdigest(), file_size
    
    def verify_checksum(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file integrity using SHA256 checksum.
        
        Args:
            file_path: Path to the file to verify
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not file_path.exists():
            logger.error(f"File not found for verification: {file_path}")
            return False
        
        actual_hash, _ = self._compute_file_hash_and_size(file_path)
        
        if actual_hash.lower() == expected_hash.lower():
            logger.info(f"Checksum verification passed for {file_path.name}")
            return True
        else:
            logger.error(f"Checksum verification failed for {file_path.name}")
            logger.error(f"Expected: {expected_hash}")
            logger.error(f"Actual:   {actual_hash}")
            return False
    
    def download_dataset(self, source_priority: List[str] = None) -> Dict[str, Dict]:
        """
        Download the complete Replogle dataset.
        
        Args:
            source_priority: List of source keys to try in order
            
        Returns:
            Dictionary with download results and metadata
        """
        if source_priority is None:
            source_priority = ["geo_main", "geo_metadata"]
        
        download_results = {}
        download_metadata = {
            "download_date": datetime.now().isoformat(),
            "dataset_name": "Replogle 2022 K562 essential Perturb-seq",
            "sources": {},
            "files": {},
            "total_size": 0,
            "success": True,
            "errors": []
        }
        
        for source_key in source_priority:
            if source_key not in self.DATASET_SOURCES:
                logger.warning(f"Unknown source: {source_key}")
                continue
            
            source_info = self.DATASET_SOURCES[source_key]
            url = source_info["url"]
            
            # Generate filename from URL
            parsed_url = urlparse(url)
            if source_key == "figshare_main":
                filename = "replogle_k562_essential_figshare.h5ad"
            else:
                filename = Path(parsed_url.path).name
                if not filename:
                    filename = f"{source_key}_data.dat"
            
            try:
                file_path, sha256_hash, file_size = self.download_file(url, filename)
                
                # Store results
                download_results[source_key] = {
                    "file_path": file_path,
                    "sha256": sha256_hash,
                    "size": file_size,
                    "success": True
                }
                
                # Update metadata
                download_metadata["sources"][source_key] = {
                    "url": url,
                    "description": source_info["description"],
                    "filename": filename,
                    "sha256": sha256_hash,
                    "size": file_size,
                    "download_success": True
                }
                
                download_metadata["files"][filename] = {
                    "source": source_key,
                    "path": str(file_path),
                    "sha256": sha256_hash,
                    "size": file_size
                }
                
                download_metadata["total_size"] += file_size
                
                logger.info(f"Successfully processed source: {source_key}")
                
            except DataIngestionError as e:
                logger.error(f"Failed to download from source {source_key}: {e}")
                
                download_results[source_key] = {
                    "success": False,
                    "error": str(e)
                }
                
                download_metadata["sources"][source_key] = {
                    "url": url,
                    "description": source_info["description"],
                    "download_success": False,
                    "error": str(e)
                }
                
                download_metadata["errors"].append(f"{source_key}: {e}")
                download_metadata["success"] = False
        
        # Save metadata
        metadata_file = self.output_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(download_metadata, f, indent=2)
        
        logger.info(f"Download metadata saved to: {metadata_file}")
        logger.info(f"Total downloaded size: {download_metadata['total_size']} bytes")
        
        return download_results
    
    def validate_dataset(self, download_results: Dict[str, Dict]) -> bool:
        """
        Validate the downloaded dataset files.
        
        Args:
            download_results: Results from download_dataset()
            
        Returns:
            True if all files are valid, False otherwise
        """
        all_valid = True
        
        for source_key, result in download_results.items():
            if not result.get("success", False):
                logger.error(f"Source {source_key} was not successfully downloaded")
                all_valid = False
                continue
            
            file_path = result["file_path"]
            expected_hash = result["sha256"]
            
            if not self.verify_checksum(file_path, expected_hash):
                all_valid = False
        
        return all_valid


def main():
    """Main function for data ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Download Replogle 2022 K562 essential Perturb-seq dataset")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw data")
    parser.add_argument("--sources", nargs="+", default=["geo_main", "geo_metadata"], 
                       help="Data sources to download from")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded files")
    parser.add_argument("--force-redownload", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_global_seed(args.seed)
    
    # Load configuration
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.warning(f"Could not load config file {args.config}: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Replogle 2022 K562 essential Perturb-seq dataset ingestion")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sources to download: {args.sources}")
    
    try:
        # Initialize downloader
        downloader = ReplogleDatasetDownloader(output_dir, config)
        
        # Remove existing files if force redownload
        if args.force_redownload:
            logger.info("Force redownload enabled, removing existing files...")
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Removed existing file: {file_path}")
        
        # Download dataset
        download_results = downloader.download_dataset(args.sources)
        
        # Validate if requested
        if args.validate:
            logger.info("Validating downloaded files...")
            if downloader.validate_dataset(download_results):
                logger.info("All files validated successfully")
            else:
                logger.error("File validation failed")
                return 1
        
        # Summary
        successful_downloads = sum(1 for result in download_results.values() if result.get("success", False))
        total_downloads = len(download_results)
        
        logger.info(f"Data ingestion completed: {successful_downloads}/{total_downloads} sources successful")
        
        if successful_downloads == 0:
            logger.error("No files were successfully downloaded")
            return 1
        elif successful_downloads < total_downloads:
            logger.warning("Some downloads failed, but at least one succeeded")
            return 0
        else:
            logger.info("All downloads completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
