#!/usr/bin/env python3
"""
Data ingestion pipeline for Replogle 2022 K562 essential Perturb-seq dataset.

This script downloads the Replogle et al. 2022 K562 essential gene Perturb-seq dataset,
implements SHA256 checksum verification for data integrity, saves dataset metadata,
and stores raw data in the data/raw/ directory.

Reference:
Replogle, J.M., Norman, T.M., Xu, A. et al. Combinatorial single-cell CRISPR screens 
by direct guide RNA capture and targeted sequencing. Nat Biotechnol 38, 954–961 (2020).
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class ReplogleDatasetDownloader:
    """
    Downloader for the Replogle 2022 K562 essential Perturb-seq dataset.
    
    The dataset is available from Figshare repository.
    """

    DATASET_SOURCES = {
        "figshare_direct": {
            "url": "https://plus.figshare.com/ndownloader/files/42444315",
            "description": "TianKampmann2021_CRISPRi.h5ad direct Figshare download",
            "expected_size": None,
            "sha256": None
        }
    }

    def __init__(self, output_dir: Path, config: Optional[Dict] = None, skip_integrity_checks: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.skip_integrity_checks = skip_integrity_checks
        self.session = requests.Session()

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
        if self.skip_integrity_checks:
            logger.info("Integrity checks are SKIPPED for faster debugging")

    def download_file(self, url: str, filename: str, expected_size: Optional[int] = None) -> Tuple[Path, str, int]:
        file_path = self.output_dir / filename

        if file_path.exists():
            logger.info(f"File {filename} already exists, checking integrity...")
            existing_hash, existing_size = self._compute_file_hash_and_size(file_path)

            if self.skip_integrity_checks:
                logger.info(f"Using existing file: {filename} (integrity checks skipped)")
                return file_path, existing_hash, existing_size

            if expected_size is None or existing_size == expected_size:
                logger.info(f"Using existing file: {filename}")
                return file_path, existing_hash, existing_size
            else:
                logger.warning(f"Existing file size mismatch, re-downloading: {filename}")

        logger.info(f"Downloading {filename} from {url}")

        try:
            response = self.session.head(url, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size != expected_size:
                logger.warning(f"Server reported size {total_size} differs from expected {expected_size}")

            response = self.session.get(url, stream=True, timeout=600)
            response.raise_for_status()

            hash_sha256 = hashlib.sha256()
            downloaded_size = 0

            if total_size > 0:
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                hash_sha256.update(chunk)
                                chunk_size = len(chunk)
                                downloaded_size += chunk_size
                                pbar.update(chunk_size)
            else:
                logger.info("File size unknown, downloading without progress bar...")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            hash_sha256.update(chunk)
                            downloaded_size += len(chunk)

            sha256_hash = hash_sha256.hexdigest()

            logger.info(f"Successfully downloaded {filename} ({downloaded_size} bytes)")
            logger.info(f"SHA256: {sha256_hash}")

            return file_path, sha256_hash, downloaded_size

        except requests.RequestException as e:
            raise DataIngestionError(f"Failed to download {url}: {e}")
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise DataIngestionError(f"Error downloading {filename}: {e}")

    def _compute_file_hash_and_size(self, file_path: Path) -> Tuple[str, int]:
        hash_sha256 = hashlib.sha256()
        file_size = 0

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
                file_size += len(chunk)

        return hash_sha256.hexdigest(), file_size

    def verify_checksum(self, file_path: Path, expected_hash: str) -> bool:
        if self.skip_integrity_checks:
            logger.info(f"Skipping integrity check for {file_path.name}")
            return True

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

    def download_figshare_k562(self) -> List[Dict]:
        logger.info("Downloading K562 essential files from Figshare...")

        API_ITEM_URL = "https://api.figshare.com/v2/articles/20029387"

        try:
            meta = requests.get(API_ITEM_URL, timeout=60)
            meta.raise_for_status()
            item = meta.json()

            files = item.get("files", [])
            target_files = []
            for f in files:
                name = f.get("name", "")
                if "K562_essential" in name and name.endswith(".h5ad") and "bulk" not in name:
                    target_files.append({
                        "name": name,
                        "download_url": f.get("download_url")
                    })

            logger.info(f"Found {len(target_files)} K562 essential files on Figshare")
            for f in target_files:
                logger.info(f"- {f['name']}")

            results = []
            for f in target_files:
                url = f["download_url"]
                if not url:
                    logger.warning(f"Skipping (no direct URL): {f['name']}")
                    continue

                filename = f["name"]
                file_path = self.output_dir / filename

                if file_path.exists():
                    logger.info(f"File {filename} already exists, checking integrity...")
                    existing_hash, existing_size = self._compute_file_hash_and_size(file_path)

                    if self.skip_integrity_checks:
                        logger.info(f"Using existing file: {filename} (integrity checks skipped)")
                        results.append({
                            "name": filename,
                            "file_path": str(file_path),
                            "sha256": existing_hash,
                            "size": existing_size,
                            "success": True
                        })
                        continue

                    logger.info(f"Using existing file: {filename} (SHA256: {existing_hash})")
                    results.append({
                        "name": filename,
                        "file_path": str(file_path),
                        "sha256": existing_hash,
                        "size": existing_size,
                        "success": True
                    })
                    continue

                logger.info(f"Downloading {filename} ...")

                head_response = requests.head(url, timeout=30)
                head_response.raise_for_status()
                total_size = int(head_response.headers.get('content-length', 0))

                hash_sha256 = hashlib.sha256()
                downloaded_size = 0

                with requests.get(url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as fh:
                        if total_size > 0:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        fh.write(chunk)
                                        hash_sha256.update(chunk)
                                        chunk_size = len(chunk)
                                        downloaded_size += chunk_size
                                        pbar.update(chunk_size)
                        else:
                            logger.info(f"File size unknown for {filename}, downloading without progress bar...")
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    fh.write(chunk)
                                    hash_sha256.update(chunk)
                                    downloaded_size += len(chunk)

                sha256_hash = hash_sha256.hexdigest()

                logger.info(f"Saved to {file_path}")
                logger.info(f"SHA256: {sha256_hash}")
                logger.info(f"Size: {downloaded_size} bytes")

                results.append({
                    "name": filename,
                    "file_path": str(file_path),
                    "sha256": sha256_hash,
                    "size": downloaded_size,
                    "success": True
                })

            return results

        except Exception as e:
            raise DataIngestionError(f"Failed to download from Figshare: {e}")

    def download_dataset(self, source_priority: List[str] = None) -> Dict[str, Dict]:
        if source_priority is None:
            source_priority = ["figshare_direct"]

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

            try:
                if source_key == "figshare_direct":
                    filename = Path(urlparse(url).path).name or "downloaded_file"
                    logger.info(f"Downloading direct file {filename} from {url}")

                    if not filename.endswith('.h5ad'):
                        filename += '.h5ad'
                    file_path, sha256_hash, file_size = self.download_file(url, filename)

                    result = {
                        "name": filename,
                        "file_path": str(file_path),
                        "sha256": sha256_hash,
                        "size": file_size,
                        "success": True
                    }

                    file_key = f"figshare_direct_{filename}"
                    download_results[file_key] = result

                    download_metadata["sources"][file_key] = {
                        "url": str(file_path),
                        "description": source_info.get("description", f"Direct download: {filename}"),
                        "filename": filename,
                        "sha256": sha256_hash,
                        "size": file_size,
                        "download_success": True
                    }

                    download_metadata["files"][filename] = {
                        "source": file_key,
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

        metadata_file = self.output_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(download_metadata, f, indent=2)

        logger.info(f"Download metadata saved to: {metadata_file}")
        logger.info(f"Total downloaded size: {download_metadata['total_size']} bytes")

        return download_results

    def validate_dataset(self, download_results: Dict[str, Dict]) -> bool:
        if self.skip_integrity_checks:
            logger.info("Skipping dataset validation for faster debugging")
            return True

        all_valid = True

        for source_key, result in download_results.items():
            if not result.get("success", False):
                logger.error(f"Source {source_key} was not successfully downloaded")
                all_valid = False
                continue

            file_path = Path(result["file_path"])
            expected_hash = result["sha256"]

            if not self.verify_checksum(file_path, expected_hash):
                all_valid = False

        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Download single-file dataset (TianKampmann 2021 CRISPRi h5ad)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/raw",
                        help="Output directory for raw data")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--sources", nargs="+", default=["figshare_direct"],
                        help="Data sources to download from (use 'figshare_direct' for the provided URL)")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded files")
    parser.add_argument("--force-redownload", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--skip-integrity-checks", action="store_true",
                        help="Skip file integrity verification for faster debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '01_ingestion.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    set_global_seed(args.seed)

    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.warning(f"Could not load config file {args.config}: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Replogle 2022 K562 essential Perturb-seq dataset ingestion")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sources to download: {args.sources}")

    try:
        downloader = ReplogleDatasetDownloader(output_dir, config, args.skip_integrity_checks)

        if args.force_redownload:
            logger.info("Force redownload enabled, removing existing files...")
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Removed existing file: {file_path}")

        download_results = downloader.download_dataset(args.sources)

        if args.validate:
            logger.info("Validating downloaded files...")
            if downloader.validate_dataset(download_results):
                logger.info("All files validated successfully")
            else:
                logger.error("File validation failed")
                return 1

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
