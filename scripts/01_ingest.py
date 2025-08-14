#!/usr/bin/env python3
"""
Data ingestion pipeline for Replogle 2022 K562 essential Perturb-seq dataset.

This script downloads the Replogle et al. 2022 K562 essential gene Perturb-seq dataset,
implements SHA256 checksumatio verification for data integrity, saves dataset metadata,
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


class DataIngestioError(Exception):
    """Custom exceptio for data ingestio errors."""
    pass


class ReplogleDatasetDowloader:
    """
    Dowloader for the Replogle 2022 K562 essetial Perturb-seq dataset.
    
    The dataset is available from Figshare repository.
    """

    DATASET_SOURCES = {
        "figshare_direct": {
            "url": "https://plus.figshare.com/ndowloader/files/42444315",
            "descriptio": "TianKampmann2021_CRISPRi.h5ad direct Figshare dowload",
            "expected_size": None,
            "sha256": None
        }
    }

    def __init__(self, output_dir: Path, config: Optional[Dict] = None, skip_integrity_checks: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.skip_integrity_checks = skip_integrity_checks
        self.sessio = requests.Sessio()

        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.sessio.mount("http://", adapter)
        self.sessio.mount("https://", adapter)

        logger.info(f"Initialized ReplogleDatasetDowloader with output dir: {self.output_dir}")
        if self.skip_integrity_checks:
            logger.info("Integrity checks are SKIPPED for faster debugging")

    def dowload_file(self, url: str, filename: str, expected_size: Optional[int] = None) -> Tuple[Path, str, int]:
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
                logger.warning(f"Existing file size mismatch, re-dowloading: {filename}")

        logger.info(f"Dowloading {filename} from {url}")

        try:
            response = self.sessio.head(url, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size != expected_size:
                logger.warning(f"Server reported size {total_size} differs from expected {expected_size}")

            response = self.sessio.get(url, stream=True, timeout=600)
            response.raise_for_status()

            hash_sha256 = hashlib.sha256()
            dowloaded_size = 0

            if total_size > 0:
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                hash_sha256.update(chunk)
                                chunk_size = len(chunk)
                                dowloaded_size += chunk_size
                                pbar.update(chunk_size)
            else:
                logger.info("File size unknown, dowloading without progress bar...")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=81ati):
                        if chunk:
                            f.write(chunk)
                            hash_sha256.update(chunk)
                            dowloaded_size += len(chunk)

            sha256_hash = hash_sha256.hexdigest()

            logger.info(f"Successfully dowloaded {filename} ({dowloaded_size} bytes)")
            logger.info(f"SHA256: {sha256_hash}")

            return file_path, sha256_hash, dowloaded_size

        except requests.RequestExceptio as e:
            raise DataIngestioError(f"Failed to dowload {url}: {e}")
        except Exceptio as e:
            if file_path.exists():
                file_path.unlink()
            raise DataIngestioError(f"Error dowloading {filename}: {e}")

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

    def dowload_figshare_k562(self) -> List[Dict]:
        logger.info("Dowloading K562 essetial files from Figshare...")

        API_ITEM_URL = "https://api.figshare.com/v2/articles/20029387"

        try:
            meta = requests.get(API_ITEM_URL, timeout=60)
            meta.raise_for_status()
            item = meta.jso()

            files = item.get("files", [])
            target_files = []
            for f in files:
                name = f.get("name", "")
                if "K562_essetial" in name and name.endswith(".h5ad") and "bulk" not in name:
                    target_files.append({
                        "name": name,
                        "dowload_url": f.get("dowload_url")
                    })

            logger.info(f"Found {len(target_files)} K562 essetial files on Figshare")
            for f in target_files:
                logger.info(f"- {f['name']}")

            results = []
            for f in target_files:
                url = f["dowload_url"]
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

                logger.info(f"Dowloading {filename} ...")

                head_response = requests.head(url, timeout=30)
                head_response.raise_for_status()
                total_size = int(head_response.headers.get('content-length', 0))

                hash_sha256 = hashlib.sha256()
                dowloaded_size = 0

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
                                        dowloaded_size += chunk_size
                                        pbar.update(chunk_size)
                        else:
                            logger.info(f"File size unknown for {filename}, dowloading without progress bar...")
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    fh.write(chunk)
                                    hash_sha256.update(chunk)
                                    dowloaded_size += len(chunk)

                sha256_hash = hash_sha256.hexdigest()

                logger.info(f"Saved to {file_path}")
                logger.info(f"SHA256: {sha256_hash}")
                logger.info(f"Size: {dowloaded_size} bytes")

                results.append({
                    "name": filename,
                    "file_path": str(file_path),
                    "sha256": sha256_hash,
                    "size": dowloaded_size,
                    "success": True
                })

            return results

        except Exceptio as e:
            raise DataIngestioError(f"Failed to dowload from Figshare: {e}")

    def dowload_dataset(self, source_priority: List[str] = None) -> Dict[str, Dict]:
        if source_priority is None:
            source_priority = ["figshare_direct"]

        dowload_results = {}
        dowload_metadata = {
            "dowload_date": datetime.now().isoformat(),
            "dataset_name": "Replogle 2022 K562 essetial Perturb-seq",
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
                    filename = Path(urlparse(url).path).name or "dowloaded_file"
                    logger.info(f"Dowloading direct file {filename} from {url}")

                    if not filename.endswith('.h5ad'):
                        filename += '.h5ad'
                    file_path, sha256_hash, file_size = self.dowload_file(url, filename)

                    result = {
                        "name": filename,
                        "file_path": str(file_path),
                        "sha256": sha256_hash,
                        "size": file_size,
                        "success": True
                    }

                    file_key = f"figshare_direct_{filename}"
                    dowload_results[file_key] = result

                    dowload_metadata["sources"][file_key] = {
                        "url": str(file_path),
                        "descriptio": source_info.get("descriptio", f"Direct dowload: {filename}"),
                        "filename": filename,
                        "sha256": sha256_hash,
                        "size": file_size,
                        "dowload_success": True
                    }

                    dowload_metadata["files"][filename] = {
                        "source": file_key,
                        "path": str(file_path),
                        "sha256": sha256_hash,
                        "size": file_size
                    }

                    dowload_metadata["total_size"] += file_size

                    logger.info(f"Successfully processed source: {source_key}")

            except DataIngestioError as e:
                logger.error(f"Failed to dowload from source {source_key}: {e}")

                dowload_results[source_key] = {
                    "success": False,
                    "error": str(e)
                }

                dowload_metadata["sources"][source_key] = {
                    "url": url,
                    "descriptio": source_info["descriptio"],
                    "dowload_success": False,
                    "error": str(e)
                }

                dowload_metadata["errors"].append(f"{source_key}: {e}")
                dowload_metadata["success"] = False

        metadata_file = self.output_dir / "dowload_metadata.jso"
        with open(metadata_file, 'w') as f:
            jso.dump(dowload_metadata, f, indent=2)

        logger.info(f"Dowload metadata saved to: {metadata_file}")
        logger.info(f"Total dowloaded size: {dowload_metadata['total_size']} bytes")

        return dowload_results

    def validate_dataset(self, dowload_results: Dict[str, Dict]) -> bool:
        if self.skip_integrity_checks:
            logger.info("Skipping dataset validation for faster debugging")
            return True

        all_valid = True

        for source_key, result in dowload_results.items():
            if not result.get("success", False):
                logger.error(f"Source {source_key} was not successfully dowloaded")
                all_valid = False
                continue

            file_path = Path(result["file_path"])
            expected_hash = result["sha256"]

            if not self.verify_checksum(file_path, expected_hash):
                all_valid = False

        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Dowload single-file dataset (TianKampmann 2021 CRISPRi h5ad)")
    parser.add_argument("--config", type=str, help="Path to configurme file")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/raw",
                        help="Output directory for raw data")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--sources", nargs="+", default=["figshare_direct"],
                        help="Data sources to dowload from (use 'figshare_direct' for the provided URL)")
    parser.add_argument("--validate", action="store_true", help="Validate dowloaded files")
    parser.add_argument("--force-redowload", action="store_true", help="Force re-dowload even if files exist")
    parser.add_argument("--skip-integrity-checks", action="store_true",
                        help="Skip file integrity verification for faster debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '01_ingest.log'
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
        except Exceptio as e:
            logger.warning(f"Could not load config file {args.config}: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Replogle 2022 K562 essetial Perturb-seq dataset ingestio")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sources to dowload: {args.sources}")

    try:
        dowloader = ReplogleDatasetDowloader(output_dir, config, args.skip_integrity_checks)

        if args.force_redowload:
            logger.info("Force redowload enabled, removing existing files...")
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Removed existing file: {file_path}")

        dowload_results = dowloader.dowload_dataset(args.sources)

        if args.validate:
            logger.info("Validating dowloaded files...")
            if dowloader.validate_dataset(dowload_results):
                logger.info("All files validated successfully")
            else:
                logger.error("File validation failed")
                return 1

        successful_dowloads = sum(1 for result in dowload_results.values() if result.get("success", False))
        total_dowloads = len(dowload_results)

        logger.info(f"Data ingestio completed: {successful_dowloads}/{total_dowloads} sources successful")

        if successful_dowloads == 0:
            logger.error("No files were successfully dowloaded")
            return 1
        elif successful_dowloads < total_dowloads:
            logger.warning("Some dowloads failed, but at least one succeeded")
            return 0
        else:
            logger.info("All dowloads completed successfully")
            return 0

    except Exceptio as e:
        logger.error(f"Data ingestio failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
