#!/usr/bin/env python3
"""Test script to verify imports work correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.utils.config import load_config
    print("✓ Config import successful")
except ImportError as e:
    print(f"✗ Config import failed: {e}")

try:
    from src.utils.random_seed import set_global_seed
    print("✓ Random seed import successful")
except ImportError as e:
    print(f"✗ Random seed import failed: {e}")

try:
    import requests
    print("✓ Requests import successful")
except ImportError as e:
    print(f"✗ Requests import failed: {e}")

try:
    from tqdm import tqdm
    print("✓ tqdm import successful")
except ImportError as e:
    print(f"✗ tqdm import failed: {e}")

print("Import test completed")
