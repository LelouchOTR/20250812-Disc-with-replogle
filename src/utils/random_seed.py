"""
Global reproducible random seed management for the single-cell perturbation analysis pipeline.
Ensures reproducibility across numpy, torch, random, and other libraries.
"""

import os
import random
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global seed state
_global_seed: Optional[int] = None
_seed_initialized: bool = False


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set global random seed for reproducible results across all libraries.
    
    Args:
        seed: Random seed value. If None, uses environment variable RANDOM_SEED
              or generates a random seed.
    
    Returns:
        The seed value that was set
    """
    global _global_seed, _seed_initialized
    
    if seed is None:
        # Try to get seed from environment variable
        env_seed = os.getenv('RANDOM_SEED')
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                logger.warning(f"Invalid RANDOM_SEED environment variable: {env_seed}")
                seed = None
        
        # Generate random seed if still None
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
            logger.info(f"Generated random seed: {seed}")
    
    # Set seeds for all libraries
    _set_seeds(seed)
    
    _global_seed = seed
    _seed_initialized = True
    
    logger.info(f"Global random seed set to: {seed}")
    return seed


def _set_seeds(seed: int) -> None:
    """
    Set random seeds for all relevant libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        
        # Set CUDA seeds if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Additional CUDA determinism settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            logger.debug("Set CUDA random seeds and deterministic settings")
        
        logger.debug("Set PyTorch random seeds")
        
    except ImportError:
        logger.debug("PyTorch not available, skipping torch seed setting")
    
    # Scanpy (if available)
    try:
        import scanpy as sc
        sc.settings.seed = seed
        logger.debug("Set scanpy random seed")
    except ImportError:
        logger.debug("Scanpy not available, skipping scanpy seed setting")
    
    # Set environment variable for other libraries that might use it
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_global_seed() -> Optional[int]:
    """
    Get the current global random seed.
    
    Returns:
        The current global seed, or None if not set
    """
    return _global_seed


def is_seed_initialized() -> bool:
    """
    Check if the global seed has been initialized.
    
    Returns:
        True if seed has been initialized, False otherwise
    """
    return _seed_initialized


def ensure_seed_initialized(default_seed: int = 42) -> int:
    """
    Ensure that the global seed is initialized, setting a default if not.
    
    Args:
        default_seed: Default seed to use if not already initialized
        
    Returns:
        The current or newly set seed
    """
    if not _seed_initialized:
        return set_global_seed(default_seed)
    return _global_seed


def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a new NumPy random number generator with optional seed.
    
    Args:
        seed: Seed for the RNG. If None, uses the global seed.
              If global seed is not set, uses a random seed.
    
    Returns:
        NumPy random number generator
    """
    if seed is None:
        if _seed_initialized:
            seed = _global_seed
        else:
            seed = np.random.randint(0, 2**31 - 1)
    
    return np.random.default_rng(seed)


def get_reproducible_state() -> dict:
    """
    Get the current state of all random number generators for reproducibility.
    
    Returns:
        Dictionary containing the state of all RNGs
    """
    state = {
        'global_seed': _global_seed,
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
    }
    
    # PyTorch state (if available)
    try:
        import torch
        state['torch_random_state'] = torch.get_rng_state()
        
        if torch.cuda.is_available():
            state['torch_cuda_random_state'] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    
    return state


def set_reproducible_state(state: dict) -> None:
    """
    Restore the state of all random number generators.
    
    Args:
        state: Dictionary containing RNG states (from get_reproducible_state)
    """
    global _global_seed, _seed_initialized
    
    if 'global_seed' in state:
        _global_seed = state['global_seed']
        _seed_initialized = True
    
    if 'python_random_state' in state:
        random.setstate(state['python_random_state'])
    
    if 'numpy_random_state' in state:
        np.random.set_state(state['numpy_random_state'])
    
    # PyTorch state (if available)
    try:
        import torch
        if 'torch_random_state' in state:
            torch.set_rng_state(state['torch_random_state'])
        
        if torch.cuda.is_available() and 'torch_cuda_random_state' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_state'])
    except ImportError:
        pass
    
    logger.info("Restored reproducible random state")


class SeedContext:
    """
    Context manager for temporary seed changes.
    
    Example:
        with SeedContext(123):
            # Code here uses seed 123
            pass
        # Original seed is restored
    """
    
    def __init__(self, seed: int):
        """
        Initialize the seed context.
        
        Args:
            seed: Temporary seed to use
        """
        self.temp_seed = seed
        self.original_state = None
    
    def __enter__(self):
        """Save current state and set temporary seed."""
        self.original_state = get_reproducible_state()
        _set_seeds(self.temp_seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state."""
        if self.original_state is not None:
            set_reproducible_state(self.original_state)


# Initialize with default seed if RANDOM_SEED environment variable is set
if os.getenv('RANDOM_SEED') is not None:
    try:
        set_global_seed()
    except Exception as e:
        logger.warning(f"Failed to initialize seed from environment: {e}")
