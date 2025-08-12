"""
Configuration loading and validation utilities for the single-cell perturbation analysis pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigLoader:
    """
    YAML configuration loader with validation and environment variable substitution.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files. 
                       Defaults to 'configs/' relative to project root.
        """
        if config_dir is None:
            # Find project root (directory containing src/)
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / 'src').exists():
                    self.config_dir = current_dir / 'configs'
                    break
                current_dir = current_dir.parent
            else:
                raise ConfigError("Could not find project root directory")
        else:
            self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise ConfigError(f"Configuration directory not found: {self.config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file with environment variable substitution.
        
        Args:
            config_name: Name of the configuration file (with or without .yaml extension)
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            ConfigError: If the configuration file is not found or invalid
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Load YAML
            config = yaml.safe_load(content)
            
            if config is None:
                raise ConfigError(f"Empty configuration file: {config_path}")
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration {config_path}: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                var_name = var_expr
                value = os.getenv(var_name)
                if value is None:
                    raise ConfigError(f"Environment variable {var_name} not found and no default provided")
                return value
        
        # Replace ${VAR_NAME} and ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_env_var, content)
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a simple schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: Schema dictionary with required keys and types
            
        Returns:
            True if valid
            
        Raises:
            ConfigError: If validation fails
        """
        def validate_recursive(cfg: Dict[str, Any], sch: Dict[str, Any], path: str = ""):
            for key, expected_type in sch.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in cfg:
                    raise ConfigError(f"Missing required configuration key: {current_path}")
                
                value = cfg[key]
                
                if isinstance(expected_type, dict):
                    if not isinstance(value, dict):
                        raise ConfigError(f"Expected dict for {current_path}, got {type(value).__name__}")
                    validate_recursive(value, expected_type, current_path)
                elif isinstance(expected_type, type):
                    if not isinstance(value, expected_type):
                        raise ConfigError(f"Expected {expected_type.__name__} for {current_path}, got {type(value).__name__}")
                elif isinstance(expected_type, (list, tuple)):
                    # Check if value is one of the allowed types
                    if not any(isinstance(value, t) for t in expected_type):
                        type_names = [t.__name__ for t in expected_type]
                        raise ConfigError(f"Expected one of {type_names} for {current_path}, got {type(value).__name__}")
        
        try:
            validate_recursive(config, schema)
            return True
        except ConfigError:
            raise


# Global configuration loader instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        Configuration dictionary
    """
    return get_config_loader().load_config(config_name)


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load all standard configuration files for the pipeline.
    
    Returns:
        Dictionary with configuration names as keys and config dictionaries as values
    """
    config_files = ['data_config', 'model_config', 'graph_config', 'pipeline_config']
    configs = {}
    
    loader = get_config_loader()
    
    for config_name in config_files:
        try:
            configs[config_name] = loader.load_config(config_name)
        except ConfigError as e:
            logger.warning(f"Could not load {config_name}: {e}")
            configs[config_name] = {}
    
    return configs


# Configuration schemas for validation
DATA_CONFIG_SCHEMA = {
    'qc_thresholds': {
        'min_genes_per_cell': int,
        'max_genes_per_cell': int,
        'min_cells_per_gene': int,
        'max_mitochondrial_fraction': (int, float)
    },
    'guide_filtering': {
        'min_guide_efficiency': (int, float),
        'max_guides_per_cell': int
    },
    'normalization': {
        'target_sum': (int, float),
        'log_transform': bool
    }
}

MODEL_CONFIG_SCHEMA = {
    'architecture': {
        'encoder_dims': list,
        'decoder_dims': list,
        'latent_dim': int,
        'dropout_rate': (int, float)
    },
    'training': {
        'batch_size': int,
        'learning_rate': (int, float),
        'num_epochs': int,
        'early_stopping_patience': int
    },
    'loss_weights': {
        'reconstruction_weight': (int, float),
        'discrepancy_weight': (int, float)
    }
}

GRAPH_CONFIG_SCHEMA = {
    'go_filtering': {
        'min_term_size': int,
        'max_term_size': int,
        'max_depth': int,
        'evidence_codes': list
    },
    'adjacency': {
        'method': str,
        'threshold': (int, float)
    }
}

PIPELINE_CONFIG_SCHEMA = {
    'data_splitting': {
        'train_fraction': (int, float),
        'val_fraction': (int, float),
        'test_fraction': (int, float)
    },
    'feature_selection': {
        'method': str,
        'n_features': int
    },
    'output_paths': {
        'models_dir': str,
        'evaluation_dir': str
    }
}
