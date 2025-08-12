"""
Ensembl gene ID namespace management and mapping utilities for the single-cell perturbation analysis pipeline.
Handles gene ID conversion, validation, and standardization to Ensembl namespace.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Union, Tuple
from pathlib import Path
import requests
import pandas as pd
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


class GeneIDError(Exception):
    """Custom exception for gene ID-related errors."""
    pass


class EnsemblGeneMapper:
    """
    Gene ID mapper for converting various gene identifiers to Ensembl gene IDs.
    Supports HGNC symbols, Entrez IDs, and other common gene identifiers.
    """
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, species: str = "human"):
        """
        Initialize the Ensembl gene mapper.
        
        Args:
            cache_dir: Directory to cache mapping files. Defaults to data/cache/
            species: Species for gene mapping (currently only 'human' supported)
        """
        self.species = species.lower()
        if self.species != "human":
            raise GeneIDError(f"Species '{species}' not supported. Only 'human' is currently supported.")
        
        if cache_dir is None:
            # Find project root and set cache directory
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / 'src').exists():
                    self.cache_dir = current_dir / 'data' / 'cache'
                    break
                current_dir = current_dir.parent
            else:
                raise GeneIDError("Could not find project root directory")
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping tables
        self._symbol_to_ensembl: Optional[Dict[str, str]] = None
        self._ensembl_to_symbol: Optional[Dict[str, str]] = None
        self._entrez_to_ensembl: Optional[Dict[str, str]] = None
        self._ensembl_to_entrez: Optional[Dict[str, str]] = None
        
        # Ensembl REST API base URL
        self.ensembl_rest_url = "https://rest.ensembl.org"
        
        logger.info(f"Initialized EnsemblGeneMapper for {species} with cache dir: {self.cache_dir}")
    
    def _download_ensembl_mapping(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Download gene mapping data from Ensembl BioMart.
        
        Args:
            force_refresh: Force download even if cached file exists
            
        Returns:
            DataFrame with gene mappings
        """
        cache_file = self.cache_dir / "ensembl_gene_mapping.tsv"
        
        if cache_file.exists() and not force_refresh:
            # Check if file is less than 30 days old
            file_age_days = (time.time() - cache_file.stat().st_mtime) / (24 * 3600)
            if file_age_days < 30:
                logger.info(f"Loading cached gene mapping from {cache_file}")
                return pd.read_csv(cache_file, sep='\t')
            else:
                logger.info("Cached gene mapping is older than 30 days, refreshing...")
        
        logger.info("Downloading gene mapping data from Ensembl BioMart...")
        
        # BioMart query URL for human gene mappings
        biomart_url = (
            "http://www.ensembl.org/biomart/martservice?"
            "query=<?xml version='1.0' encoding='UTF-8'?>"
            "<!DOCTYPE Query>"
            "<Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' count='' datasetConfigVersion='0.6'>"
            "<Dataset name='hsapiens_gene_ensembl' interface='default'>"
            "<Attribute name='ensembl_gene_id'/>"
            "<Attribute name='external_gene_name'/>"
            "<Attribute name='entrezgene_id'/>"
            "<Attribute name='gene_biotype'/>"
            "<Attribute name='chromosome_name'/>"
            "</Dataset>"
            "</Query>"
        )
        
        try:
            response = requests.get(biomart_url, timeout=300)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(response.text)
            
            # Load as DataFrame
            df = pd.read_csv(cache_file, sep='\t')
            
            # Clean up the data
            df = df.dropna(subset=['Gene stable ID', 'Gene name'])
            df = df[df['Gene stable ID'].str.startswith('ENSG')]
            
            logger.info(f"Downloaded {len(df)} gene mappings from Ensembl BioMart")
            return df
            
        except requests.RequestException as e:
            raise GeneIDError(f"Failed to download gene mapping data: {e}")
        except Exception as e:
            raise GeneIDError(f"Error processing gene mapping data: {e}")
    
    def _load_mappings(self, force_refresh: bool = False) -> None:
        """
        Load gene ID mappings from Ensembl.
        
        Args:
            force_refresh: Force refresh of cached data
        """
        if (self._symbol_to_ensembl is not None and 
            self._ensembl_to_symbol is not None and 
            not force_refresh):
            return
        
        df = self._download_ensembl_mapping(force_refresh)
        
        # Create mapping dictionaries
        self._symbol_to_ensembl = {}
        self._ensembl_to_symbol = {}
        self._entrez_to_ensembl = {}
        self._ensembl_to_entrez = {}
        
        for _, row in df.iterrows():
            ensembl_id = row['Gene stable ID']
            symbol = row['Gene name']
            entrez_id = row['NCBI gene (formerly Entrezgene) ID']
            
            if pd.notna(symbol) and symbol:
                self._symbol_to_ensembl[symbol.upper()] = ensembl_id
                self._ensembl_to_symbol[ensembl_id] = symbol
            
            if pd.notna(entrez_id) and str(entrez_id).isdigit():
                entrez_str = str(int(entrez_id))
                self._entrez_to_ensembl[entrez_str] = ensembl_id
                self._ensembl_to_entrez[ensembl_id] = entrez_str
        
        logger.info(f"Loaded {len(self._symbol_to_ensembl)} symbol mappings and "
                   f"{len(self._entrez_to_ensembl)} Entrez mappings")
    
    @lru_cache(maxsize=10000)
    def symbol_to_ensembl(self, symbol: str) -> Optional[str]:
        """
        Convert gene symbol to Ensembl gene ID.
        
        Args:
            symbol: Gene symbol (e.g., 'TP53')
            
        Returns:
            Ensembl gene ID or None if not found
        """
        self._load_mappings()
        return self._symbol_to_ensembl.get(symbol.upper())
    
    @lru_cache(maxsize=10000)
    def ensembl_to_symbol(self, ensembl_id: str) -> Optional[str]:
        """
        Convert Ensembl gene ID to gene symbol.
        
        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000141510')
            
        Returns:
            Gene symbol or None if not found
        """
        self._load_mappings()
        return self._ensembl_to_symbol.get(ensembl_id)
    
    @lru_cache(maxsize=10000)
    def entrez_to_ensembl(self, entrez_id: Union[str, int]) -> Optional[str]:
        """
        Convert Entrez gene ID to Ensembl gene ID.
        
        Args:
            entrez_id: Entrez gene ID
            
        Returns:
            Ensembl gene ID or None if not found
        """
        self._load_mappings()
        entrez_str = str(entrez_id)
        return self._entrez_to_ensembl.get(entrez_str)
    
    @lru_cache(maxsize=10000)
    def ensembl_to_entrez(self, ensembl_id: str) -> Optional[str]:
        """
        Convert Ensembl gene ID to Entrez gene ID.
        
        Args:
            ensembl_id: Ensembl gene ID
            
        Returns:
            Entrez gene ID or None if not found
        """
        self._load_mappings()
        return self._ensembl_to_entrez.get(ensembl_id)
    
    def convert_to_ensembl(self, gene_ids: Union[str, List[str]], 
                          id_type: str = "auto") -> Dict[str, Optional[str]]:
        """
        Convert gene IDs to Ensembl gene IDs.
        
        Args:
            gene_ids: Single gene ID or list of gene IDs
            id_type: Type of input IDs ('symbol', 'entrez', 'ensembl', or 'auto')
            
        Returns:
            Dictionary mapping input IDs to Ensembl IDs (None if not found)
        """
        if isinstance(gene_ids, str):
            gene_ids = [gene_ids]
        
        results = {}
        
        for gene_id in gene_ids:
            gene_id = str(gene_id).strip()
            
            if id_type == "auto":
                # Auto-detect ID type
                ensembl_id = self._auto_convert_to_ensembl(gene_id)
            elif id_type == "symbol":
                ensembl_id = self.symbol_to_ensembl(gene_id)
            elif id_type == "entrez":
                ensembl_id = self.entrez_to_ensembl(gene_id)
            elif id_type == "ensembl":
                ensembl_id = gene_id if self.is_valid_ensembl_id(gene_id) else None
            else:
                raise GeneIDError(f"Unknown ID type: {id_type}")
            
            results[gene_id] = ensembl_id
        
        return results
    
    def _auto_convert_to_ensembl(self, gene_id: str) -> Optional[str]:
        """
        Auto-detect gene ID type and convert to Ensembl.
        
        Args:
            gene_id: Gene ID of unknown type
            
        Returns:
            Ensembl gene ID or None if not found
        """
        gene_id = gene_id.strip()
        
        # Check if already Ensembl ID
        if self.is_valid_ensembl_id(gene_id):
            return gene_id
        
        # Check if numeric (likely Entrez)
        if gene_id.isdigit():
            return self.entrez_to_ensembl(gene_id)
        
        # Try as gene symbol
        return self.symbol_to_ensembl(gene_id)
    
    @staticmethod
    def is_valid_ensembl_id(gene_id: str) -> bool:
        """
        Check if a string is a valid Ensembl gene ID.
        
        Args:
            gene_id: Gene ID to validate
            
        Returns:
            True if valid Ensembl gene ID format
        """
        # Ensembl gene IDs follow pattern: ENSG followed by 11 digits
        pattern = r'^ENSG\d{11}$'
        return bool(re.match(pattern, gene_id))
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded mappings.
        
        Returns:
            Dictionary with mapping statistics
        """
        self._load_mappings()
        
        return {
            'total_symbols': len(self._symbol_to_ensembl),
            'total_entrez': len(self._entrez_to_ensembl),
            'total_ensembl': len(self._ensembl_to_symbol)
        }
    
    def validate_gene_list(self, gene_ids: List[str], 
                          id_type: str = "auto") -> Tuple[List[str], List[str]]:
        """
        Validate a list of gene IDs and return valid/invalid lists.
        
        Args:
            gene_ids: List of gene IDs to validate
            id_type: Type of input IDs
            
        Returns:
            Tuple of (valid_ensembl_ids, invalid_input_ids)
        """
        conversion_results = self.convert_to_ensembl(gene_ids, id_type)
        
        valid_ensembl = []
        invalid_input = []
        
        for input_id, ensembl_id in conversion_results.items():
            if ensembl_id is not None:
                valid_ensembl.append(ensembl_id)
            else:
                invalid_input.append(input_id)
        
        return valid_ensembl, invalid_input
    
    def standardize_gene_list(self, gene_ids: List[str], 
                             id_type: str = "auto",
                             remove_invalid: bool = True) -> List[str]:
        """
        Standardize a list of gene IDs to Ensembl format.
        
        Args:
            gene_ids: List of gene IDs to standardize
            id_type: Type of input IDs
            remove_invalid: Whether to remove IDs that cannot be converted
            
        Returns:
            List of Ensembl gene IDs
        """
        valid_ensembl, invalid_input = self.validate_gene_list(gene_ids, id_type)
        
        if invalid_input and not remove_invalid:
            raise GeneIDError(f"Could not convert {len(invalid_input)} gene IDs: {invalid_input[:10]}...")
        
        if invalid_input:
            logger.warning(f"Could not convert {len(invalid_input)} gene IDs to Ensembl format")
        
        logger.info(f"Standardized {len(valid_ensembl)} gene IDs to Ensembl format")
        return valid_ensembl


# Global gene mapper instance
_gene_mapper: Optional[EnsemblGeneMapper] = None


def get_gene_mapper() -> EnsemblGeneMapper:
    """Get the global gene mapper instance."""
    global _gene_mapper
    if _gene_mapper is None:
        _gene_mapper = EnsemblGeneMapper()
    return _gene_mapper


def convert_to_ensembl(gene_ids: Union[str, List[str]], 
                      id_type: str = "auto") -> Dict[str, Optional[str]]:
    """
    Convenience function to convert gene IDs to Ensembl format.
    
    Args:
        gene_ids: Gene ID(s) to convert
        id_type: Type of input IDs
        
    Returns:
        Dictionary mapping input IDs to Ensembl IDs
    """
    return get_gene_mapper().convert_to_ensembl(gene_ids, id_type)


def standardize_gene_list(gene_ids: List[str], 
                         id_type: str = "auto",
                         remove_invalid: bool = True) -> List[str]:
    """
    Convenience function to standardize gene IDs to Ensembl format.
    
    Args:
        gene_ids: List of gene IDs to standardize
        id_type: Type of input IDs
        remove_invalid: Whether to remove invalid IDs
        
    Returns:
        List of Ensembl gene IDs
    """
    return get_gene_mapper().standardize_gene_list(gene_ids, id_type, remove_invalid)


def is_valid_ensembl_id(gene_id: str) -> bool:
    """
    Convenience function to check if a gene ID is valid Ensembl format.
    
    Args:
        gene_id: Gene ID to validate
        
    Returns:
        True if valid Ensembl gene ID
    """
    return EnsemblGeneMapper.is_valid_ensembl_id(gene_id)
