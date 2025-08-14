#!/usr/bin/env python3
"""
Gene Ontology graph generation pipeline for single-cell perturbation analysis.

This script downloads human GO gene annotations from Gene Ontology Consortium,
builds gene adjacency graph based on GO term relationships, applies configurable
term size and depth filters, caches the graph structure and metadata in data/graphs/,
and provides utilities for graph loading and validation.
"""

import os
import sys
import logging
import argparse
import pickle
import json
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
import requests
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.random_seed import set_global_seed
from src.utils.gene_ids import get_gene_mapper

logger = logging.getLogger(__name__)


class GraphGenerationError(Exception):
    """Custom exception for graph generation errors."""
    pass


class GOAnnotationParser:
    """
    Parser for Gene Ontology annotation files (GAF format).
    """

    def __init__(self):
        self.annotations = defaultdict(set)
        self.term_genes = defaultdict(set)
        self.term_info = {}

    def parse_gaf_file(self, gaf_path: Path, evidence_codes: Optional[Set[str]] = None,
                       exclude_codes: Optional[Set[str]] = None) -> None:
        logger.info(f"Parsing GAF file: {gaf_path}")

        if exclude_codes is None:
            exclude_codes = set()

        annotations_count = 0

        if gaf_path.suffix == '.gz':
            file_handle = gzip.open(gaf_path, 'rt')
        else:
            file_handle = open(gaf_path, 'r')

        try:
            with file_handle as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('!'):
                        continue

                    fields = line.strip().split('\t')
                    if len(fields) < 15:
                        continue

                    db_object_symbol = fields[2]
                    go_id = fields[4]
                    evidence_code = fields[6]
                    aspect = fields[8]
                    db_object_name = fields[9]
                    db_object_synonym = fields[10]
                    db_object_type = fields[11]
                    taxon = fields[12]

                    if 'taxon:9606' not in taxon:
                        continue

                    if db_object_type not in ['gene', 'protein']:
                        continue

                    if evidence_codes and evidence_code not in evidence_codes:
                        continue

                    if evidence_code in exclude_codes:
                        continue

                    self.annotations[db_object_symbol].add(go_id)
                    self.term_genes[go_id].add(db_object_symbol)

                    if go_id not in self.term_info:
                        self.term_info[go_id] = {
                            'aspect': aspect,
                            'evidence_codes': set()
                        }
                    self.term_info[go_id]['evidence_codes'].add(evidence_code)

                    annotations_count += 1

                    if line_num % 100000 == 0:
                        logger.debug(f"Processed {line_num} lines, {annotations_count} annotations")

        finally:
            file_handle.close()

        logger.info(f"Parsed {annotations_count} annotations for {len(self.annotations)} genes")
        logger.info(f"Found {len(self.term_genes)} unique GO terms")


class GOOntologyParser:
    """
    Parser for Gene Ontology OBO format ontology files.
    """

    def __init__(self):
        self.terms = {}
        self.relationships = defaultdict(set)
        self.children = defaultdict(set)

    def parse_obo_file(self, obo_path: Path) -> None:
        logger.info(f"Parsing OBO file: {obo_path}")

        current_term = None

        if obo_path.suffix == '.gz':
            file_handle = gzip.open(obo_path, 'rt')
        else:
            file_handle = open(obo_path, 'r')

        try:
            with file_handle as f:
                for line in f:
                    line = line.strip()

                    if line == '[Term]':
                        current_term = {}
                        continue

                    if line == '' or line.startswith('['):
                        if current_term and 'id' in current_term:
                            term_id = current_term['id']
                            self.terms[term_id] = current_term
                        current_term = None
                        continue

                    if current_term is None:
                        continue

                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip()

                        if key == 'id':
                            current_term['id'] = value
                        elif key == 'name':
                            current_term['name'] = value
                        elif key == 'namespace':
                            current_term['namespace'] = value
                        elif key == 'def':
                            current_term['definition'] = value
                        elif key == 'is_a':
                            parent_id = value.split('!')[0].strip()
                            if 'parents' not in current_term:
                                current_term['parents'] = []
                            current_term['parents'].append(parent_id)
                            self.relationships[current_term['id']].add(parent_id)
                            self.children[parent_id].add(current_term['id'])
                        elif key == 'relationship':
                            parts = value.split()
                            if len(parts) >= 2:
                                rel_type = parts[0]
                                parent_id = parts[1]
                                if 'relationships' not in current_term:
                                    current_term['relationships'] = []
                                current_term['relationships'].append((rel_type, parent_id))

                                self.relationships[current_term['id']].add(parent_id)
                                self.children[parent_id].add(current_term['id'])
                        elif key == 'is_obsolete':
                            current_term['obsolete'] = value.lower() == 'true'

        finally:
            file_handle.close()

        logger.info(f"Parsed {len(self.terms)} GO terms")

        self._calculate_depths()

    def _calculate_depths(self) -> None:
        logger.info("Calculating term depths...")

        root_terms = set()
        for term_id, term_info in self.terms.items():
            if not self.relationships.get(term_id):
                root_terms.add(term_id)

        depths = {}
        queue = [(term_id, 0) for term_id in root_terms]

        while queue:
            term_id, depth = queue.pop(0)

            if term_id in depths:
                depths[term_id] = min(depths[term_id], depth)
            else:
                depths[term_id] = depth

                for child_id in self.children.get(term_id, set()):
                    queue.append((child_id, depth + 1))

        for term_id, depth in depths.items():
            if term_id in self.terms:
                self.terms[term_id]['depth'] = depth

        logger.info(f"Calculated depths for {len(depths)} terms")


class GeneOntologyGraphBuilder:
    """
    Builder for gene adjacency graphs based on Gene Ontology annotations.
    """

    def __init__(self, config: Dict, cache_dir: Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.go_config = config.get('go_data_source', {})
        self.filter_config = config.get('go_filtering', {})
        self.adj_config = config.get('adjacency', {})

        self.annotation_parser = GOAnnotationParser()
        self.ontology_parser = GOOntologyParser()

        self.gene_mapper = get_gene_mapper()

        logger.info(f"Initialized GeneOntologyGraphBuilder with cache dir: {cache_dir}")

    def download_go_data(self, force_refresh: bool = False) -> Tuple[Path, Path]:
        annotation_url = self.go_config.get(
            'annotation_url',
            'http://current.geneontology.org/annotations/goa_human.gaf.gz'
        )
        ontology_url = self.go_config.get(
            'ontology_url',
            'http://current.geneontology.org/ontology/go-basic.obo'
        )

        cache_expiry_days = self.go_config.get('cache_expiry_days', 30)

        annotation_file = self.cache_dir / 'goa_human.gaf.gz'
        if self._should_download(annotation_file, cache_expiry_days, force_refresh):
            logger.info(f"Downloading GO annotations from: {annotation_url}")
            self._download_file(annotation_url, annotation_file)
        else:
            logger.info(f"Using cached GO annotations: {annotation_file}")

        ontology_file = self.cache_dir / 'go-basic.obo'
        if self._should_download(ontology_file, cache_expiry_days, force_refresh):
            logger.info(f"Downloading GO ontology from: {ontology_url}")
            self._download_file(ontology_url, ontology_file)
        else:
            logger.info(f"Using cached GO ontology: {ontology_file}")

        return annotation_file, ontology_file

    def _should_download(self, file_path: Path, cache_expiry_days: int, force_refresh: bool) -> bool:
        if force_refresh:
            return True

        if not file_path.exists():
            return True

        file_age_days = (datetime.now().timestamp() - file_path.stat().st_mtime) / (24 * 3600)
        return file_age_days > cache_expiry_days

    def _download_file(self, url: str, file_path: Path) -> None:
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded: {file_path}")

        except Exception as e:
            raise GraphGenerationError(f"Failed to download {url}: {e}")

    def parse_go_data(self, annotation_file: Path, ontology_file: Path) -> None:
        self.ontology_parser.parse_obo_file(ontology_file)

        evidence_codes = set(self.filter_config.get('evidence_codes', []))
        exclude_codes = set(self.filter_config.get('exclude_evidence_codes', []))

        if not evidence_codes:
            evidence_codes = None

        self.annotation_parser.parse_gaf_file(
            annotation_file,
            evidence_codes=evidence_codes,
            exclude_codes=exclude_codes
        )

    def filter_go_terms(self) -> Set[str]:
        logger.info("Filtering GO terms...")

        min_term_size = self.filter_config.get('min_term_size', 5)
        max_term_size = self.filter_config.get('max_term_size', 500)
        min_depth = self.filter_config.get('min_depth', 2)
        max_depth = self.filter_config.get('max_depth', 10)
        include_namespaces = set(self.filter_config.get('include_namespaces', []))

        valid_terms = set()

        for term_id, genes in self.annotation_parser.term_genes.items():
            term_size = len(genes)
            if term_size < min_term_size or term_size > max_term_size:
                continue

            if term_id not in self.ontology_parser.terms:
                continue

            term_info = self.ontology_parser.terms[term_id]

            depth = term_info.get('depth', 0)
            if depth < min_depth or depth > max_depth:
                continue

            if include_namespaces:
                namespace = term_info.get('namespace', '')
                if namespace not in include_namespaces:
                    continue

            if term_info.get('obsolete', False):
                continue

            valid_terms.add(term_id)

        logger.info(f"Filtered to {len(valid_terms)} valid GO terms")
        return valid_terms

    def build_gene_adjacency_graph(self, valid_terms: Set[str]) -> nx.Graph:
        logger.info("Building gene adjacency graph...")

        filtered_annotations = {}
        all_genes = set()

        for gene, terms in self.annotation_parser.annotations.items():
            valid_gene_terms = terms.intersection(valid_terms)
            if valid_gene_terms:
                filtered_annotations[gene] = valid_gene_terms
                all_genes.add(gene)

        logger.info(f"Building graph for {len(all_genes)} genes with {len(valid_terms)} GO terms")

        gene_list = list(all_genes)
        ensembl_mapping = self.gene_mapper.convert_to_ensembl(gene_list, id_type="auto")

        valid_genes = []
        gene_to_ensembl = {}

        for gene, ensembl_id in ensembl_mapping.items():
            if ensembl_id:
                valid_genes.append(gene)
                gene_to_ensembl[gene] = ensembl_id

        logger.info(f"Mapped {len(valid_genes)} genes to Ensembl IDs")

        method = self.adj_config.get('method', 'jaccard')
        threshold = self.adj_config.get('threshold', 0.1)

        logger.info(f"Computing {method} similarity with threshold {threshold}")

        gene_term_matrix = self._create_gene_term_matrix(valid_genes, filtered_annotations, valid_terms)

        if method == 'jaccard':
            similarity_matrix = self._compute_jaccard_similarity(gene_term_matrix)
        elif method == 'overlap':
            similarity_matrix = self._compute_overlap_similarity(gene_term_matrix)
        elif method == 'cosine':
            similarity_matrix = self._compute_cosine_similarity(gene_term_matrix)
        else:
            raise GraphGenerationError(f"Unknown similarity method: {method}")

        graph = nx.Graph()

        for gene in valid_genes:
            ensembl_id = gene_to_ensembl[gene]
            graph.add_node(ensembl_id, original_id=gene)

        n_genes = len(valid_genes)
        edges_added = 0

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                similarity = similarity_matrix[i, j]

                if similarity >= threshold:
                    gene1_ensembl = gene_to_ensembl[valid_genes[i]]
                    gene2_ensembl = gene_to_ensembl[valid_genes[j]]

                    graph.add_edge(gene1_ensembl, gene2_ensembl, weight=similarity)
                    edges_added += 1

        logger.info(f"Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

        graph = self._filter_graph(graph)

        return graph

    def _create_gene_term_matrix(self, genes: List[str], annotations: Dict[str, Set[str]],
                                 terms: Set[str]) -> np.ndarray:
        term_list = sorted(list(terms))
        term_to_idx = {term: idx for idx, term in enumerate(term_list)}

        matrix = np.zeros((len(genes), len(terms)), dtype=bool)

        for gene_idx, gene in enumerate(genes):
            gene_terms = annotations.get(gene, set())
            for term in gene_terms:
                if term in term_to_idx:
                    term_idx = term_to_idx[term]
                    matrix[gene_idx, term_idx] = True

        return matrix

    def _compute_jaccard_similarity(self, matrix: np.ndarray) -> np.ndarray:
        n_genes = matrix.shape[0]
        similarity_matrix = np.zeros((n_genes, n_genes))

        for i in range(n_genes):
            for j in range(i, n_genes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = np.sum(matrix[i] & matrix[j])
                    union = np.sum(matrix[i] | matrix[j])

                    if union > 0:
                        jaccard = intersection / union
                    else:
                        jaccard = 0.0

                    similarity_matrix[i, j] = jaccard
                    similarity_matrix[j, i] = jaccard

        return similarity_matrix

    def _compute_overlap_similarity(self, matrix: np.ndarray) -> np.ndarray:
        n_genes = matrix.shape[0]
        similarity_matrix = np.zeros((n_genes, n_genes))

        for i in range(n_genes):
            for j in range(i, n_genes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = np.sum(matrix[i] & matrix[j])
                    min_size = min(np.sum(matrix[i]), np.sum(matrix[j]))

                    if min_size > 0:
                        overlap = intersection / min_size
                    else:
                        overlap = 0.0

                    similarity_matrix[i, j] = overlap
                    similarity_matrix[j, i] = overlap

        return similarity_matrix

    def _compute_cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(matrix.astype(float))

    def _filter_graph(self, graph: nx.Graph) -> nx.Graph:
        if self.adj_config.get('remove_isolated_nodes', True):
            isolated_nodes = list(nx.isolates(graph))
            graph.remove_nodes_from(isolated_nodes)
            logger.info(f"Removed {len(isolated_nodes)} isolated nodes")

        min_degree = self.adj_config.get('min_degree', 1)
        if min_degree > 1:
            low_degree_nodes = [node for node, degree in graph.degree() if degree < min_degree]
            graph.remove_nodes_from(low_degree_nodes)
            logger.info(f"Removed {len(low_degree_nodes)} nodes with degree < {min_degree}")

        max_edges_per_gene = self.adj_config.get('max_edges_per_gene', 100)
        if max_edges_per_gene > 0:
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                if len(neighbors) > max_edges_per_gene:
                    edge_weights = [(neighbor, graph[node][neighbor]['weight']) for neighbor in neighbors]
                    edge_weights.sort(key=lambda x: x[1], reverse=True)

                    edges_to_remove = edge_weights[max_edges_per_gene:]
                    for neighbor, _ in edges_to_remove:
                        graph.remove_edge(node, neighbor)

        logger.info(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph

    def save_graph(self, graph: nx.Graph, output_dir: Path) -> None:
        logger.info("Saving graph and metadata...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formats = self.config.get('output', {}).get('save_formats', ['pickle', 'graphml'])

        for format_name in formats:
            if format_name == 'pickle':
                graph_file = output_dir / 'gene_adjacency_graph.pkl'
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
                logger.info(f"Saved graph as pickle: {graph_file}")

            elif format_name == 'graphml':
                graph_file = output_dir / 'gene_adjacency_graph.graphml'
                nx.write_graphml(graph, graph_file)
                logger.info(f"Saved graph as GraphML: {graph_file}")

            elif format_name == 'gml':
                graph_file = output_dir / 'gene_adjacency_graph.gml'
                nx.write_gml(graph, graph_file)
                logger.info(f"Saved graph as GML: {graph_file}")

            elif format_name == 'adjacency':
                adjacency_matrix = nx.adjacency_matrix(graph)
                adjacency_file = output_dir / 'adjacency_matrix.npz'
                sparse.save_npz(adjacency_file, adjacency_matrix)

                nodes_file = output_dir / 'node_mapping.json'
                node_mapping = {i: node for i, node in enumerate(graph.nodes())}
                with open(nodes_file, 'w') as f:
                    json.dump(node_mapping, f, indent=2)

                logger.info(f"Saved adjacency matrix: {adjacency_file}")
                logger.info(f"Saved node mapping: {nodes_file}")

        metadata = {
            'creation_date': datetime.now().isoformat(),
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'config_used': self.config,
            'go_terms_used': len(self.annotation_parser.term_genes),
            'genes_mapped': len([node for node in graph.nodes()]),
            'similarity_method': self.adj_config.get('method', 'jaccard'),
            'similarity_threshold': self.adj_config.get('threshold', 0.1),
            'graph_statistics': {
                'density': nx.density(graph),
                'average_degree': sum(
                    dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
                'number_of_components': nx.number_connected_components(graph),
                'largest_component_size': len(
                    max(nx.connected_components(graph), key=len)) if graph.number_of_nodes() > 0 else 0
            }
        }

        metadata_file = output_dir / 'graph_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {metadata_file}")


def load_graph(graph_dir: Path, format_name: str = 'pickle') -> nx.Graph:
    graph_dir = Path(graph_dir)

    if format_name == 'pickle':
        graph_file = graph_dir / 'gene_adjacency_graph.pkl'
        with open(graph_file, 'rb') as f:
            return pickle.load(f)

    elif format_name == 'graphml':
        graph_file = graph_dir / 'gene_adjacency_graph.graphml'
        return nx.read_graphml(graph_file)

    elif format_name == 'gml':
        graph_file = graph_dir / 'gene_adjacency_graph.gml'
        return nx.read_gml(graph_file)

    else:
        raise ValueError(f"Unsupported format: {format_name}")


def validate_graph(graph: nx.Graph) -> Dict[str, Union[bool, str, int, float]]:
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    if graph.number_of_nodes() == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Graph has no nodes")
        return validation_results

    if graph.number_of_edges() == 0:
        validation_results['warnings'].append("Graph has no edges")

    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        validation_results['warnings'].append(f"Graph has {len(self_loops)} self-loops")

    validation_results['statistics'] = {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'n_components': nx.number_connected_components(graph),
        'largest_component_size': len(max(nx.connected_components(graph), key=len)),
        'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'has_self_loops': len(self_loops) > 0
    }

    logger.info("Graph validation completed")
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Generate Gene Ontology gene adjacency graph")
    parser.add_argument("--config", type=str, default="graph_config", help="Configuration file name")
    parser.add_argument("--output-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/graphs",
                        help="Output directory")
    parser.add_argument("--log-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/logs",
                        help="Log directory")
    parser.add_argument("--cache-dir", type=str, default="/data/gidb/shared/results/tmp/replogle/cache/go",
                        help="Cache directory")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached data")
    parser.add_argument("--validate", action="store_true", help="Validate generated graph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    
    log_dir = Path(args.log_dir) 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / '03_graphs.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    logger.info("Starting Gene Ontology graph generation pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")

    try:
        builder = GeneOntologyGraphBuilder(config, args.cache_dir)

        annotation_file, ontology_file = builder.download_go_data(args.force_refresh)

        builder.parse_go_data(annotation_file, ontology_file)

        valid_terms = builder.filter_go_terms()

        graph = builder.build_gene_adjacency_graph(valid_terms)

        builder.save_graph(graph, args.output_dir)

        if args.validate:
            validation_results = validate_graph(graph)
            logger.info(f"Graph validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")

            if validation_results['errors']:
                for error in validation_results['errors']:
                    logger.error(f"Validation error: {error}")

            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    logger.warning(f"Validation warning: {warning}")

        logger.info("Gene Ontology graph generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
