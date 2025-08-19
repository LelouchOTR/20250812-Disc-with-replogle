# 🧬 Discrepancy VAE for Perturbation Analysis

Welcome! This project is all about making sense of large-scale, single-cell CRISPR screens. We use a specialized deep learning model to turn a massive, complex dataset into a clear, interpretable map of gene function.

## 🎯 Project Goal

Our main goal is to understand the impact of genetic perturbations: which genes matter, how strong their effects are, and how they work together in functional pathways.

## 🧠 A GEARS-inspired, Graph-based Approach

This project leverages a graph-based variational autoencoder (VAE) to decipher the complex transcriptional effects of genetic perturbations. Our methodology is heavily inspired by the GEARS framework, where each cell is modeled as an individual graph, allowing the model to learn perturbation effects within the context of each cell's unique transcriptional state.

### Graph Construction: From Blueprint to Cell-Specific Instances

Our graph construction is a two-step process that combines a static, knowledge-based blueprint with dynamic, cell-specific data:

1.  **The Biological Blueprint (Gene Ontology Network):** We first construct a high-quality, global gene-gene interaction network using the Gene Ontology (GO). In this network, genes are nodes, and an edge between them signifies a shared biological function, calculated using Jaccard similarity on their GO term annotations. This network serves as a robust, static "blueprint" of known biological relationships.

2.  **Cell-Specific Graphs:** For each cell in our single-cell dataset, we then create a unique graph instance. The topology of this graph—its connections or `edge_index`—is inherited directly from our GO blueprint. However, the node features (`x`) are the cell's actual gene expression values. This creates a powerful representation where each cell is a graph, grounded in biological reality but specific to its own state.

### The Discrepancy VAE Model

The core of our project is a Discrepancy VAE that learns from this rich, graph-based data:

-   **Graph-Aware Encoder:** The model's encoder is a **Graph Convolutional Network (GCN)**. For each cell, it processes the graph of gene expression values, respecting the underlying biological network structure. This produces a low-dimensional "latent representation" of each cell that is inherently graph-aware.

-   **Discrepancy Learning:** The VAE is trained to map these latent representations to a space where the vector difference between a perturbed cell and its control counterparts—the "discrepancy vector"—accurately captures the magnitude and direction of the perturbation's effect.

By modeling each cell as a graph, our Discrepancy VAE learns a nuanced, systems-level understanding of gene function and perturbation response, moving beyond simple gene expression to capture the interplay between genes in a network context.

## 📊 Interpreting the Output: The UMAP Plot

The main result is a UMAP plot, which is a 2D picture of the cell map the model learned. Each dot is a cell, and it gives you a bird's-eye view of your whole experiment.

![Enhanced UMAP Plot](./outputs/images/umap_enhanced.png)

**How to read the map:**
*   **Control Hub:** Your normal, unperturbed cells will form a large central cluster.
*   **Perturbation Clusters:** Cells with the same perturbation will group together.
    *   Clusters **far** from the hub had a **strong effect**.
    *   Clusters **close** to the hub had a **weak effect**.
    *   Clusters that are **neighbors** on the map likely affect **similar biological pathways**.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/LelouchOTR/20250812-Disc-with-replogle.git
cd 20250812-Disc-with-replogle
```

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate disc-with-replogle
```

### 3. Configure Output Paths
Before running, you can configure your main output directory in `configs/pipeline_config.yaml`. By default, all results are saved to the `outputs/` directory.

### 4. Run the Pipeline
```bash
python run_pipeline.py all
```

## 🛠️ Pipeline Usage

You can also run individual steps of the pipeline: `ingest`, `process`, `graphs`, `train`, `evaluate`.

```bash
# Run a specific step
python run_pipeline.py <step_name>
```

## 📦 Output Directory Structure

The pipeline generates a highly structured output directory for each run, typically within `outputs/`. This ensures that all results, logs, and models are organized and easy to find.

```
run_root/
├── run_metadata.json      # Run metadata (config, seed)
├── raw/                   # Raw input data
├── processed/             # Processed data (train/val/test splits)
├── graphs/                # Gene interaction graphs
├── models/                # Trained model checkpoints
├── evaluation/            # Evaluation results and plots
├── logs/                  # Log files from each pipeline step
└── cache/                 # Cached data
```

## ⚙️ Configuration

The pipeline is configured using YAML files in the `configs/` directory. You can modify these files to change parameters for data processing, graph construction, and model hyperparameters.

## 💻 HPC Execution

For large datasets, you can submit the pipeline to a Slurm cluster using the provided script. You may need to customize `scripts/slurm_pipeline.sh` for your cluster.

### Verifying the Environment on a Compute Node

Before launching the full pipeline, it's good practice to verify that your Conda environment works correctly on the cluster's compute nodes. We've included a script to help with this.

```bash
# Submit the verification job
sbatch scripts/setup_and_verify_environment.slurm
```
After the job runs, check the log file it creates in the `logs/` directory. It will confirm whether PyTorch can successfully detect and use the GPU on the node.

### Running the Main Pipeline
To submit the main analysis pipeline to Slurm, use the following command:
```bash
sbatch scripts/slurm_pipeline.sh
```
