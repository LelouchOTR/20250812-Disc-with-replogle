# 🧬 Discrepancy VAE for Perturbation Analysis

Welcome! This project is all about making sense of large-scale, single-cell CRISPR screens. We use a specialized deep learning model to turn a massive, complex dataset into a clear, interpretable map of gene function.

## 🎯 Project Goal

Our main goal is to understand the impact of genetic perturbations: which genes matter, how strong their effects are, and how they work together in functional pathways.

## 🧠 How it Works: The Model

Think of the model as a **"smart biological detective"** that learns about your cells in three ways:

*   **🗺️ The Cell Map:** It first learns to draw a "map" where cells with similar biology are placed close together. This map is technically a 32-dimensional "latent space."

*   **🔬 The Perturbation Effect:** It then figures out how each genetic perturbation "moves" a cell on this map. It learns a "discrepancy vector"—an arrow pointing from the normal cell state to the perturbed state. The length and direction of the arrow tell you the strength and type of the effect.

*   **🕸️ The Biological Hint (Graph Integration):** To make the map more biologically meaningful, we provide the model with a gene-gene interaction network. The model's encoder is a **Graph Convolutional Network (GCN)** that processes this graph directly. This allows the model to learn from the connections between genes, encouraging genes that are functionally related to have similar representations. A **graph Laplacian regularization** term in the loss function further reinforces this, ensuring that connected genes are mapped closely together in the latent space.

## 🌐 The Gene-Gene Interaction Graph

The "biological hint" given to the model is a graph representing known functional relationships between genes. Here's how it's built:

1.  **Data Source:** We use the **Gene Ontology (GO)** database, a comprehensive resource of gene functions.
2.  **Term Filtering:** We filter GO terms to select those that are most informative, based on the number of genes they annotate and their specificity.
3.  **Similarity Calculation:** For every pair of genes, we calculate a **Jaccard similarity score** based on the GO terms they share. A high score means two genes are involved in many of the same biological processes.
4.  **Graph Construction:** We create a network where each gene is a node. An edge is drawn between two genes if their similarity score is above a predefined threshold. This results in a graph where connected genes are likely to be functionally related.

This graph is then fed into the GCN encoder of the DiscrepancyVAE, providing a strong biological prior that guides the model's learning process.

### How Our Approach Compares to GEARS: A Tale of Two Maps

Both our project and the well-known GEARS model use gene networks to understand perturbation effects, but we do so in fundamentally different ways. Think of it as the difference between using a detailed, static atlas versus a dynamic, real-time GPS.

**Our Approach: The Comprehensive Atlas**

In this project, we first build a single, comprehensive "atlas" of the gene world. This graph is a rich, static map of all known functional relationships between genes, based on decades of curated biological knowledge from the Gene Ontology.

Our Discrepancy VAE model then takes the single-cell data from our experiment and "plots" it onto this master map. By using a **Graph Convolutional Network (GCN)**, the model is forced to consider the known relationships between genes as it learns. This provides a stable, global context, allowing the model to learn a single, consistent "discrepancy vector" for each perturbation. The result is a highly interpretable and robust view of how each gene fits into the broader landscape of cellular biology.

**The GEARS Approach: The Real-Time GPS**

GEARS, in contrast, acts more like a "GPS" for each individual cell. Instead of starting with one big map, it constructs a unique, smaller graph for *every single cell* in the dataset. This graph represents the specific gene network that is active in that cell at that moment.

It then models a perturbation as a signal that propagates through this individualized, dynamic network. This makes GEARS exceptionally powerful for predicting the precise outcome of a perturbation within the unique context of a single cell.

**The Core Difference**

*   **Our Method:** Uses one **static, knowledge-rich graph** to provide a stable biological "scaffold" for interpreting experimental data. It excels at creating a robust and interpretable global map of gene function.
*   **GEARS' Method:** Uses many **dynamic, cell-specific graphs** to predict the effects of perturbations in a highly context-aware manner. It excels at predictive accuracy at the single-cell level.

By providing our Discrepancy VAE with a single, stable biological atlas, we guide it to learn the fundamental, context-independent roles of genes, which is the primary goal of this project.

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
