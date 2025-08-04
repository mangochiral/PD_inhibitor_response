# Multi-Modal Cancer Treatment Response Prediction

A PyTorch-based deep learning framework for predicting cancer treatment response using multi-modal data integration of RNA sequencing and clinical features.

## Overview

This project implements a neural network that combines RNA expression data with clinical biomarkers to predict patient response to cancer treatment. The model uses a dual-branch architecture that processes genomic and clinical data separately before fusion for final prediction.

## Features

- **Multi-modal data integration**: Combines RNA-seq gene expression with clinical features
- **Deep learning architecture**: Dual-branch neural network with batch normalization and dropout
- **Model interpretability**: SHAP analysis for feature importance
- **Dimensionality reduction**: UMAP visualization of learned RNA embeddings
- **Comprehensive evaluation**: ROC-AUC scoring and performance metrics

## Architecture

The model consists of three main components:

1. **RNA Branch**: Processes gene expression data (genes → 512 → 256 → 64 dimensions)
2. **Clinical Branch**: Processes clinical features (features → 32 → 16 dimensions)  
3. **Classifier**: Fuses both branches for binary treatment response prediction (80 → 32 → 1)

## Dataset

The project processes data from **GSE91061**, which includes:
- RNA sequencing data from cancer patients
- Clinical metadata (visit timing, cytolytic scores)
- Treatment response labels (PRCR/SD/PD classifications)

### Data Files Required

Place these files in your data directory:
- `GSE91061_series_matrix.txt` - Sample metadata
- `GSE91061_BMS038109Sample_Cytolytic_Score_20161026.txt` - Cytolytic scores
- `Human.GRCh38.p13.annot.tsv` - Gene annotations
- `GSE91061_raw_counts_GRCh38.p13_NCBI.tsv` - Raw RNA-seq counts

## Installation

### Dependencies

```bash
pip install torch
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install shap
pip install umap-learn
```


## Usage

1. **Run the main script:**
   ```bash
   python main.py
   ```

2. **Enter your data directory when prompted:**
   ```
   Enter the data directory: /path/to/your/data
   Enter the plot directory: /path/to/your/plot
   ```

3. **The script will automatically:**
   - Process and normalize the data
   - Train the multi-modal neural network
   - Generate SHAP interpretability plots
   - Create UMAP visualizations
   - Save the trained model as `model.pth`

## Data Processing Pipeline

1. **Metadata Extraction**: Parses GEO series matrix for sample information
2. **Gene Filtering**: Removes low-expression genes (< 10 counts in < 10 samples)
3. **Log Normalization**: Applies log1p transformation to count data
4. **Binary Encoding**: Converts treatment responses to binary labels (PRCR=1, SD/PD=0)
5. **Data Alignment**: Ensures samples match across RNA and clinical datasets

## Model Training

- **Architecture**: Dual-branch neural network with batch normalization
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with learning rate 1e-3
- **Training**: 100 epochs with 80/20 train/test split
- **Batch Size**: 16 samples per batch

## Outputs

The pipeline generates several outputs:

### Model Files
- `model.pth` - Trained PyTorch model weights

### Visualizations
- `shap_rna.png` - SHAP feature importance for gene expression
- `shap_clinical.png` - SHAP feature importance for clinical features  
- `umap_response.png` - UMAP visualization of RNA embeddings colored by response

### Data Files
- `metadata_GSE.csv` - Processed metadata
- Console output with training loss and test AUC per epoch

## Model Performance

The model outputs:
- **Training loss** per epoch
- **Test AUC** (Area Under ROC Curve) for binary classification
- **Prediction vs ground truth** comparison dataframe

## Interpretability Features

### SHAP Analysis
- Identifies most important genes for treatment response prediction
- Shows clinical feature contributions to model decisions
- Generates summary plots for both modalities

### UMAP Visualization  
- 2D embedding of learned RNA representations
- Colored by treatment response for pattern identification
- Helps visualize learned feature separability

## Key Classes and Functions

### `MultiModalDataset` (modelutils.py)
- PyTorch Dataset class for loading RNA and clinical data
- Handles tensor conversion and data alignment

### `MultiModalNet` (modelutils.py)  
- Dual-branch neural network architecture
- Separate processing for RNA and clinical modalities
- Fusion layer for final prediction

### `Makemetadat` (makemeta.py)
- Parses GEO series matrix files
- Extracts sample metadata and characteristics
- Handles visit timing and response classification

## Hardware Requirements

- **GPU Recommended**: CUDA-compatible GPU for faster training
- **Memory**: Sufficient RAM for loading full gene expression matrices
- **Storage**: Space for model checkpoints and visualization outputs

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all Python files are in the same directory
2. **File Not Found**: Check that data directory path is correct and contains all required files
3. **Memory Issues**: Reduce batch size if encountering out-of-memory errors
4. **CUDA Errors**: Model automatically falls back to CPU if GPU unavailable



## Data Source
- **Dataset**: [GSE91061](https://www-ncbi-nlm-nih-gov.proxy.ulib.uits.iu.edu/geo/query/acc.cgi?acc=GSE91061)
- **Publication**: [Tumor and Microenvironment Evolution during Immunotherapy with Nivolumab](https://www-sciencedirect-com.proxy.ulib.uits.iu.edu/science/article/pii/S0092867417311224?via%3Dihub)
