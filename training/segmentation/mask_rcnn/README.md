# Mask R-CNN Training and Evaluation

This directory contains comprehensive tools for training and evaluating Mask R-CNN models for instance segmentation tasks. The pipeline includes automated hyperparameter optimization using Optuna, comprehensive evaluation metrics, extensive visualization capabilities, and intelligent checkpoint management.


## Overview

The training pipeline provides:

```
Dataset Loading → Model Creation → Training with Optuna → Hyperparameter Optimization → Best Model Selection
                     ↓
               Evaluation → Deployment
```

## Directory Structure

```
mask_rcnn/
├── README.md                          # This file
├── train_config.yaml                  # Hyperparameter and training configuration file
├── tune.py                            # Main training script with Optuna optimization
├── eval.py                            # Model evaluation and testing script
└── utils.py                           # Training utilities and helper functions
```

## Scripts Overview

### 1. tune.py
**Purpose**: Main training script with automated hyperparameter optimization using Optuna

**Key Features**:
- **Hyperparameter Optimization**: Automated tuning using Optuna framework with PatientPruner
- **Top-K Checkpoint Management**: Saves only best K trials to conserve disk space
- **Multiple Optimizers**: Support for Adam, SGD, and AdamW
- **Learning Rate Scheduling**: Optional StepLR scheduler
- **Data Augmentation**: Comprehensive augmentation pipeline (color jitter, flips, grayscale)
- **Model Checkpointing**: Automatic saving of best models with CPU checkpoint fallback
- **Comprehensive Logging**: Detailed training progress tracking with per-epoch metrics
- **Confusion Matrices**: Automatic generation for top-performing models
- **YOLO Dataset Support**: Can load from both LabelMe and YOLO folder structures

### 2. eval.py
**Purpose**: Evaluate trained models on test datasets

**Key Features**:
- **Interactive Visualization**: Side-by-side ground truth vs predictions with ESC key navigation
- **Confidence Filtering**: Configurable confidence threshold for predictions
- **Visual Analysis**: Prediction visualizations with confidence scores and bounding boxes
- **Segmentation Mask Overlay**: Visual comparison of predicted and ground truth masks
- **Batch Evaluation**: Process multiple test images efficiently
- **YOLO Dataset Support**: Can evaluate on both LabelMe and YOLO folder structures
- **Random Sampling**: Reproducible evaluation with optional seed parameter

### 3. utils.py
**Purpose**: Utility functions for training and evaluation

**Key Features**:
- **Dataset Classes**: DefectDataset for LabelMe-format polygon annotations
- **Configuration Management**: YAML-based hyperparameter loading and saving
- **Model Creation**: Mask R-CNN model builder (versions 1 and 2)
- **Checkpoint Management**: CPU checkpoint saving for memory efficiency
- **Metrics Calculation**: Accuracy, precision, recall, F1-score (weighted and macro)
- **Visualization Tools**: Training curves, confusion matrices, parameter analysis plots
- **Training Loops**: Modular training and validation with early stopping
- **Reporting**: Comprehensive trial summaries and study analysis

## Prerequisites

### Required Python Packages

```bash
# Core ML and Computer Vision
pip install torch torchvision opencv-python pillow numpy

# Data Processing and Analysis
pip install pandas matplotlib seaborn scikit-learn

# Hyperparameter Optimization
pip install optuna

# Utilities and Visualization
pip install tqdm distinctipy cjm-pil-utils cjm-pytorch-utils cjm-psl-utils

# COCO evaluation tools
pip install pycocotools
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM for large datasets
- **Storage**: 50GB+ for models, datasets, and logs

## Quick Start

### 1. Basic Training

```bash
# Train with default hyperparameter configuration
python tune.py \
    --result-folder ../training_results \
    --dataset-folder ../tiled_dataset

# Train with custom configuration file
python tune.py \
    --result-folder ../training_results \
    --dataset-folder ../tiled_dataset \
    --train-config custom_config.yaml

# Show sample images before training
python tune.py \
    --result-folder ../training_results \
    --dataset-folder ../tiled_dataset \
    --show-samples

# Use YOLO dataset folder structure
python tune.py \
    --result-folder ../training_results \
    --dataset-folder ../yolo_dataset \
    --yolo
```

### 2. Hyperparameter Optimization

Hyperparameter settings are configured in `train_config.yaml`. Edit the file to adjust:

```yaml
training:
  trials: 100              # Number of optimization trials
  patience: 15             # Early stopping patience
  save_top_k: 10           # Number of top models to save
  num_workers: 8           # Data loading workers
```

### 3. Model Evaluation

```bash
# Evaluate best model
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --test-folder ../test_image \
    --threshold 0.5 \
    --images 20 \
    --model-version 1
```

## Detailed Usage

### Training Configuration

#### Command Line Arguments

The training script supports the following command line arguments:

```bash
python tune.py \
    --result-folder <path> \              # Required: Path to results folder
    --dataset-folder <path> \             # Required: Path to dataset folder
    --train-config <path> \               # Optional: Path to training YAML config
    --show-samples \                      # Show sample images before training
    --yolo                                # Use YOLO dataset folder structure
```

#### Hyperparameter Configuration

Hyperparameters are managed through a YAML configuration file (`train_config.yaml`). This provides:

- **Centralized Configuration**: All hyperparameters in one file
- **Reproducible Experiments**: Version-controlled parameter settings
- **Easy Customization**: Modify search spaces without code changes
- **Multiple Configurations**: Use different configs for different experiments

**Default Configuration Location**: `training/segmentation/mask_rcnn/train_config.yaml`

#### Hyperparameter Search Space

The YAML configuration defines the search space for optimization:

```yaml
hyperparameters:
  lr:                                    # Learning rate
    low: 1.0e-5                          # Minimum value
    high: 1.0e-2                         # Maximum value
    log: true                            # Use logarithmic sampling
    
  batch_size:                            # Batch size options
    choices: [2, 4, 8]                   # Available batch sizes
    
  optimizer:                             # Optimizer selection
    choices: ["adam"]                    # Available optimizers (adam, sgd, adamw)
    
  num_epochs:                            # Training epochs
    low: 10                              # Minimum epochs
    high: 100                            # Maximum epochs
    step: 10                             # Step size for sampling
    
  weight_decay:                          # L2 regularization
    low: 1.0e-6                          # Minimum weight decay
    high: 1.0e-2                         # Maximum weight decay
    log: true                            # Use logarithmic sampling
    
  use_grad_clip:                         # Gradient clipping
    choices: [true, false]               # Enable/disable options
    
  max_grad_norm:                         # Gradient clipping threshold
    low: 0.1                             # Minimum threshold
    high: 10.0                           # Maximum threshold
    log: true                            # Use logarithmic sampling
    
  use_scheduler:                         # Learning rate scheduling
    choices: [true, false]               # Enable/disable options
  lr_scheduler:
    step_size: 5                         # Int value
    gamma: 0.5                           # Float value
```

#### Training Settings

Additional training settings can be configured in the YAML file:

```yaml
training:
  trials: 50                         # Default number of optimization trials
  train_pct: 0.8                         # Default training data percentage
  patience: 10                           # Default early stopping patience
  save_top_k: 5                          # Default number of top trials to save
  seed: None                             # Default random seed
  num_workers: 4                         # Default number of workers 

model:
  version: 1                             # Mask R-CNN model version of Pytorch
```

### Basic Evaluation

```bash
python eval.py \
    --checkpoint models/best_model.pth \  # Path to trained model
    --test-folder ../test_image \         # Test dataset directory
    --threshold 0.5 \                     # Confidence threshold
    --images 50 \                         # Number of images to evaluate
    --seed 42 \                           # Random seed (optional)
    --model-version 1 \                   # Mask R-CNN Pytorch model version (1 or 2)
    --yolo                                # Use YOLO dataset folder structure
```

## Output Files

### Training Outputs

```
training_results/
├── analysis/
│   ├── optimization_history.png            # Optuna optimization progress
│   ├── parameter_importance.png            # Hyperparameter importance visualization
│   └── trial_results.csv                   # Summary of all trial results
├── report/
│   ├── categorical_analysis.png            # Categorical parameter analysis
│   ├── parameter_trends.png                # Parameter trends across trials
│   ├── report.json                         # Detailed report in JSON format
│   └── report.md                           # Human-readable training report with insights
├── study.pkl                               # Serialized Optuna study object
├── top5_configurations.json                # Top 5 hyperparameter configurations with ranks
├── training_config.yaml                    # Copy of configuration used for this run
└── trial_0/ ... trial_N/
    ├── best_model_trial_X.pth              # Best model checkpoint (only for top K trials)
    ├── confusion_matrix_trial_X.png        # Confusion matrix visualization
    ├── confusion_matrix_trial_X_data.json  # Confusion matrix raw data
    ├── epoch_metrics_trial_X.json          # Per-epoch metrics for trial X
    ├── training_curves_trial_X.png         # Training/validation loss curves
    └── trial_X_summary.json                # Trial summary with stopping reason
```

**Note**: Only the top K trials (default: 5) will have `best_model_trial_X.pth` checkpoints saved to conserve disk space. All trials retain their metrics, curves, and confusion matrices for analysis.

## Support and Resources

- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **Torchvision Models**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- **Optuna Documentation**: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- **COCO Evaluation**: [https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
