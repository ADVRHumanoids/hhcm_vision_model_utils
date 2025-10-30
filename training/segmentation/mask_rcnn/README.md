# Mask R-CNN Training and Evaluation

This directory contains comprehensive tools for training and evaluating Mask R-CNN models for instance segmentation tasks. The pipeline includes automated hyperparameter optimization, comprehensive evaluation metrics, and extensive visualization capabilities.

## Overview

The training pipeline provides:

```
Dataset Loading → Model Creation → Training with Optuna → Hyperparameter Optimization → Best Model Selection
                     ↓
               Evaluation → Deployment
```

## Directory Structure

```
mask_rcnn_training/
├── README.md                          # This file
├── hyperparam_config.yaml             # Hyperparameter configuration file
├── train.py                           # Main training script with Optuna optimization
├── eval.py                            # Model evaluation and testing script
└── utils.py                           # Training utilities and helper functions
```

## Scripts Overview

### 1. `train.py`
**Purpose**: Main training script with automated hyperparameter optimization using Optuna

**Key Features**:
- **Hyperparameter Optimization**: Automated tuning using Optuna framework
- **Multiple Optimizers**: Support for Adam and SGD
- **Learning Rate Scheduling**: Step schedulers
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Model Checkpointing**: Automatic saving of best models
- **Comprehensive Logging**: Detailed training progress tracking

### 2. `eval.py`
**Purpose**: Evaluate trained models on test datasets

**Key Features**:
- **Multiple Metrics**: mAP, precision, recall, F1-score calculations
- **Visual Analysis**: Prediction visualizations with confidence scores
- **Confusion Matrices**: Class-wise performance analysis
- **Threshold Optimization**: Find optimal confidence thresholds
- **Batch Evaluation**: Process multiple test images efficiently

### 3. `utils.py`
**Purpose**: Utility functions for training and evaluation

**Key Features**:
- **Dataset Classes**: Custom PyTorch Dataset implementations
- **Data Loaders**: Optimized data loading with augmentation
- **Model Creation**: Mask R-CNN model builder functions
- **Metrics Calculation**: mAP, IoU, and other evaluation metrics
- **Visualization Tools**: Plotting and visualization utilities
- **Training Loops**: Modular training and validation functions

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
python train.py 
    --result-folder ../training_results 
    --dataset-folder ../tiled_dataset

# Train with custom hyperparameter file
python train.py 
    --result-folder ../training_results 
    --dataset-folder ../tiled_dataset 
    --hyperparam-config custom_hyperparams.yaml
```

### 2. Hyperparameter Optimization

```bash
# Extended hyperparameter search
python train.py 
    --result-folder ../training_results 
    --dataset-folder ../tiled_dataset 
    --num-trials 100 
    --patience 15 
    --save-top-k 10
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
python train.py \
    --result-folder <path> \              # Required: Path to results folder
    --dataset-folder <path> \             # Required: Path to dataset folder  
    --train-config <path> \               # Optional: Path to training YAML config
    --show-samples \                      # Show sample images before training
```

#### Hyperparameter Configuration

Hyperparameters are managed through a YAML configuration file (`train_config.yaml`). This provides:

- **Centralized Configuration**: All hyperparameters in one file
- **Reproducible Experiments**: Version-controlled parameter settings
- **Easy Customization**: Modify search spaces without code changes
- **Multiple Configurations**: Use different configs for different experiments

**Default Configuration Location**: `mask_rcnn_training/train_config.yaml`

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
    choices: ["adam", "sgd"]                    # Available optimizers
    
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
    --seed 42 \                           # Random seed
    --model-version 1                     # Mask R-CNN Pytorch model version
```

## Output Files

### Training Outputs

```
training_results/
├── analysis/
│   ├── optimization_history.png            # Optuna optimization progress
│   ├── parameter_importance.png            # Hyperparameter importance visualization
│   └── trial_results.csv                   # Summary of all trial results
├── report  
│   ├── categorical_analysis.png            # Categorical parameter analysis
│   ├── parameter_trends.png                # Parameter trends across trials
│   ├── report.json                         # Detailed report in JSON format
│   └── report.md                           # Human-readable training report
├── search_parameters.json                  # Search space and parameter settings
├── study.pkl                               # Serialized Optuna study object
├── top5_configurations.json                # Top 5 hyperparameter configurations
├── trial_0/ ... trial_N/   
│   ├── best_model_trial_X.pth              # Best model checkpoint for trial X (if available)
│   ├── confusion_matrix_trial_X.png        # Confusion matrix visualization
│   ├── confusion_matrix_trial_X_data.json  # Confusion matrix data
│   ├── epoch_metrics_trial_X.json          # Per-epoch metrics for trial X
│   ├── training_curves_trial_X.png         # Training/validation curves
│   └── trial_X_summary.json                # Summary of trial X
```
Each trial folder contains outputs specific to that hyperparameter configuration, including metrics, model checkpoints, and visualizations.

## Support and Resources

- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **Torchvision Models**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- **Optuna Documentation**: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- **COCO Evaluation**: [https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
