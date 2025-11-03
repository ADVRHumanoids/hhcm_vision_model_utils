#!/usr/bin/env python3
"""
Utility functions for Mask R-CNN training and evaluation.

This module provides a comprehensive set of utility functions and classes for training and
evaluating Mask R-CNN models, including:

- Configuration management (loading/saving YAML configs)
- Dataset classes (DefectDataset for LabelMe format)
- Model creation and checkpoint management
- Training and validation loops with metrics
- Visualization utilities (bounding boxes, masks, training curves)
- Reporting functions (confusion matrices, parameter analysis)
- Memory management and cleanup utilities

Modified by: Alessio Lovato, 03-11-2025
"""

import os
import json
import yaml
import warnings
import traceback
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional
from PIL import Image, ImageDraw
import torchvision
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import seaborn as sns
from functools import partial

# Import utility functions
from cjm_psl_utils.core import download_file


# --------------------
# Hyperparameter Configuration Loading
# --------------------

def load_training_config(config_path: str = None):
    """
    Load hyperparameter configuration from YAML file.
    
    Args:
        config_path: Path to hyperparameter config YAML file
                    If None, looks for hyperparam_config.yaml in current directory
    
    Returns:
        dict: Configuration dictionary or None if loading fails
    """

    if config_path is None or not os.path.exists(config_path):
        warnings.warn(f"Hyperparameter config file not found at {config_path}. Using default configuration.")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        warnings.warn(f"Error loading hyperparameter config: {e}.")
        return None

def get_default_config():
    """
    Get default configuration dictionary for training.

    Returns:
        dict: Default configuration with training settings, model version,
              and hyperparameter search space
    """
    return {
        'training': {
            'trials': 50,
            'train_pct': 0.8,
            'patience': 10,
            'save_top_k': 5,
            'seed': None,
            'num_workers': 4
        },
        'model': {
            'version': 1
        },
        'hyperparameters': {
            "lr": {"low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": {"choices": [2, 4, 8]},
            "optimizer": {"choices": ["adam"]},
            "num_epochs": {"low": 10, "high": 40, "step": 10},
            "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
            "use_grad_clip": {"choices": [True, False]},
            "max_grad_norm": {"low": 0.1, "high": 10.0, "log": True},
            "use_scheduler": {"choices": [True, False]},
            "lr_scheduler": {
                "step_size": 5,
                "gamma": 0.5
            }
        }
    }

def save_default_config(save_path: str):
    """
    Save default configuration to YAML file.

    Args:
        save_path (str): Path where the YAML configuration file will be saved
    """
    config = get_default_config()
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
    except Exception as e:
        print(f"Failed to save default configuration: {e}")


# --------------------
# Cleanup utilities
# --------------------

def cleanup_partial_checkpoints(trial_dir, trial_number):
    """
    Remove partial checkpoints and temporary files after successful trial completion.
    
    Args:
        trial_dir (str): Directory containing trial files
        trial_number (int): Trial number for file naming
    """
    partial_files_to_remove = [
        f"best_model_trial_{trial_number}_partial.pth",
        f"trial_{trial_number}_summary_partial.json"
    ]
    
    removed_count = 0
    for filename in partial_files_to_remove:
        file_path = os.path.join(trial_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed partial checkpoint: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {filename}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} partial checkpoint(s) for trial {trial_number}")

# --------------------
# Optimizer and checkpointing
# --------------------
def get_optimizer(params, optimizer_name: str, lr: float, weight_decay: float = 1e-4):
    """
    @brief: Get the optimizer based on the name provided.
    @param params: Parameters to optimize.
    @param optimizer_name: Name of the optimizer to use.
    @param lr: Learning rate for the optimizer.
    @param weight_decay: Weight decay for the optimizer (default=1e-4).
    @return: Optimizer instance.

    @note: Supports 'adam', 'sgd', and 'adamw'. If non supported name is provided, defaults to 'adam'.
    """
    optimizers = {
        "adam": torch.optim.Adam(params, lr=lr, weight_decay=weight_decay),
        "sgd": torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay),
        "adamw": torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    }
    return optimizers.get(optimizer_name, torch.optim.Adam(params, lr=lr, weight_decay=weight_decay))

def save_model_checkpoint(model, optimizer, epoch, loss, config, save_path):
    """
    Save model checkpoint with training state.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch (int): Current training epoch
        loss (float): Current loss value
        config (dict): Training configuration
        save_path (str): Path where checkpoint will be saved
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'model_info': {
            'hidden_layer': config.get('hidden_layer', 256),
            'num_classes': config.get('num_classes', 4)
        }
    }
    torch.save(checkpoint, save_path)

# --------------------
# Dataset
# --------------------
class DefectDataset(Dataset):
    """
    PyTorch Dataset for loading images with segmentation masks and bounding box annotations.

    This dataset class is designed for instance segmentation tasks with LabelMe-format annotations.
    It loads images along with their corresponding polygon segmentation masks, automatically
    generates bounding boxes from masks, and applies optional transforms for data augmentation.

    Attributes:
        _img_keys (list): List of image identifiers
        _annotation_df (DataFrame): DataFrame containing image annotations
        _img_dict (dict): Dictionary mapping image keys to file paths
        _class_to_idx (dict): Dictionary mapping class names to indices
        _transforms (callable): Optional transforms to apply to images and annotations
    """
    def __init__(self, img_keys, annotation_df, img_dict, class_to_idx, transforms=None):
        """
        Initialize the DefectDataset.

        Args:
            img_keys (list): List of unique identifiers for images
            annotation_df (DataFrame): DataFrame containing the image annotations
            img_dict (dict): Dictionary mapping image identifiers to image file paths
            class_to_idx (dict): Dictionary mapping class labels to indices
            transforms (callable, optional): Optional transform to be applied on a sample
        """
        super(Dataset, self).__init__()
        
        self._img_keys = img_keys  # List of image keys
        self._annotation_df = annotation_df  # DataFrame containing annotations
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._class_to_idx = class_to_idx  # Dictionary mapping class names to class indices
        self._transforms = transforms  # Image transforms to be applied
        
    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._img_keys)
        
    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to fetch from the dataset

        Returns:
            tuple: (image, target) where target is a dict with keys:
                - 'masks': Binary segmentation masks (BoolTensor)
                - 'boxes': Bounding boxes in xyxy format (BoundingBoxes)
                - 'labels': Class labels (LongTensor)
        """
        # Retrieve the key for the image at the specified index
        img_key = self._img_keys[index]
        # Get the annotations for this image
        annotation = self._annotation_df.loc[img_key]
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(annotation)
        
        # Apply the transformations, if any
        if self._transforms:
            image, target = self._transforms(image, target)
        
        return image, target

    def _load_image_and_target(self, annotation):
        """
        Load an image and its target from annotations.

        Reads image file, converts polygon annotations to binary masks, generates
        bounding boxes from masks, and prepares class labels.

        Args:
            annotation (pandas.Series): The annotations for an image

        Returns:
            tuple: (image, target) where:
                - image: PIL Image in RGB format
                - target: dict with 'masks', 'boxes', and 'labels'
        """
        # Retrieve the file path of the image
        filepath = self._img_dict[annotation.name]
        # Open the image file and convert it to RGB
        image = Image.open(filepath).convert('RGB')
        
        # Convert the class labels to indices
        labels = [shape['label'] for shape in annotation['shapes']]
        labels = torch.Tensor([self._class_to_idx[label] for label in labels])
        labels = labels.to(dtype=torch.int64)

        # Convert polygons to mask images
        shape_points = [shape['points'] for shape in annotation['shapes']]
        xy_coords = [[tuple(p) for p in points] for points in shape_points]
        mask_imgs = [create_polygon_mask(image.size, xy) for xy in xy_coords]
        masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

        # Generate bounding box annotations from segmentation masks
        bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=image.size[::-1])
                
        return image, {'masks': masks,'boxes': bboxes, 'labels': labels}


def load_labelme_dataset(
    train_keys, val_keys, annotation_df, img_dict, class_names,
    train_tfms=None, valid_tfms=None):
    """
    Load a LabelMe-format dataset and create Dataset objects for training and validation.

    Creates DefectDataset instances with proper class-to-index mappings and optional
    transforms for data augmentation. Prints dataset sizes for verification.

    Args:
        train_keys (list): List of keys for the training images
        val_keys (list): List of keys for the validation images
        annotation_df (DataFrame): DataFrame containing the image annotations
        img_dict (dict): Dictionary mapping image keys to image file paths
        class_names (list): List of class names in the dataset (including background)
        train_tfms (callable, optional): Transformations to apply to training images
    valid_tfms (callable, optional): Transformations to apply to validation images.
    
    Returns:
    train_dataset (DefectDataset): Dataset for training.
    valid_dataset (DefectDataset): Dataset for validation.
    """

    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Instantiate the datasets using the defined transformations
    train_dataset = DefectDataset(train_keys, annotation_df, img_dict, class_to_idx, train_tfms)
    valid_dataset = DefectDataset(val_keys, annotation_df, img_dict, class_to_idx, valid_tfms)

    # Print the number of samples in the training and validation datasets
    pd.Series({
        'Training dataset size:': len(train_dataset),
        'Validation dataset size:': len(valid_dataset)}
    ).to_frame().style.hide(axis='columns')

    # Return the training and validation DataLoaders
    return train_dataset, valid_dataset

# --------------------
# Model
# --------------------
def create_model(num_classes: int, model_version: int = 1):
    """
    Create a Mask R-CNN model with custom number of classes.

    Args:
        num_classes (int): Number of classes including background
        model_version (int): Model version (1 for maskrcnn_resnet50_fpn, 2 for v2)

    Returns:
        MaskRCNN: Initialized Mask R-CNN model with pretrained backbone
    """
    model = None
    if model_version == 1:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=num_classes
    )


    return model

def load_model_checkpoint(num_classes: int, checkpoint_path: str, device: str, model_version: int = 1):
    """
    Load a model from checkpoint file.

    Args:
        num_classes (int): Number of classes including background
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on ('cpu' or 'cuda')
        model_version (int): Model version to use (1 or 2)

    Returns:
        MaskRCNN: Loaded model with weights from checkpoint

    Raises:
        FileNotFoundError: If checkpoint file does not exist
    """
    # Create model instance
    model = create_model(num_classes=num_classes, model_version=model_version)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def save_cpu_checkpoint(model, optimizer, epoch, loss, config, save_path):
    """
    Save a CPU-only checkpoint by moving all tensors to CPU before saving.

    Useful when GPU memory is scarce and we want to persist a safe checkpoint
    without keeping GPU memory allocated.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save (can be None)
        epoch (int): Current training epoch
        loss (float or None): Current loss value
        config (dict): Training configuration dictionary
        save_path (str): Path where checkpoint will be saved

    Returns:
        None: Checkpoint is saved to disk at save_path
    """
    try:
        # Move model state dict to CPU
        state_dict = model.state_dict()
        state_dict_cpu = {}
        for k, v in state_dict.items():
            try:
                state_dict_cpu[k] = v.detach().cpu()
            except Exception:
                # If something can't be detached, attempt to clone to CPU
                state_dict_cpu[k] = v.cpu()
        # Optimizer state dict to CPU if available
        opt_cpu = None
        if optimizer is not None:
            try:
                opt_state = optimizer.state_dict()
                # convert PyTorch tensors inside optimizer state to CPU
                def _to_cpu(obj):
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [_to_cpu(x) for x in obj]
                    elif torch.is_tensor(obj):
                        return obj.detach().cpu()
                    else:
                        return obj
                opt_cpu = _to_cpu(opt_state)
            except Exception:
                opt_cpu = None

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict_cpu,
            'optimizer_state_dict': opt_cpu,
            'loss': loss,
            'config': config
        }
        torch.save(checkpoint, save_path)
    except Exception as e:
        print(f"Failed to save CPU checkpoint to {save_path}: {e}")
        traceback.print_exc()
# --------------------
# Validation
# --------------------
def validate_model(model, val_loader, device, confidence_threshold=0.5):
    """
    Validate model and compute comprehensive metrics.

    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        confidence_threshold (float): Confidence threshold for predictions (default: 0.5)

    Returns:
        tuple: (avg_val_loss, all_predictions, all_targets, metrics) where metrics dict
               contains accuracy, precision, recall, F1 scores (weighted and macro)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    validation_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, annotations in tqdm(val_loader, desc="Validating"):
            imgs = [img.to(device) for img in imgs]
            
            # Move annotations to device for loss calculation
            targets = []
            for annot in annotations:
                target = {}
                target['masks'] = annot['masks'].to(device)
                target['boxes'] = annot['boxes'].to(device)
                target['labels'] = annot['labels'].to(device)
                targets.append(target)
            
            # Get predictions
            model.train()  # Temporarily set to train mode for loss calculation
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            validation_loss += losses.item()
            
            # Get predictions in eval mode
            model.eval()
            predictions = model(imgs)
            
            # Process predictions and targets for metrics
            for pred, target in zip(predictions, targets):
                # Get target labels
                target_labels = target['labels'].cpu().numpy()
                
                # Filter predictions by confidence
                if len(pred['scores']) > 0:
                    keep = pred['scores'] > confidence_threshold
                    if keep.sum() > 0:  # Only add if there are confident predictions
                        pred_labels = pred['labels'][keep].cpu().numpy()
                        # For each target, find the closest prediction or mark as background (0)
                        # This is a simplified approach for object detection metrics
                        for target_label in target_labels:
                            if len(pred_labels) > 0:
                                # Take the first prediction (could be improved with IoU matching)
                                all_predictions.append(pred_labels[0])
                                all_targets.append(target_label)
                            else:
                                # No confident prediction, assume background
                                all_predictions.append(0)  # background class
                                all_targets.append(target_label)
                    else:
                        # No confident predictions, all targets are missed (predict background)
                        for target_label in target_labels:
                            all_predictions.append(0)  # background class
                            all_targets.append(target_label)
                else:
                    # No predictions at all, all targets are missed
                    for target_label in target_labels:
                        all_predictions.append(0)  # background class
                        all_targets.append(target_label)
            
            num_batches += 1
    
    avg_val_loss = validation_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate metrics
    metrics = {}
    if len(all_predictions) > 0 and len(all_targets) > 0 and len(all_predictions) == len(all_targets):
        print(f"\nCalculating metrics with {len(all_predictions)} samples")
        
        # Accuracy
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Precision, Recall, F1-score (weighted average for multi-class)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'num_predictions': len(all_predictions),
            'num_targets': len(all_targets)
        }
    else:
        print(f"Skipping metrics calculation: predictions={len(all_predictions)}, targets={len(all_targets)}")
        # Default metrics when no predictions/targets available or inconsistent lengths
        metrics = {
            'accuracy': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'num_predictions': len(all_predictions),
            'num_targets': len(all_targets)
        }
    
    return avg_val_loss, all_predictions, all_targets, metrics

def calculate_training_metrics(model, train_loader, device, max_batches=None, confidence_threshold=0.5):
    """
    Calculate metrics on a subset of training data without affecting gradients.

    Args:
        model: PyTorch model to evaluate
        train_loader: DataLoader for training data
        device: Device to run evaluation on
        max_batches (int, optional): Maximum number of batches to evaluate
        confidence_threshold (float): Confidence threshold for predictions (default: 0.5)

    Returns:
        dict: Metrics dictionary with accuracy, precision, recall, F1 scores
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (imgs, annotations) in enumerate(train_loader):
            if max_batches and i >= max_batches:
                break
                
            imgs = [img.to(device) for img in imgs]
            
            # Move annotations to device
            targets = []
            for annot in annotations:
                target = {}
                target['masks'] = annot['masks'].to(device)
                target['boxes'] = annot['boxes'].to(device)
                target['labels'] = annot['labels'].to(device)
                targets.append(target)
            
            # Get predictions in eval mode
            predictions = model(imgs)
            
            # Process predictions and targets for metrics
            for pred, target in zip(predictions, targets):
                # Get target labels
                target_labels = target['labels'].cpu().numpy()
                
                # Filter predictions by confidence
                if len(pred['scores']) > 0:
                    keep = pred['scores'] > confidence_threshold
                    if keep.sum() > 0:  # Only add if there are confident predictions
                        pred_labels = pred['labels'][keep].cpu().numpy()
                        # For each target, find the closest prediction or mark as background (0)
                        # This is a simplified approach for object detection metrics
                        for target_label in target_labels:
                            if len(pred_labels) > 0:
                                # Take the first prediction (could be improved with IoU matching)
                                all_predictions.append(pred_labels[0])
                                all_targets.append(target_label)
                            else:
                                # No confident prediction, assume background
                                all_predictions.append(0)  # background class
                                all_targets.append(target_label)
                    else:
                        # No confident predictions, all targets are missed (predict background)
                        for target_label in target_labels:
                            all_predictions.append(0)  # background class
                            all_targets.append(target_label)
                else:
                    # No predictions at all, all targets are missed
                    for target_label in target_labels:
                        all_predictions.append(0)  # background class
                        all_targets.append(target_label)
    
    # Calculate metrics
    metrics = {}
    if len(all_predictions) > 0 and len(all_targets) > 0 and len(all_predictions) == len(all_targets):
        # Accuracy
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Precision, Recall, F1-score (weighted average for multi-class)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'num_predictions': len(all_predictions),
            'num_targets': len(all_targets)
        }
    else:
        # Default metrics when no predictions/targets available or inconsistent lengths
        metrics = {
            'accuracy': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'num_predictions': len(all_predictions),
            'num_targets': len(all_targets)
        }
    
    return metrics

# --------------------
# Plotting functions
# --------------------


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation curves with improvement tracking.

    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path where the plot image will be saved
    """
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Loss improvement
    plt.subplot(1, 2, 2)
    train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
    plt.plot(epochs, train_improvement, 'b-', label='Training Improvement (%)')
    plt.plot(epochs, val_improvement, 'r-', label='Validation Improvement (%)')
    plt.title('Loss Improvement Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Improvement (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------
# Draw utilities
# --------------------

# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Download the font file
if not os.path.exists(font_file):
    download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")


# Bounding box drawing function
draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=25)

# Polygon mask creation function
def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img

def save_epoch_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, save_path, config=None, trial_number=None):
    """
    Save epoch metrics to a JSON file for tracking training progress.

    Appends metrics for the current epoch to an existing metrics file or creates
    a new one if it doesn't exist. Includes trial information and hyperparameters
    for reproducibility.

    Args:
        epoch (int): Current epoch number
        train_loss (float): Training loss for this epoch
        train_metrics (dict): Training metrics dictionary (accuracy, precision, recall, F1)
        val_loss (float): Validation loss for this epoch
        val_metrics (dict): Validation metrics dictionary
        save_path (str): Path where metrics JSON file will be saved
        config (dict, optional): Hyperparameter configuration dictionary
        trial_number (int, optional): Optuna trial number for this training run

    Returns:
        None: Metrics are appended to JSON file at save_path
    """
    metrics_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_metrics': train_metrics,
        'val_loss': val_loss,
        'val_metrics': val_metrics
    }
    
    # Load existing metrics if file exists
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'trial_info': {
                'trial_number': trial_number,
                'hyperparameters': config,
                'start_time': datetime.now().isoformat() if epoch == 1 else None
            },
            'epochs': []
        }
    
    # Ensure the structure exists
    if 'epochs' not in data:
        data['epochs'] = []
    
    # Update trial info if this is the first epoch or if not set
    if 'trial_info' not in data or epoch == 1:
        data['trial_info'] = {
            'trial_number': trial_number,
            'hyperparameters': config,
            'start_time': datetime.now().isoformat()
        }
    
    data['epochs'].append(metrics_data)
    
    # Save updated metrics
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def save_trial_summary(trial_number, config, best_val_loss, total_epochs, train_losses, val_losses, save_path, confusion_matrix_path=None, stopping_reason="completed"):
    """
    Save a comprehensive summary of a single optimization trial.

    Saves trial metadata, hyperparameters, training history, best results, and paths
    to generated artifacts in a structured JSON format for later analysis.

    Args:
        trial_number (int): Optuna trial number
        config (dict): Hyperparameter configuration used in this trial
        best_val_loss (float): Best validation loss achieved
        total_epochs (int): Total number of epochs completed
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the summary JSON file
        confusion_matrix_path (str, optional): Path to confusion matrix image
        stopping_reason (str): Reason for stopping ("completed", "early_stopping", "pruned", "error")
    """
    summary = {
        'trial_number': trial_number,
        'hyperparameters': config,
        'results': {
            'best_validation_loss': best_val_loss,
            'total_epochs_completed': total_epochs,
            'stopping_reason': stopping_reason,  # "completed", "early_stopping", "pruned", "error"
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
        },
        'artifacts': {
            'confusion_matrix_path': confusion_matrix_path,
            'training_curves_path': f"training_curves_trial_{trial_number}.png",
            'best_model_path': f"best_model_trial_{trial_number}.pth",
            'epoch_metrics_path': f"epoch_metrics_trial_{trial_number}.json"
        },
        'completion_time': datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

def generate_confusion_matrix(model, dataloader, device, class_names, save_path, confidence_threshold=0.5):
    """
    Generate and save confusion matrix for the model predictions.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names (list): List of class names for labeling
        save_path (str): Path where confusion matrix image will be saved
        confidence_threshold (float): Confidence threshold for predictions (default: 0.5)

    Returns:
        np.ndarray or None: Confusion matrix array if successful, None if insufficient data
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, annotations in tqdm(dataloader, desc="Generating confusion matrix"):
            imgs = [img.to(device) for img in imgs]
            
            # Move annotations to device
            targets = []
            for annot in annotations:
                target = {}
                target['masks'] = annot['masks'].to(device)
                target['boxes'] = annot['boxes'].to(device)
                target['labels'] = annot['labels'].to(device)
                targets.append(target)
            
            # Get predictions in eval mode
            predictions = model(imgs)
            
            # Process predictions and targets for confusion matrix
            for pred, target in zip(predictions, targets):
                # Get target labels
                target_labels = target['labels'].cpu().numpy()
                
                # Filter predictions by confidence
                if len(pred['scores']) > 0:
                    keep = pred['scores'] > confidence_threshold
                    if keep.sum() > 0:  # Only add if there are confident predictions
                        pred_labels = pred['labels'][keep].cpu().numpy()
                        # For each target, find the closest prediction or mark as background (0)
                        # This is a simplified approach for object detection metrics
                        for target_label in target_labels:
                            if len(pred_labels) > 0:
                                # Take the first prediction (could be improved with IoU matching)
                                all_predictions.append(pred_labels[0])
                                all_targets.append(target_label)
                            else:
                                # No confident prediction, assume background
                                all_predictions.append(0)  # background class
                                all_targets.append(target_label)
                    else:
                        # No confident predictions, all targets are missed (predict background)
                        for target_label in target_labels:
                            all_predictions.append(0)  # background class
                            all_targets.append(target_label)
                else:
                    # No predictions at all, all targets are missed
                    for target_label in target_labels:
                        all_predictions.append(0)  # background class
                        all_targets.append(target_label)
    
    if len(all_predictions) > 0 and len(all_targets) > 0 and len(all_predictions) == len(all_targets):
        print(f"Generating confusion matrix with {len(all_predictions)} samples")
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=range(len(class_names)))
        
        # Plot and save confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw confusion matrix data
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'predictions_count': len(all_predictions),
            'targets_count': len(all_targets)
        }
        
        cm_json_path = save_path.replace('.png', '_data.json')
        with open(cm_json_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        
        print(f"Confusion matrix saved to: {save_path}")
        print(f"Confusion matrix data saved to: {cm_json_path}")
        
        return cm
    else:
        print(f"Cannot generate confusion matrix: predictions={len(all_predictions)}, targets={len(all_targets)}")
        return None

def _create_parameter_analysis_plots(completed_trials, report_dir):
    """Create visual analysis plots for parameter trends"""
    try:
        # Sort trials by performance
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        
        # Extract all unique parameters
        all_params = set()
        for trial in completed_trials:
            all_params.update(trial.params.keys())
        
        # Create parameter trend plots
        numeric_params = []
        categorical_params = []
        
        for param in all_params:
            sample_value = completed_trials[0].params.get(param)
            if isinstance(sample_value, (int, float)):
                numeric_params.append(param)
            else:
                categorical_params.append(param)
        
        # Plot numeric parameters
        if numeric_params:
            n_numeric = len(numeric_params)
            fig, axes = plt.subplots(2, (n_numeric + 1) // 2, figsize=(15, 8))
            if n_numeric == 1:
                axes = [axes]
            elif n_numeric == 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, param in enumerate(numeric_params):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Get parameter values and corresponding losses
                    param_values = [trial.params.get(param) for trial in sorted_trials if param in trial.params]
                    losses = [trial.value for trial in sorted_trials if param in trial.params]
                    
                    # Create scatter plot
                    ax.scatter(param_values, losses, alpha=0.7, s=50)
                    ax.set_xlabel(param)
                    ax.set_ylabel('Validation Loss')
                    ax.set_title(f'{param} vs Performance')
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line if enough points
                    if len(param_values) >= 3:
                        z = np.polyfit(param_values, losses, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(param_values), max(param_values), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Hide empty subplots
            for i in range(len(numeric_params), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "parameter_trends.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create categorical parameter analysis
        if categorical_params:
            fig, axes = plt.subplots(1, len(categorical_params), figsize=(5 * len(categorical_params), 6))
            if len(categorical_params) == 1:
                axes = [axes]
            
            for i, param in enumerate(categorical_params):
                ax = axes[i]
                
                # Group trials by categorical value
                param_groups = {}
                for trial in completed_trials:
                    if param in trial.params:
                        param_value = trial.params[param]
                        if param_value not in param_groups:
                            param_groups[param_value] = []
                        param_groups[param_value].append(trial.value)
                
                # Create box plot
                categories = list(param_groups.keys())
                values = [param_groups[cat] for cat in categories]
                
                bp = ax.boxplot(values, labels=categories, patch_artist=True)
                ax.set_ylabel('Validation Loss')
                ax.set_title(f'{param} Performance Distribution')
                ax.grid(True, alpha=0.3)
                
                # Color boxes based on median performance
                medians = [np.median(v) for v in values]
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(medians)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "categorical_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Parameter analysis plots saved to {report_dir}")
        
    except Exception as e:
        print(f"Failed to create parameter analysis plots: {e}")
        traceback.print_exc()

def create_report(study, result_folder, class_names):
    """
    Create a comprehensive report of all optimization trials.

    Generates detailed analysis including:
    - Parameter importance and trends
    - Best vs worst trial comparisons
    - Training completion statistics
    - Confusion matrices for top trials
    - Markdown and JSON reports

    Args:
        study: Optuna study object with trial results
        result_folder (str): Path to results folder
        class_names (list): List of class names for reporting

    Returns:
        str: Path to the generated report file
    """
    report_dir = os.path.join(result_folder, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Get completed trials
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials to create report")
        return
    
    # Create parameter analysis plots
    if len(completed_trials) >= 3:
        _create_parameter_analysis_plots(completed_trials, report_dir)
    
    # Create summary report
    report_data = {
        'study_summary': {
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'study_name': study.study_name,
            'generation_time': datetime.now().isoformat()
        },
        'class_names': class_names,
        'completed_trials_summary': []
    }
    
    # Process each completed trial
    for trial in sorted(completed_trials, key=lambda t: t.value):
        trial_dir = os.path.join(result_folder, f"trial_{trial.number}")
        
        trial_summary = {
            'trial_number': trial.number,
            'validation_loss': trial.value,
            'hyperparameters': trial.params,
            'artifacts': {
                'confusion_matrix': f"trial_{trial.number}/confusion_matrix_trial_{trial.number}.png",
                'confusion_matrix_data': f"trial_{trial.number}/confusion_matrix_trial_{trial.number}_data.json",
                'training_curves': f"trial_{trial.number}/training_curves_trial_{trial.number}.png",
                'epoch_metrics': f"trial_{trial.number}/epoch_metrics_trial_{trial.number}.json",
                'trial_summary': f"trial_{trial.number}/trial_{trial.number}_summary.json",
                'best_model': f"trial_{trial.number}/best_model_trial_{trial.number}.pth"
            }
        }
        
        # Read trial summary to get epochs and stopping information
        trial_summary_path = os.path.join(trial_dir, f"trial_{trial.number}_summary.json")
        if os.path.exists(trial_summary_path):
            try:
                with open(trial_summary_path, 'r') as f:
                    summary_data = json.load(f)
                    results = summary_data.get('results', {})
                    trial_summary['total_epochs'] = results.get('total_epochs_completed', 'Unknown')
                    trial_summary['stopping_reason'] = results.get('stopping_reason', 'Unknown')
                    trial_summary['was_pruned'] = results.get('was_pruned', False)
            except Exception as e:
                print(f"Failed to load trial summary for trial {trial.number}: {e}")
                trial_summary['total_epochs'] = 'Unknown'
                trial_summary['stopping_reason'] = 'Unknown'
                trial_summary['was_pruned'] = False
        else:
            trial_summary['total_epochs'] = 'Unknown'
            trial_summary['stopping_reason'] = 'Unknown'
            trial_summary['was_pruned'] = False
        
        # Check if confusion matrix data exists and add metrics
        cm_data_path = os.path.join(trial_dir, f"confusion_matrix_trial_{trial.number}_data.json")
        if os.path.exists(cm_data_path):
            try:
                with open(cm_data_path, 'r') as f:
                    cm_data = json.load(f)
                    trial_summary['confusion_matrix_metrics'] = {
                        'predictions_count': cm_data.get('predictions_count', 0),
                        'targets_count': cm_data.get('targets_count', 0)
                    }
            except Exception as e:
                print(f"Failed to load confusion matrix data for trial {trial.number}: {e}")
        
        report_data['completed_trials_summary'].append(trial_summary)
    
    # Save report
    report_file = os.path.join(report_dir, "report.json")
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create a markdown summary with analysis
    md_file = os.path.join(report_dir, "report.md")
    with open(md_file, 'w') as f:
        f.write("# Mask R-CNN Training Report\n\n")
        f.write(f"**Study Name:** {study.study_name}<br>\n")
        f.write(f"**Generation Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Study Summary\n")
        f.write(f"- Total Trials: {report_data['study_summary']['total_trials']}\n")
        f.write(f"- Completed Trials: {report_data['study_summary']['completed_trials']}\n")
        f.write(f"- Pruned Trials: {report_data['study_summary']['pruned_trials']}\n")
        f.write(f"- Failed Trials: {report_data['study_summary']['failed_trials']}\n\n")
        
        # Add training completion analysis
        if completed_trials:
            f.write("### Training Completion Analysis\n")
            
            # Analyze stopping reasons
            stopping_reasons = {}
            total_epochs_list = []
            valid_epochs = 0
            
            for trial_data in report_data['completed_trials_summary']:
                reason = trial_data.get('stopping_reason', 'Unknown')
                stopping_reasons[reason] = stopping_reasons.get(reason, 0) + 1
                
                epochs = trial_data.get('total_epochs', 'Unknown')
                if isinstance(epochs, (int, float)) and epochs != 'Unknown':
                    total_epochs_list.append(epochs)
                    valid_epochs += 1
            
            # Display stopping reasons
            for reason, count in stopping_reasons.items():
                percentage = (count / len(completed_trials)) * 100
                if reason == "completed":
                    emoji = "✅"
                    description = "Reached maximum epochs"
                elif reason == "early_stopping":
                    emoji = "⏱️"
                    description = "Early stopping triggered"
                elif reason == "pruned":
                    emoji = "✂️"
                    description = "Pruned by Optuna"
                elif reason == "error":
                    emoji = "❌"
                    description = "Training error"
                else:
                    emoji = "❓"
                    description = "Other reason"
                
                f.write(f"- {emoji} **{reason.replace('_', ' ').title()}:** {count} trials ({percentage:.1f}%) - {description}\n")
            
            # Display epoch statistics
            if total_epochs_list:
                avg_epochs = sum(total_epochs_list) / len(total_epochs_list)
                min_epochs = min(total_epochs_list)
                max_epochs = max(total_epochs_list)
                
                f.write(f"\n**Epoch Statistics:**\n")
                f.write(f"- Average epochs: {avg_epochs:.1f}\n")
                f.write(f"- Minimum epochs: {min_epochs}\n")
                f.write(f"- Maximum epochs: {max_epochs}\n")
                f.write(f"- Trials with epoch data: {valid_epochs}/{len(completed_trials)}\n")
            
            f.write("\n")
        
        f.write("## Class Names\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")
        f.write("\n")
        
        # Add parameter analysis section
        if len(completed_trials) >= 3:  # Only analyze if we have enough trials
            f.write("## 📊 Parameter Analysis & Insights\n\n")
            
            # Reference the generated plots
            f.write("### 📈 Visual Parameter Analysis\n\n")
            f.write("![Parameter Trends](parameter_trends.png)\n\n")
            f.write("*The scatter plots above show the relationship between each numeric parameter and validation loss. ")
            f.write("Red dashed lines indicate the trend direction.*\n\n")
            f.write("![Categorical Analysis](categorical_analysis.png)\n\n")
            f.write("*Box plots showing performance distribution for categorical parameters. ")
            f.write("Green indicates better performance, red indicates worse performance.*\n\n")
            
            # Get top 5 and bottom 5 trials for comparison
            top_5_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
            bottom_5_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
            
            # Analyze parameter trends
            f.write("### 🎯 Best vs Worst Parameter Comparison\n\n")
            
            # Extract all unique parameters
            all_params = set()
            for trial in completed_trials:
                all_params.update(trial.params.keys())
            
            for param in sorted(all_params):
                # Get values from top and bottom trials
                top_values = [trial.params.get(param) for trial in top_5_trials if param in trial.params]
                bottom_values = [trial.params.get(param) for trial in bottom_5_trials if param in trial.params]
                
                if top_values and bottom_values:
                    f.write(f"**{param}:**\n")
                    
                    # Handle numeric parameters
                    if isinstance(top_values[0], (int, float)):
                        top_avg = sum(top_values) / len(top_values)
                        bottom_avg = sum(bottom_values) / len(bottom_values)
                        top_range = f"{min(top_values):.4f} - {max(top_values):.4f}"
                        bottom_range = f"{min(bottom_values):.4f} - {max(bottom_values):.4f}"
                        
                        f.write(f"- 🟢 **Best trials average:** {top_avg:.4f} (range: {top_range})\n")
                        f.write(f"- 🔴 **Worst trials average:** {bottom_avg:.4f} (range: {bottom_range})\n")
                        
                        # Provide direction insight
                        if top_avg > bottom_avg:
                            f.write(f"- 💡 **Insight:** Higher {param} values tend to perform better\n")
                        elif top_avg < bottom_avg:
                            f.write(f"- 💡 **Insight:** Lower {param} values tend to perform better\n")
                        else:
                            f.write(f"- 💡 **Insight:** {param} shows no clear directional trend\n")
                    
                    # Handle categorical parameters
                    else:
                        from collections import Counter
                        top_counts = Counter(top_values)
                        bottom_counts = Counter(bottom_values)
                        
                        f.write(f"- 🟢 **Best trials prefer:** {dict(top_counts)}\n")
                        f.write(f"- 🔴 **Worst trials show:** {dict(bottom_counts)}\n")
                        
                        # Find most successful category
                        if top_counts:
                            best_category = top_counts.most_common(1)[0][0]
                            f.write(f"- 💡 **Insight:** '{best_category}' appears most often in top trials\n")
                    
                    f.write("\n")
            
            # Recommended configuration based on analysis
            f.write("### 🚀 Recommended Configuration\n\n")
            f.write("Based on the analysis of successful trials, consider these parameter ranges:\n\n")
            
            best_trial = top_5_trials[0]
            # Find the corresponding trial data to get epoch information
            best_trial_data = next((td for td in report_data['completed_trials_summary'] 
                                  if td['trial_number'] == best_trial.number), None)
            
            f.write(f"**🏆 Best performing trial (#{best_trial.number}) used:**\n")
            for param, value in best_trial.params.items():
                f.write(f"- `{param}`: {value}\n")
            f.write(f"- **Validation Loss:** {best_trial.value:.6f}\n")
            
            if best_trial_data:
                epochs = best_trial_data.get('total_epochs', 'Unknown')
                stopping_reason = best_trial_data.get('stopping_reason', 'Unknown')
                f.write(f"- **Epochs Completed:** {epochs}\n")
                f.write(f"- **Stopping Reason:** {stopping_reason.replace('_', ' ').title()}\n")
            
            f.write("\n")
            
            # Parameter stability analysis
            f.write("### 📈 Parameter Stability Analysis\n\n")
            for param in sorted(all_params):
                param_values = [trial.params.get(param) for trial in completed_trials if param in trial.params]
                if param_values and isinstance(param_values[0], (int, float)):
                    std_dev = np.std(param_values)
                    mean_val = np.mean(param_values)
                    cv = std_dev / mean_val if mean_val != 0 else 0  # Coefficient of variation
                    
                    if cv < 0.2:
                        stability = "🟢 Stable"
                    elif cv < 0.5:
                        stability = "🟡 Moderate"
                    else:
                        stability = "🔴 Highly Variable"
                    
                    f.write(f"- **{param}:** {stability} (CV: {cv:.3f})\n")
            
            f.write("\n")
            
            # Success rate analysis
            f.write("### 📊 Training Success Patterns\n\n")
            total_trials = len(study.trials)
            success_rate = len(completed_trials) / total_trials * 100
            f.write(f"- **Overall Success Rate:** {success_rate:.1f}% ({len(completed_trials)}/{total_trials})\n")
            
            if len(completed_trials) > 1:
                # Performance improvement over time
                trial_numbers = [t.number for t in completed_trials]
                trial_values = [t.value for t in completed_trials]
                
                # Check if there's improvement trend
                if len(trial_values) >= 5:
                    recent_avg = np.mean(trial_values[-10:])  # Last 10 trials
                    early_avg = np.mean(trial_values[:10])    # First 10 trials
                    
                    if recent_avg < early_avg:
                        trend = "🟢 Improving"
                        improvement = ((early_avg - recent_avg) / early_avg) * 100
                        f.write(f"- **Performance Trend:** {trend} ({improvement:.1f}% better)\n")
                    else:
                        trend = "🔴 Declining"
                        decline = ((recent_avg - early_avg) / early_avg) * 100
                        f.write(f"- **Performance Trend:** {trend} ({decline:.1f}% worse)\n")
            
            f.write("\n")
        
        f.write("## 📋 Completed Trials (Sorted by Validation Loss)\n\n")
        for i, trial_data in enumerate(report_data['completed_trials_summary'], 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            f.write(f"### {rank_emoji} Trial {trial_data['trial_number']}\n")
            f.write(f"**Validation Loss:** {trial_data['validation_loss']:.6f}<br>\n")

            # Add epochs and stopping information
            total_epochs = trial_data['hyperparameters'].get('num_epochs', 'Unknown')
            epochs_completed = trial_data.get('total_epochs', 'Unknown')
            stopping_reason = trial_data.get('stopping_reason', 'Unknown')

            f.write(f"**Epochs Completed:** {epochs_completed}/{total_epochs}<br>\n")

            # Format stopping reason with appropriate emoji
            if stopping_reason == "completed":
                stopping_text = "Completed (reached max epochs)"
            elif stopping_reason == "early_stopping":
                stopping_text = "Early stopping (no improvement)"
            elif stopping_reason == "pruned":
                stopping_text = "Pruned by Optuna"
            elif stopping_reason == "error":
                stopping_text = "Error occurred"
            else:
                stopping_text = f"Unknown ({stopping_reason})"

            f.write(f"**Stopping Reason:** {stopping_text}<br>\n")
            
            f.write("\n")
            
            f.write("**Hyperparameters:**\n")
            for param, value in trial_data['hyperparameters'].items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
    
    print(f"Report saved to: {report_file}")
    print(f"Markdown summary saved to: {md_file}")
    
    return report_file