#!/usr/bin/env python3
"""
@brief Mask R-CNN Training Script with Hyperparameter Optimization using Optuna
@details This script trains a Mask R-CNN model on a custom dataset with bounding box and segmentation mask annotations.
    It includes data loading, augmentation, model creation, training, validation, and hyperparameter optimization using Optuna.
    It also generates a comprehensive report with training metrics and confusion matrices.
    
    Hyperparameters are loaded from hyperparam_config.yaml by default, or from a custom YAML file specified
    with --hyperparam-config. This allows for easy configuration management and reproducible experiments.
    
    Arguments:
    --result-folder: Path to the folder where results will be saved.
    --dataset-folder: Path to the folder containing the dataset in LabelMe format.
    --hyperparam-config: Path to hyperparameter configuration YAML file (optional).
    --show-samples: Flag to display sample images with annotations before training.

@note The dataset should be in LabelMe format (JSON files with polygon annotations).
@note The script supports early stopping and saves the best model checkpoints based on validation loss.
@note Hyperparameters can be customized by modifying hyperparam_config.yaml or providing a custom config file.
"""
# Import Python Standard Library dependencies
import os
import gc
import json
import shutil
import random
import argparse
import traceback
import multiprocessing
from pathlib import Path
from datetime import datetime

# Import utility functions
from cjm_pil_utils.core import get_img_files
from cjm_pytorch_utils.core import tensor_to_pil

# Import the distinctipy module
from distinctipy import distinctipy

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image, ImageDraw

# Import PyTorch dependencies
import torch
from torch.utils.data import DataLoader
from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2  as transforms

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import Optuna for hyperparameter optimization
import optuna
import pickle

from .utils import (
    get_default_config,
    load_training_config,
    save_default_config,
    get_optimizer,
    save_cpu_checkpoint,
    create_model,
    validate_model,
    calculate_training_metrics,
    save_epoch_metrics,
    save_trial_summary,
    generate_confusion_matrix,
    create_report,
    load_labelme_dataset,
    plot_training_curves,
    cleanup_partial_checkpoints,
    draw_bboxes,
    draw_segmentation_masks,
    create_polygon_mask,
    Mask
)

def main():
    
    
    parser = argparse.ArgumentParser(description="Mask R-CNN Training")
    parser.add_argument("--result-folder", type=str, required=True, help="Path to results folder")
    parser.add_argument("--dataset-folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--train-config", type=str, help="Path to training configuration YAML file")
    parser.add_argument("--show-samples", action="store_true", help="Show sample images with annotations")
    parser.add_argument("--yolo", action="store_true", help="Use YOLODataset folder structure")

    args = parser.parse_args()

    # Load training configuration
    config = load_training_config(args.train_config)

    if config is None:
        # Load and save default config
        config = get_default_config()
        save_default_config(os.path.join(args.result_folder, "training_config.yaml"))
    else:
        # Copy configuration to results folder for reproducibility
        shutil.copy(args.train_config, args.result_folder)

    training_config = config.get('training', {})
    model_config = config.get('model', {})

    # Export CUDA configuration
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"CUDA expandable segments: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    # Set random seed for reproducibility
    seed = training_config.get('seed', None)
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)

    # check dataset folder
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Dataset folder {args.dataset_folder} does not exist.")

    # create result folder
    if not os.path.exists(args.result_folder):
        print(f"Warning: Result folder {args.result_folder} does not exist. Creating it.")
        os.makedirs(args.result_folder, exist_ok=False)

    # Load search parameters
    param_space = config.get('hyperparameters', {})
    augmentation_params = config.get('augmentation', {})

    # --------------------
    # Data loading and augmentation
    # --------------------

    # Compose transforms for data augmentation
    brightness = (augmentation_params.get('brightness', {}).get('min', 0.875), augmentation_params.get('brightness', {}).get('max', 1.125))
    contrast = (augmentation_params.get('contrast', {}).get('min', 0.5), augmentation_params.get('contrast', {}).get('max', 1.5))
    saturation = (augmentation_params.get('saturation', {}).get('min', 0.5), augmentation_params.get('saturation', {}).get('max', 1.5))
    hue = (augmentation_params.get('hue', {}).get('min', -0.05), augmentation_params.get('hue', {}).get('max', 0.05))


    data_aug_tfms = transforms.Compose(
        transforms=[
            transforms.RandomGrayscale(augmentation_params.get('gray_scale', 0.0)),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.RandomHorizontalFlip(augmentation_params.get('horizontal_flip', 0.0)),
        ],
    )

    # Compose transforms to sanitize bounding boxes and normalize input data
    final_tfms = transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
    ])

    # Define the transformations for training and validation datasets
    train_tfms = transforms.Compose([
        #data_aug_tfms,
        final_tfms
    ])

    valid_tfms = final_tfms

    # --------------------
    # Load datasets
    # --------------------

    print(f"Loading dataset from {args.dataset_folder} ...")

    if getattr(args, "yolo", False):
        # Use YOLODataset folder structure
        yolo_base = Path(args.dataset_folder)
        train_img_dir = yolo_base / "YOLODataset/images/train"
        val_img_dir = yolo_base / "YOLODataset/images/val"

        # Helper function to extract base name by cutting at last underscore
        def base_name(stem):
            parts = stem.split('_')
            if len(parts) > 1:
                return '_'.join(parts[:-1])
            return stem

        # Collect image file names (without extension) from train and val folders
        train_img_files = list(train_img_dir.glob("*")) if train_img_dir.exists() else []
        val_img_files = list(val_img_dir.glob("*")) if val_img_dir.exists() else []

        # Extract stems and convert to base names
        train_stems = [file.stem for file in train_img_files if file.is_file()]
        val_stems = [file.stem for file in val_img_files if file.is_file()]
        
        # Cut stems at last underscore to get base names
        train_base_names = [base_name(stem) for stem in train_stems]
        val_base_names = [base_name(stem) for stem in val_stems]
        
        print(f"YOLO mode enabled.")
        print(f"Raw train image files found: {len(train_stems)}")
        print(f"Raw validation image files found: {len(val_stems)}")

        # Get all images and JSONs from the main dataset folder
        dataset_folder = Path(args.dataset_folder)
        img_file_paths = get_img_files(args.dataset_folder)
        all_jsons = list(dataset_folder.glob('*.json'))
        
        print(f"Total images in dataset folder: {len(img_file_paths)}")
        print(f"Total JSON files in dataset folder: {len(all_jsons)}")

        # Create mapping from base names to actual file paths
        img_dict = {}  # base_name -> image_file_path
        json_dict = {}  # base_name -> json_file_path
        
        # Map images by their exact stem (no cutting needed - these are original names)
        for img_path in img_file_paths:
            img_dict[img_path.stem] = img_path
                
        # Map JSONs by their exact stem (no cutting needed - these are original names)
        for json_path in all_jsons:
            json_dict[json_path.stem] = json_path

        print(f"Unique image stems in dataset: {len(img_dict)}")
        print(f"Unique JSON stems in dataset: {len(json_dict)}")

        # Find matching images and annotations for train/val base names
        train_keys = []
        val_keys = []
        annotation_file_paths = []
        
        # Process train base names - match cut names against exact dataset stems
        train_missing = 0
        for base in train_base_names:
            if base in img_dict and base in json_dict:
                train_keys.append(base)
                if json_dict[base] not in annotation_file_paths:
                    annotation_file_paths.append(json_dict[base])
            else:
                train_missing += 1
                
        # Process val base names - match cut names against exact dataset stems
        val_missing = 0
        for base in val_base_names:
            if base in img_dict and base in json_dict:
                val_keys.append(base)
                if json_dict[base] not in annotation_file_paths:
                    annotation_file_paths.append(json_dict[base])
            else:
                val_missing += 1

        print(f"Train matches: {len(train_keys)} (missing: {train_missing})")
        print(f"Val matches: {len(val_keys)} (missing: {val_missing})")  
        print(f"Total annotation files: {len(annotation_file_paths)}")
        
        # Update img_dict to use the matched base names as keys
        img_dict = {base: img_dict[base] for base in train_keys + val_keys}
    else:
        # Get a list of image files in the dataset
        img_file_paths = get_img_files(args.dataset_folder)

        # Get a list of JSON files in the dataset
        dataset_folder = Path(args.dataset_folder)
        annotation_file_paths = list(dataset_folder.glob('*.json'))

        # Create a dictionary that maps file names to file paths
        img_dict = {file.stem : file for file in img_file_paths}

        # Print the number of image files
        print(f"Number of Images: {len(img_dict)}")

        # Get the list of image IDs
        img_keys = list(img_dict.keys())

        # Shuffle the image IDs
        random.shuffle(img_keys)

        # Calculate the index at which to split the subset of image paths into training and validation sets
        train_split = int(len(img_keys)*training_config.get('train_pct', 0.8))

        # Split the subset of image paths into training and validation sets
        train_keys = img_keys[:train_split]
        val_keys = img_keys[train_split:]

    # Print the number of images in the training and validation sets
    print (f"Training Samples: {len(train_keys)}")
    print (f"Validation Samples: {len(val_keys)}")

    # Create a generator that yields Pandas DataFrames containing the data from each JSON file
    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))

    # Concatenate the DataFrames into a single DataFrame
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)

    # Assign the image file name as the index for each row
    annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
    annotation_df = annotation_df.set_index('index')

    # Keep only the rows that correspond to the filenames in the 'img_dict' dictionary
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # Create a generator that yields Pandas DataFrames containing the data from each JSON file
    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))

    # Concatenate the DataFrames into a single DataFrame
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)

    # Assign the image file name as the index for each row
    annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
    annotation_df = annotation_df.set_index('index')

    # Keep only the rows that correspond to the filenames in the 'img_dict' dictionary
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # Explode the 'shapes' column in the annotation_df dataframe
    # Convert the resulting series to a dataframe and rename the 'shapes' column to 'shapes'
    # Apply the pandas Series function to the 'shapes' column of the dataframe
    shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

    # Get a list of unique labels in the 'annotation_df' DataFrame
    class_names = shapes_df['label'].unique().tolist()

   # # Display labels using a Pandas DataFrame
   # pd.DataFrame(class_names)
   # 
   # # Display the class distribution
   # class_counts = shapes_df['label'].value_counts()
   #
   # # Plot the distribution
   # class_counts.plot(kind='bar')
   # plt.title('Class distribution')
   # plt.ylabel('Count')
   # plt.xlabel('Classes')
   # plt.xticks(range(len(class_counts.index)), class_names, rotation=75)  # Set the x-axis tick labels
   # plt.show()

    # Prepend a `background` class to the list of class names
    class_names = ['background']+class_names

    # Get the DataLoaders for training and validation datasets
    train_dataset, valid_dataset = load_labelme_dataset(
        train_keys, val_keys, annotation_df, img_dict,
        class_names=class_names,
        train_tfms=train_tfms,
        valid_tfms=valid_tfms
    )

    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(len(class_names))

    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c*255) for c in color) for color in colors]

    # ---------------------
    # Show sample images with annotations
    # ---------------------

    if args.show_samples:
        # Get the file ID of the first image file
        file_id = train_dataset._img_keys[0]

        # Open the associated image file as a RGB image
        train_sample_img = Image.open(img_dict[file_id]).convert('RGB')

        # Print the dimensions of the image
        print(f"Image Dims: {train_sample_img.size}")

        # Get the row from the 'annotation_df' DataFrame corresponding to the 'file_id'
        annotation_df.loc[file_id].to_frame()

        # Extract the labels for the sample
        labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
        # Extract the polygon points for segmentation mask
        shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
        # Format polygon points for PIL
        xy_coords = [[tuple(p) for p in points] for points in shape_points]
        # Generate mask images from polygons
        mask_imgs = [create_polygon_mask(train_sample_img.size, xy) for xy in xy_coords]
        # Convert mask images to tensors
        masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
        # Generate bounding box annotations from segmentation masks
        bboxes = torchvision.ops.masks_to_boxes(masks)

        # Annotate the sample image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=transforms.PILToTensor()(train_sample_img), 
            masks=masks, 
            alpha=0.3, 
            colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
        )

        # Annotate the sample image with labels and bounding boxes
        annotated_tensor = draw_bboxes(
            image=annotated_tensor, 
            boxes=bboxes, 
            labels=labels, 
            colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
        )

        annot_train_sample_img = tensor_to_pil(annotated_tensor)

        # Get a sample from the validation dataset
        valid_sample_img = valid_dataset[0]  # Get the first sample from the validation dataset
        print(f"Validation Sample Image Dims: {valid_sample_img[0].shape[2]}x{valid_sample_img[0].shape[1]}")

        # Get colors for dataset sample
        sample_colors = [int_colors[int(i.item())] for i in valid_sample_img[1]['labels']]

        # Annotate the sample image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=(valid_sample_img[0]*255).to(dtype=torch.uint8),
            masks=valid_sample_img[1]['masks'], 
            alpha=0.3, 
            colors=sample_colors
        )

        # Annotate the sample image with bounding boxes
        annotated_tensor = draw_bboxes(
            image=annotated_tensor, 
            boxes=valid_sample_img[1]['boxes'], 
            labels=[class_names[int(i.item())] for i in valid_sample_img[1]['labels']], 
            colors=sample_colors
        )

        annot_valid_sample_img = tensor_to_pil(annotated_tensor)

        # Display the annotated images
        fig, axs = plt.subplots(1, 4, figsize=(16, 6))
        images = [
            train_sample_img,
            annot_train_sample_img,
            tensor_to_pil((valid_sample_img[0]*255).to(dtype=torch.uint8)),
            annot_valid_sample_img
        ]
        titles = [
            "Train Sample (Raw)",
            "Train Sample (Annotated)",
            "Validation Sample (Raw)",
            "Validation Sample (Annotated)"
        ]
        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.show()


    # ---------------------
    # Run Hyperparameter Optimization
    # ---------------------

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create study with pruning
    pruner = optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        patience=10
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=f"maskrcnn_optimization_{timestamp}"
    )

    print("Starting hyperparameter optimization...")

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, 
                                    train_dataset, valid_dataset, 
                                    args.result_folder, 
                                    param_space, 
                                    len(class_names),
                                    class_names,
                                    num_workers=training_config.get('num_workers', 4),
                                    patience=training_config.get('patience', 4),
                                    top_k=training_config.get('top_k', 5),
                                    model_version=model_config.get('version', 1)
                                   ),
            n_trials=training_config.get('trials', 20),
            timeout=None,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
    except Exception as e:
        # Catch-all to avoid crashing and try to save study and partial results
        print(f"Study optimize raised an exception: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Analyze results
    print("\nAnalyzing results...")
    analyze_study_results(study, os.path.join(args.result_folder,"analysis"))

    # Print study statistics
    print("Study statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    if complete_trials:
        print("\nTop 5 best trials:")
        best_trials = sorted(complete_trials, key=lambda t: t.value)[:5]
        best_configs = []
        for rank, trial in enumerate(best_trials, start=1):
            print(f"Rank {rank}:")
            print(f"  Trial Number: {trial.number}")
            print(f"  Value (Validation Loss): {trial.value:.6f}")
            print("  Parameters:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            best_configs.append({
            "rank": rank,
            "trial_number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "timestamp": timestamp,
            "checkpoint_path": f"trial_{trial.number}/best_model_trial_{trial.number}.pth",
            "study_name": study.study_name
            })

        # Save top 5 configurations
        config_file = os.path.join(args.result_folder,"top5_configurations.json")
        with open(config_file, 'w') as f:
            json.dump(best_configs, f, indent=4)
        
        # Save study object
        study_file = os.path.join(args.result_folder,"study.pkl")
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        print(f"\nResults saved to:")
        print(f"Best configuration: {config_file}")
        print(f"Study object: {study_file}")

        print("Best hyperparameters found:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
    else:
        print("No trials completed successfully")

    # Create report including confusion matrices
    print("\nCreating report...")
    try:
        report_file = create_report(study, args.result_folder, class_names)
        print(f"Report created: {report_file}")
    except Exception as e:
        print(f"Failed to create report: {e}")
        traceback.print_exc()




def is_top_trial(current_loss, result_folder, current_trial_number, top_k=5):
    """
    Check if the current trial should save its model checkpoint based on performance ranking.
    Only save checkpoints for top K performing trials to save disk space.
    Also cleans up checkpoints (but not confusion matrices) for trials that fall out of top K.

    Args:
        current_loss (float): Current trial's best validation loss
        result_folder (str): Path to results folder to check existing trials
        current_trial_number (int): Current trial number
        top_k (int): Number of top trials to keep model checkpoints for
    
    Returns:
        bool: True if this trial should save checkpoint and confusion matrix, False otherwise
    """
    try:
        # If current loss is inf, don't save
        if current_loss == float('inf'):
            print(f"Warning: Trial {current_trial_number} has infinite loss, not saving checkpoint")
            return False
            
        # Get all existing trial summary files to extract validation losses
        trial_data = []  # List of (trial_number, validation_loss) tuples
        
        for item in os.listdir(result_folder):
            if item.startswith('trial_') and os.path.isdir(os.path.join(result_folder, item)):
                try:
                    trial_number = int(item.split('_')[1])
                except ValueError:
                    continue
                    
                summary_file = os.path.join(result_folder, item, f"{item}_summary.json")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                            best_loss = data.get('results', {}).get('best_validation_loss')
                            if best_loss is not None and best_loss != float('inf'):
                                trial_data.append((trial_number, best_loss))
                    except Exception:
                        continue
        
        # Add current trial to the list for evaluation
        trial_data.append((current_trial_number, current_loss))
        
        # If we have fewer than top_k trials, always save
        if len(trial_data) <= top_k:
            return True
        
        # Sort by validation loss (ascending - lower is better)
        trial_data.sort(key=lambda x: x[1])
        
        # Get current top K trials
        current_top_k = trial_data[:top_k]
        current_top_k_numbers = [trial[0] for trial in current_top_k]
        
        # Find trials that should be cleaned up (those with checkpoints/confusion matrices but not in top K)
        all_trial_numbers = [trial[0] for trial in trial_data]
        trials_to_cleanup = [num for num in all_trial_numbers if num not in current_top_k_numbers]
        
        # Clean up artifacts for trials that fell out of top K
        for trial_num in trials_to_cleanup:
            trial_dir = os.path.join(result_folder, f"trial_{trial_num}")
            if os.path.exists(trial_dir):
                # Remove checkpoint
                checkpoint_path = os.path.join(trial_dir, f"best_model_trial_{trial_num}.pth")
                if os.path.exists(checkpoint_path):
                    try:
                        os.remove(checkpoint_path)
                        print(f"🗑️  Removed checkpoint for trial {trial_num} (fell out of top {top_k})")
                    except Exception as e:
                        print(f"Failed to remove checkpoint for trial {trial_num}: {e}")
        
        # Check if current trial is in the top K
        is_top = current_trial_number in current_top_k_numbers
        
        return is_top
        
    except Exception as e:
        # If anything goes wrong, err on the side of caution and save
        print(f"Warning: Failed to check if trial should save checkpoint: {e}")
        return True


def objective(trial, train_dataset, valid_dataset,
              result_folder, param_space, num_classes, class_names, num_workers=4,
              patience=5, top_k=5, model_version=1):
    """
    Modular Optuna objective function.
    
    Parameters:
        trial: optuna trial object
        model: model to be trained and validated
        train_dataset, val_dataset: datasets for training and validation
        result_folder: folder path to save the results
        param_space: dict of hyperparameters configuration
        num_classes: number of classes in the dataset (including background)
        class_names: list of class names for confusion matrix
        num_workers: number of workers for data loading
        patience: patience for early stopping
        top_k: number of top trials to save checkpoints for
        model_version: version of the model to use ('maskrcnn_resnet50_fpn' or 'maskrcnn_resnet50_fpn_v2')
    """

    # --- Sample hyperparameters ---
    lr = trial.suggest_float(
        "lr",
        param_space["lr"]["low"],
        param_space["lr"]["high"],
        log=param_space["lr"].get("log", False)
    )

    batch_size = trial.suggest_categorical("batch_size", param_space["batch_size"]["choices"])
    optimizer_name = trial.suggest_categorical("optimizer", param_space["optimizer"]["choices"])

    num_epochs = trial.suggest_int(
        "num_epochs",
        param_space["num_epochs"]["low"],
        param_space["num_epochs"]["high"],
        step=param_space["num_epochs"].get("step", 1)  # default step=1
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        param_space["weight_decay"]["low"],
        param_space["weight_decay"]["high"],
        log=param_space["weight_decay"].get("log", False)
    )
    max_grad_norm = trial.suggest_float(
        "max_grad_norm",
        param_space["max_grad_norm"]["low"],
        param_space["max_grad_norm"]["high"],
        log=param_space["max_grad_norm"].get("log", False)  # default log=False
        
    )

    use_grad_clip = trial.suggest_categorical("use_grad_clip", param_space["use_grad_clip"]["choices"])
    use_scheduler = trial.suggest_categorical("use_scheduler", param_space["use_scheduler"]["choices"])


    config = {
        "lr": lr, "batch_size": batch_size,
        "optimizer": optimizer_name, "num_epochs": num_epochs,
        "weight_decay": weight_decay, "use_grad_clip": use_grad_clip,
        "max_grad_norm": max_grad_norm, "use_scheduler": use_scheduler
    }

    trial_dir = os.path.join(result_folder, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # --- Data loaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': batch_size,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': device.type == "cuda",  # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        "pin_memory_device": str(device) if device.type == "cuda" else "",  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': lambda batch: tuple(zip(*batch)),
    }

    # Create DataLoader for training data. Data is shuffled for every epoch.
    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

    # Create DataLoader for validation data. Shuffling is not necessary for validation data.
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    # --- Model ---
    model = create_model(num_classes, model_version=model_version)
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn' if model_version==1 else 'maskrcnn_resnet50_fpn_v2'
    print(f"Using model: {model.name}")
    model.to(device)

    # --- Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(params, optimizer_name, lr, weight_decay)

    # --- Scheduler ---
    scheduler = None
    if use_scheduler:
        print(f"Using learning rate scheduler)")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param_space["lr_scheduler"]["step_size"], gamma=param_space["lr_scheduler"]["gamma"])

    # --- Training loop ---
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, patience_counter = patience, 0
    stopping_reason = "in_progress"

    try:
        for epoch in range(num_epochs):            
            model.train()
            epoch_loss = 0.0
            for imgs, annotations in tqdm(train_dataloader, desc=f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs} - Training"):
                # Move data to device
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in annot.items()} for annot in annotations]

                optimizer.zero_grad()
                loss_dict = model(imgs, targets)
                loss = sum(loss for loss in loss_dict.values())

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                    raise optuna.TrialPruned()

                loss.backward()
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()

                # immediate cleanup of large objects used in the batch
                del loss_dict, loss
                del targets
                del imgs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # Calculate training metrics on a subset of data (to avoid slowing down training too much)
            train_metrics = calculate_training_metrics(model, train_dataloader, device, max_batches=5)

            # Validation
            avg_val_loss, val_preds, val_targets, val_metrics = validate_model(model, valid_dataloader, device)
            val_losses.append(avg_val_loss)

            # # Print metrics for each epoch
            # print(f"\nEpoch {epoch+1}/{num_epochs} - Trial {trial.number} Results:")
            # print(f"  Training Loss: {avg_train_loss:.6f}")
            # print(f"  Training Metrics (subset):")
            # print(f"    Accuracy: {train_metrics['accuracy']:.4f}")
            # print(f"    Precision (weighted): {train_metrics['precision_weighted']:.4f}")
            # print(f"    Recall (weighted): {train_metrics['recall_weighted']:.4f}")
            # print(f"    F1-score (weighted): {train_metrics['f1_weighted']:.4f}")
            # print(f"    Predictions/Targets: {train_metrics['num_predictions']}/{train_metrics['num_targets']}")
            # print(f"  Validation Loss: {avg_val_loss:.6f}")
            # print(f"  Validation Metrics:")
            # print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
            # print(f"    Precision (weighted): {val_metrics['precision_weighted']:.4f}")
            # print(f"    Recall (weighted): {val_metrics['recall_weighted']:.4f}")
            # print(f"    F1-score (weighted): {val_metrics['f1_weighted']:.4f}")
            # print(f"    Precision (macro): {val_metrics['precision_macro']:.4f}")
            # print(f"    Recall (macro): {val_metrics['recall_macro']:.4f}")
            # print(f"    F1-score (macro): {val_metrics['f1_macro']:.4f}")
            # print(f"    Predictions/Targets: {val_metrics['num_predictions']}/{val_metrics['num_targets']}")
            
            # Print validation loss for the epoch
            print(f"Validation Loss: {avg_val_loss:.6f}")

            # Save epoch metrics to file
            metrics_file = os.path.join(trial_dir, f"epoch_metrics_trial_{trial.number}.json")
            best_checkpoint_path = os.path.join(trial_dir, f"best_model_trial_{trial.number}.pth")
            if epoch == num_epochs - 1:
                stopping_reason = "completed"
            
            save_epoch_metrics(epoch + 1, avg_train_loss, train_metrics, avg_val_loss, val_metrics, 
                              metrics_file, config, trial.number)


            # Update best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save checkpoint for the new best model
                print(f"New best validation loss: {best_val_loss:.6f} at epoch {epoch+1}")
                if is_top_trial(best_val_loss, result_folder, trial.number, top_k=top_k):
                    print(f"Saving checkpoint for trial {trial.number} with loss {best_val_loss:.6f}")
                    save_cpu_checkpoint(model, optimizer, epoch, best_val_loss, config, best_checkpoint_path)
                else:
                    print(f"Skipping checkpoint save for trial {trial.number} with loss {best_val_loss:.6f} (not in top {top_k})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                    stopping_reason = "early_stopping"
                    break

            if scheduler:
                scheduler.step()

            # Report to Optuna for pruning (report current validation loss, not best)
            trial.report(avg_val_loss, epoch)
                
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Training completed successfully
        print(f"\n{'='*60}")
        print(f"Training completed for Trial {trial.number}")
        print(f"Best validation loss achieved: {best_val_loss:.6f}")
        print(f"Total epochs completed: {epoch+1}")
        print(f"{'='*60}")

        # Determine final stopping reason if not set by early stopping
        if 'stopping_reason' not in locals():
            stopping_reason = "completed"

        cm = None
        cm_path = None

        # Generate confusion matrix for the best model
        if os.path.exists(best_checkpoint_path):
            print("Generating confusion matrix...")
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

                cm_save_path = os.path.join(trial_dir, f"confusion_matrix_trial_{trial.number}.png")
                cm = generate_confusion_matrix(model, valid_dataloader, device, class_names, cm_save_path)
                if cm is not None:
                    print("Confusion matrix generated successfully")
                    cm_path = f"confusion_matrix_trial_{trial.number}.png"
            except Exception as cm_e:
                print(f"Failed to generate confusion matrix: {cm_e}")
            traceback.print_exc()
        else:
            print("No checkpoint found for confusion matrix generation")

        # Always save trial summary and plot training curves
        summary_file = os.path.join(trial_dir, f"trial_{trial.number}_summary.json")
        save_trial_summary(trial.number, config, best_val_loss, epoch+1, train_losses, val_losses, summary_file, cm_path,
                           stopping_reason=stopping_reason)

        plot_training_curves(train_losses, val_losses, os.path.join(trial_dir, f"training_curves_trial_{trial.number}.png"))

    except optuna.TrialPruned:
        print(f"\n--- Trial {trial.number} was pruned by Optuna ---")
        
        # Save final epoch metrics with pruning information
        if 'epoch' in locals() and 'train_losses' in locals() and 'val_losses' in locals():
            try:
                # Save pruned trial summary
                summary_file = os.path.join(trial_dir, f"trial_{trial.number}_summary.json")
                save_trial_summary(
                    trial_number=trial.number,
                    config=config,
                    best_val_loss=best_val_loss if best_val_loss < float('inf') else None,
                    total_epochs=epoch + 1 if 'epoch' in locals() else 0,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    save_path=summary_file,
                    stopping_reason="pruned"
                )
                
            except Exception as save_e:
                print(f"Failed to save pruned trial information: {save_e}")
        
        # Clean up GPU memory after pruning
        try:
            del model, optimizer, scheduler, train_dataloader, valid_dataloader
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise  # Re-raise to let Optuna handle the pruning properly

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        msg = str(e).lower()
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)
        print(f"\n--- Exception in trial {trial.number}: {type(e).__name__} ---")
        traceback.print_exc()

        # Save the best model found so far as CPU checkpoint
        partial_path = os.path.join(trial_dir, f"best_model_trial_{trial.number}_partial.pth")
        try:
            if best_val_loss < float('inf'):
                print(f"Saving partial best checkpoint to {partial_path}")
                save_cpu_checkpoint(model, optimizer, epoch if 'epoch' in locals() else -1, best_val_loss, config, partial_path)
            else:
                print(f"No best model found: saving current model state (cpu) to {partial_path}")
                save_cpu_checkpoint(model, optimizer, epoch if 'epoch' in locals() else -1, None, config, partial_path)
        except Exception as save_e:
            print(f"Failed saving partial checkpoint: {save_e}")
            traceback.print_exc()

        # cleanup GPU memory
        try:
            del model, optimizer, scheduler, train_dataloader, valid_dataloader
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sentinel = best_val_loss if best_val_loss < float('inf') else 1e6
        print(f"Trial {trial.number} returning sentinel value {sentinel} after exception.")
        
        # Save partial trial summary
        summary_file = os.path.join(trial_dir, f"trial_{trial.number}_summary_partial.json")
        try:
            save_trial_summary(trial.number, config, best_val_loss, epoch + 1 if 'epoch' in locals() else 0, 
                             train_losses, val_losses, summary_file, confusion_matrix_path=None,
                             stopping_reason="error")
        except Exception as summary_e:
            print(f"Failed to save partial trial summary: {summary_e}")
        
        return sentinel

    except Exception as e:
        print(f"\n--- Unexpected exception in trial {trial.number}: {type(e).__name__} ---")
        traceback.print_exc()
        partial_path = os.path.join(trial_dir, f"best_model_trial_{trial.number}_partial.pth")
        try:
            save_cpu_checkpoint(model, optimizer, epoch if 'epoch' in locals() else -1, None, config, partial_path)
        except Exception as save_e:
            print(f"Failed saving partial checkpoint: {save_e}")
            traceback.print_exc()
        try:
            del model, optimizer, scheduler, train_dataloader, valid_dataloader
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sentinel = best_val_loss if best_val_loss < float('inf') else 1e6
        
        # Save partial trial summary
        summary_file = os.path.join(trial_dir, f"trial_{trial.number}_summary_partial.json")
        try:
            save_trial_summary(
                trial_number=trial.number, 
                config=config, 
                best_val_loss=best_val_loss,
                total_epochs=epoch + 1 if 'epoch' in locals() else 0, 
                train_losses=train_losses, 
                val_losses=val_losses, 
                save_path=summary_file,
                confusion_matrix_path=None,
                stopping_reason="error"
            )
        except Exception as summary_e:
            print(f"Failed to save partial trial summary: {summary_e}")
        
        return sentinel

    # final cleanup after successful trial
    try:
        del model, optimizer, scheduler, train_dataloader, valid_dataloader
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up any partial checkpoints since trial completed successfully
    cleanup_partial_checkpoints(trial_dir, trial.number)

    return best_val_loss


def analyze_study_results(study, save_dir="study_analysis"):
    """Comprehensive analysis of the optimization study"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all completed trials
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials to analyze")
        return
    
    # Create DataFrame for analysis
    trial_data = []
    for trial in completed_trials:
        data = trial.params.copy()
        data['trial_number'] = trial.number
        data['value'] = trial.value
        trial_data.append(data)
    
    df = pd.DataFrame(trial_data)
    
    # Save trial data
    df.to_csv(os.path.join(save_dir, "trial_results.csv"), index=False)
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        
        plt.figure(figsize=(10, 6))
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Parameter Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "parameter_importance.png"))
        plt.close()
        
    except Exception as e:
        print(f"Could not compute parameter importance: {e}")
    
    # Optimization history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    trial_numbers = [t.number for t in completed_trials]
    trial_values = [t.value for t in completed_trials]
    plt.plot(trial_numbers, trial_values, 'b-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Loss')
    plt.title('Optimization History')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    best_values = []
    best_so_far = float('inf')
    for value in trial_values:
        if value < best_so_far:
            best_so_far = value
        best_values.append(best_so_far)
    
    plt.plot(trial_numbers, best_values, 'r-o')
    plt.xlabel('Trial Number')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Value Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimization_history.png"))
    plt.close()
    
    # Parameter distributions for best trials
    top_n = min(5, len(completed_trials))
    best_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]
    
    print(f"\nTop {top_n} trials:")
    for i, trial in enumerate(best_trials):
        print(f"Trial {trial.number} (Rank {i+1}): Loss = {trial.value:.6f}")
        print(f"  Parameters: {trial.params}")
        print()


if __name__ == "__main__":
    main()
