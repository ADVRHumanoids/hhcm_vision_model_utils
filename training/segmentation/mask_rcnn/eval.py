#!/usr/bin/env python3
"""
Evaluate a trained Mask R-CNN model on a test dataset.
This script loads a pre-trained model from a specified checkpoint and evaluates its performance
on a subset of images from the test dataset.
Arguments:
    --checkpoint: Path to the model checkpoint file.
    --test-folder: Path to the folder containing the test images and annotations.
    --images: Number of images to evaluate (default: 10).
    --threshold: Confidence threshold for predictions (default: 0.5).
    --seed: Random seed for reproducibility (optional).
"""

# Import Python Standard Library dependencies
import os
import json
import random
import argparse
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Import utility functions
from cjm_pil_utils.core import get_img_files, stack_imgs
from cjm_pytorch_utils.core import tensor_to_pil, move_data_to_device

# Import the pandas package
import pandas as pd

# Import the numpy package
import numpy as np

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PyTorch dependencies
import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.v2  as transforms

# Import the distinctipy module
from distinctipy import distinctipy

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import custom utility functions
from utils import (
    load_model_checkpoint,
    draw_bboxes,
    draw_segmentation_masks,
    create_polygon_mask,
    Mask
)

def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--test-folder", type=str, required=True, help="Path to test data folder")
    parser.add_argument("--images", type=int, default=10, help="Number of images to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions")
    parser.add_argument("--seed", type=int, required=False, help="Random seed for reproducibility")
    parser.add_argument("--model-version", type=int, default=1, choices=[1, 2], help="Model version to use (1 or 2)")
    parser.add_argument("--yolo", action="store_true", default=False, help="Use YOLODataset folder structure to test")
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if os.path.exists(args.test_folder):
        if getattr(args, "yolo", False):
            print("Using YOLODataset folder structure for test data.")
            # Check if 'test' folder exists, else fallback to 'val' folder
            yolo_base = Path(args.test_folder + "/YOLODataset/images")
            test_folder_path = yolo_base / "test"
            if not test_folder_path.exists():
                # Try fallback to 'val' folder in the same parent directory
                val_folder_path = yolo_base / "val"
                if not val_folder_path.exists():
                    raise FileNotFoundError(
                        f"Neither test folder nor val folder found:\n"
                        f"Test: {test_folder_path}\nVal: {val_folder_path}"
                    )
                print(f"Test folder not found. Falling back to val folder: {val_folder_path}")
                test_folder_path = val_folder_path
            else:
                print(f"Using test folder: {test_folder_path}")
        else:
            print("Using standard folder structure for test data.")
            test_folder_path = Path(args.test_folder)
    else:
        raise FileNotFoundError(f"Test folder not found: {args.test_folder}")
    # Get a list of image files in the dataset folder
    img_file_paths = get_img_files(str(test_folder_path))
    print(f"Found {len(img_file_paths)} image files in folder: {test_folder_path}")
    if not img_file_paths:
        raise FileNotFoundError(f"No image files found in folder: {test_folder_path}")
    
    # Get a list of JSON annotation files in the dataset folder
    json_folder_path = Path(args.test_folder)
    print(f"Looking for annotation JSON files in folder: {json_folder_path}")
    annotation_file_paths = list(json_folder_path.glob('*.json'))
    if args.yolo:
        # Filter out any JSON files based on the name file in the images folder
        annotation_file_paths = [f for f in annotation_file_paths if f.stem in {img.stem for img in img_file_paths}]
        print(f"Filtered annotation files to match image files. {len(annotation_file_paths)} annotation files remain.")
        # Change image file path to match same folder of the annotation files
        img_file_paths = [f.with_suffix('.jpg') for f in annotation_file_paths]  # Assuming images are in .jpg format; adjust if necessary
    if not annotation_file_paths:
        raise FileNotFoundError(f"No annotation JSON files found in folder: {json_folder_path}")
    
    # Create a dictionary that maps file names to file paths
    img_dict = {file.stem: file for file in img_file_paths}
    print(f"Number of Images: {len(img_dict)}")
    
    # Load annotation dataframes
    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)
    
    # Assign the image file name as the index for each row
    annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
    annotation_df = annotation_df.set_index('index')
    
    # Sanity check: keep only rows that correspond to images in img_dict
    annotation_df = annotation_df.loc[list(img_dict.keys())]
    if annotation_df.empty:
        raise ValueError("No matching annotation entries for image files found.")

    # Keep only the rows that correspond to the filenames in the 'img_dict' dictionary
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    # Explode the 'shapes' column in the annotation_df dataframe
    # Convert the resulting series to a dataframe and rename the 'shapes' column to 'shapes'
    # Apply the pandas Series function to the 'shapes' column of the dataframe
    shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

    # Get a list of unique labels in the 'annotation_df' DataFrame
    class_names = shapes_df['label'].unique().tolist()

    # Prepend a `background` class to the list of class names
    class_names = ['background']+class_names

    # Generate a list of colors with a length equal to the number of labels
    colors = distinctipy.get_colors(len(class_names))

    # Make a copy of the color map in integer format
    int_colors = [tuple(int(c*255) for c in color) for color in colors]

    # Get the list of image IDs
    if args.seed is not None:
        random.seed(args.seed)

    img_keys = random.sample(list(img_dict.keys()), k=args.images)
    for i in tqdm(range(len(img_keys)), desc="Evaluating images"):
        # Load the test image
        file_id = img_keys[i]
        test_img = Image.open(img_dict[file_id]).convert("RGB")
        print(f"Evaluating image: {file_id}")

        # Extract the polygon points for segmentation mask
        target_shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
        # Format polygon points for PIL
        target_xy_coords = [[tuple(p) for p in points] for points in target_shape_points]
        # Generate mask images from polygons
        target_mask_imgs = [create_polygon_mask(test_img.size, xy) for xy in target_xy_coords]
        # Convert mask images to tensors
        target_masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in target_mask_imgs]))

        # Get the target labels and bounding boxes
        target_labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
        target_bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(target_masks), format='xyxy', canvas_size=test_img.size[::-1])

        # Get the best trial directory
        print(f"Loading best model from: {args.checkpoint}")

        # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_checkpoint(len(class_names), args.checkpoint, device=device)
        model.eval()

        # Ensure the model and input data are on the same device
        model.to(device)
        input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(test_img)[None].to(device)

        # Make a prediction with the model
        with torch.no_grad():
            model_output = model(input_tensor)

        # Move model output to the CPU
        model_output = move_data_to_device(model_output, 'cpu')
        print(f"Model returned {len(model_output[0]['boxes'])} boxes")

        # Filter the output based on the confidence threshold
        scores_mask = model_output[0]['scores'] > args.threshold
        print(f"{scores_mask.sum()} boxes remaining after thresholding at {args.threshold}")
        if scores_mask.sum() == 0:
            print("No detections above the confidence threshold.")
            continue

        # Scale the predicted bounding boxes
        pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask], format='xyxy', canvas_size=test_img.size[::-1])

        # Get the class names for the predicted label indices
        pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]

        # Extract the confidence scores
        pred_scores = model_output[0]['scores']

        # Scale and stack the predicted segmentation masks
        pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=test_img.size[::-1])
        pred_masks = torch.concat([Mask(torch.where(mask >= args.threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])

        # Get the annotation colors for the targets and predictions
        target_colors=[int_colors[i] for i in [class_names.index(label) for label in target_labels]]
        pred_colors=[int_colors[i] for i in [class_names.index(label) for label in pred_labels]]

        # Convert the test images to a tensor
        img_tensor = transforms.PILToTensor()(test_img)

        # Annotate the test image with the target segmentation masks
        annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=target_masks, alpha=0.3, colors=target_colors)
        # Annotate the test image with the target bounding boxes
        annotated_tensor = draw_bboxes(image=annotated_tensor, boxes=target_bboxes, labels=target_labels, colors=target_colors)
        # Display the annotated test image
        annotated_test_img = tensor_to_pil(annotated_tensor)

        # Annotate the test image with the predicted segmentation masks
        annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.3, colors=pred_colors)
        # Annotate the test image with the predicted labels and bounding boxes
        annotated_tensor = draw_bboxes(
            image=annotated_tensor, 
            boxes=pred_bboxes, 
            labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)],
            colors=pred_colors
        )

        # Display the annotated test image with the predicted bounding boxes
        stacked_img = stack_imgs([annotated_test_img, tensor_to_pil(annotated_tensor)])
        
        # Create full screen figure
        fig = plt.figure(figsize=(16, 10))  # Large figure size
        fig.canvas.manager.full_screen_toggle()  # Toggle to full screen
        
        # Display image
        plt.imshow(stacked_img)
        plt.axis('off')
        plt.title(f"Evaluation Results - Image: {file_id}", fontsize=16, pad=20)
        
        # Add text with instructions
        plt.figtext(0.02, 0.02, "Press ESC to close and continue, or close window manually", 
                   fontsize=12, ha='left', va='bottom', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Add key press event handler
        def on_key_press(event):
            if event.key == 'escape':
                plt.close(fig)
        
        # Connect the key press event
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Show the plot and wait for user interaction
        plt.tight_layout()
        plt.show()

        # Print the prediction data as a Pandas DataFrame for easy formatting
        pd.Series({
            "Target BBoxes:": [f"{label}:{bbox}" for label, bbox in zip(target_labels, np.round(target_bboxes.numpy(), decimals=3))],
            "Predicted BBoxes:": [f"{label}:{bbox}" for label, bbox in zip(pred_labels, pred_bboxes.round(decimals=3).numpy())],
            "Confidence Scores:": [f"{label}: {prob*100:.2f}%" for label, prob in zip(pred_labels, pred_scores)]
        }).to_frame().style.hide(axis='columns')




if __name__ == "__main__":
    main()
