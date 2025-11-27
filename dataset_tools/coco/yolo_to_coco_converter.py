#!/usr/bin/env python3
"""!
@file converter.py
@brief Convert YOLO segmentation dataset to COCO format.

@details This script converts YOLO11 segmentation datasets to COCO format, supporting both
         split preservation (train/val/test) and merged single-folder output. The script
         handles normalized polygon coordinates from YOLO format and converts them to
         absolute pixel coordinates in COCO format, including bounding boxes and area
         calculations.

@author Alessio Lovato
@date 2025-11-19

@section usage Usage
@code{.sh}
# Convert with existing splits (train/val/test folders)
python yolo_to_coco_converter.py --input-dir /path/to/yolo/dataset --output-dir ./coco-output

# Merge all splits into single folder
python yolo_to_coco_converter.py --input-dir /path/to/yolo/dataset --output-dir ./coco-merged --merge
@endcode

@section args Arguments
  --input-dir INPUT_DIR      Path to YOLO dataset directory (required)
  --output-dir OUTPUT_DIR    Output directory for COCO format dataset (default: ./coco-segmentation-output)
  --merge                    Merge all splits into a single dataset folder (optional)

@section structure Expected Input Structure
@code
dataset/
├── dataset.yaml           # Contains class names
├── labels/
│   ├── train/            # Training labels (.txt files)
│   ├── val/              # Validation labels (.txt files)
│   └── test/             # Test labels (.txt files, optional)
└── images/
    ├── train/            # Training images
    ├── val/              # Validation images
    └── test/             # Test images (optional)
@endcode

@section output Output Structure
Without --merge:
@code
output-dir/
├── train/
│   ├── annotations.json
│   └── [images]
├── val/
│   ├── annotations.json
│   └── [images]
└── test/
    ├── annotations.json
    └── [images]
@endcode

With --merge:
@code
output-dir/
├── annotations.json
└── [all images]
@endcode
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
from datetime import datetime
import random

def yolo_to_coco_segmentation(yolo_dir, output_dir, images_dir=None, copy_images=True):
    """
    Convert YOLO segmentation dataset to COCO format with images.
    
    Args:
        yolo_dir: Path to YOLO labels directory
        output_dir: Output directory for COCO format (will create annotations.json and copy images)
        images_dir: Path to images directory (if None, assumes same parent as yolo_dir)
        copy_images: Whether to copy images to output directory
    """
    
    # Initialize COCO structure
    coco_format = {
        "info": {
            "description": "YOLO to COCO Conversion",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Set paths
    yolo_path = Path(yolo_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if images_dir is None:
        # Assume images are in parallel directory structure
        # labels/train -> images/train
        images_path = yolo_path.parent.parent / "images" / yolo_path.name
    else:
        images_path = Path(images_dir)
    
    # Load class names from classes.txt or data.yaml
    # Look in the dataset root directory (parent of labels directory)
    classes = load_classes(yolo_path.parent.parent)
    
    # Create categories
    for idx, class_name in enumerate(classes):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })
    
    # Process each label file
    image_id = 0
    annotation_id = 0
    
    for label_file in sorted(yolo_path.glob("*.txt")):
        # Find corresponding image
        image_name = label_file.stem
        image_file = find_image_file(images_path, image_name)
        
        if image_file is None:
            print(f"Warning: Image not found for {label_file.name}")
            continue
        
        # Get image dimensions
        try:
            img = Image.open(image_file)
            width, height = img.size
        except Exception as e:
            print(f"Error reading image {image_file}: {e}")
            continue
        
        # Copy image to output directory if requested
        if copy_images:
            dest_image = output_path / image_file.name
            if not dest_image.exists():
                shutil.copy2(image_file, dest_image)
        
        # Add image info
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })
        
        # Read YOLO annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Need at least class + 2 points
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixel coordinates
            segmentation = []
            for i in range(0, len(coords), 2):
                x = coords[i] * width
                y = coords[i + 1] * height
                segmentation.extend([x, y])
            
            # Calculate bounding box from segmentation
            x_coords = segmentation[0::2]
            y_coords = segmentation[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            bbox_width = max(x_coords) - x_min
            bbox_height = max(y_coords) - y_min
            
            # Calculate area (approximate using bounding box)
            area = bbox_width * bbox_height
            
            # Add annotation
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON as annotations.json
    output_json = output_path / "annotations.json"
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Conversion complete!")
    print(f"Total images: {image_id}")
    print(f"Total annotations: {annotation_id}")
    print(f"Output saved to: {output_json}")
    
    return image_id, annotation_id

def load_classes(dataset_path):
    """Load class names from classes.txt or data.yaml"""
    dataset_path = Path(dataset_path)
    
    # Try classes.txt first
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # Try data.yaml or dataset.yaml
    for yaml_name in ['data.yaml', 'dataset.yaml']:
        yaml_file = dataset_path / yaml_name
        if yaml_file.exists():
            import yaml
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    # Handle both list and dict formats
                    names = data['names']
                    if isinstance(names, dict):
                        # Convert dict to list, sorted by key
                        max_idx = max(names.keys())
                        return [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
                    return names
    
    # Default classes if no file found
    print("Warning: No classes.txt or data.yaml found. Using default classes.")
    return [f"class_{i}" for i in range(80)]

def find_image_file(images_path, image_name):
    """Find image file with various extensions"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for ext in extensions:
        image_file = images_path / f"{image_name}{ext}"
        if image_file.exists():
            return image_file
    return None

def split_dataset(all_files, train_percent, val_percent, test_percent):
    """Split dataset into train/val/test sets"""
    random.shuffle(all_files)
    total = len(all_files)
    
    train_size = int(total * train_percent / 100)
    val_size = int(total * val_percent / 100)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    return train_files, val_files, test_files


def convert_split(label_files, images_path, output_dir, classes, split_name):
    """Convert a specific split to COCO format"""
    print(f"\n{'='*50}")
    print(f"Converting {split_name} set ({len(label_files)} images)...")
    print(f"{'='*50}")
    
    # Initialize COCO structure
    coco_format = {
        "info": {
            "description": "YOLO to COCO Conversion",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    for idx, class_name in enumerate(classes):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each label file
    image_id = 0
    annotation_id = 0
    
    for label_file in sorted(label_files):
        # Find corresponding image
        image_name = label_file.stem
        
        # For merged datasets, try to find image in the same split as the label
        # label file path structure: .../labels/train/file.txt -> .../images/train/
        label_split = label_file.parent.name if label_file.parent.name in ['train', 'val', 'test'] else None
        
        if label_split and images_path.name != label_split:
            # Images are organized in splits (train/val/test)
            actual_images_path = images_path / label_split
        else:
            actual_images_path = images_path
        
        image_file = find_image_file(actual_images_path, image_name)
        
        if image_file is None:
            print(f"Warning: Image not found for {label_file.name}")
            continue
        
        # Get image dimensions
        try:
            img = Image.open(image_file)
            width, height = img.size
        except Exception as e:
            print(f"Error reading image {image_file}: {e}")
            continue
        
        # Copy image to output directory
        dest_image = output_path / image_file.name
        if not dest_image.exists():
            shutil.copy2(image_file, dest_image)
        
        # Add image info
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })
        
        # Read YOLO annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Need at least class + 2 points
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixel coordinates
            segmentation = []
            for i in range(0, len(coords), 2):
                x = coords[i] * width
                y = coords[i + 1] * height
                segmentation.extend([x, y])
            
            # Calculate bounding box from segmentation
            x_coords = segmentation[0::2]
            y_coords = segmentation[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            bbox_width = max(x_coords) - x_min
            bbox_height = max(y_coords) - y_min
            
            # Calculate area (approximate using bounding box)
            area = bbox_width * bbox_height
            
            # Add annotation
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON as annotations.json
    output_json = output_path / "annotations.json"
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"✓ Images: {image_id}")
    print(f"✓ Annotations: {annotation_id}")
    print(f"✓ Output: {output_path}")
    
    return image_id, annotation_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert YOLO segmentation dataset to COCO format using existing splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python converter.py --input-dir /path/to/yolo/dataset --output-dir ./coco-output
        """
    )
    
    parser.add_argument('--input-dir', type=str,
                        help='Path to YOLO dataset directory')
    parser.add_argument('--output-dir', type=str, 
                        default='./coco-segmentation-output',
                        help='Output directory for COCO format dataset')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all splits into a single dataset folder')
    
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    # Load classes
    classes = load_classes(base_dir)
    print(f"\nClasses found: {classes}")
    
    if args.merge:
        # Merge all splits into a single dataset
        print("\n" + "="*70)
        print("Merging all splits into single dataset")
        print("="*70)
        
        all_label_files = []
        for split in ['train', 'val', 'test']:
            labels_dir = base_dir / "labels" / split
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                if label_files:
                    all_label_files.extend(label_files)
                    print(f"Found {len(label_files)} files in {split}")
        
        if not all_label_files:
            print("\nError: No valid train/val/test splits found!")
            print("Expected structure: dataset/labels/train/, dataset/labels/val/, dataset/labels/test/ (optional)")
            exit(1)
        
        print(f"\nTotal files to merge: {len(all_label_files)}")
        
        # Convert all files to a single output directory
        images_base = base_dir / "images"
        convert_split(all_label_files, images_base, output_base, classes, 'merged')
    else:
        # Use existing train/val/test directory structure
        print("\n" + "="*70)
        print("Converting existing train/val/test splits")
        print("="*70)
        
        found_splits = False
        for split in ['train', 'val', 'test']:
            labels_dir = base_dir / "labels" / split
            images_dir = base_dir / "images" / split
            output_dir = output_base / split
            
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                if label_files:
                    found_splits = True
                    convert_split(label_files, images_dir, output_dir, classes, split)
                else:
                    print(f"Info: {split} directory exists but contains no label files")
            else:
                if split != 'test':  # test is optional
                    print(f"Warning: {split} directory not found: {labels_dir}")
        
        if not found_splits:
            print("\nError: No valid train/val/test splits found!")
            print("Expected structure: dataset/labels/train/, dataset/labels/val/, dataset/labels/test/ (optional)")
            exit(1)
    
    print("\n" + "="*70)
    print("✓ Conversion completed successfully!")
    print("="*70)
