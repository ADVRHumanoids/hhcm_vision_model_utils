# COCO Format Conversion Tools

This directory contains utilities for converting datasets to COCO (Common Objects in Context) format, which is a widely-used standard for object detection and instance segmentation datasets.

## Overview

COCO format provides a standardized JSON structure for dataset annotations, making it compatible with numerous computer vision frameworks and tools including PyTorch, TensorFlow, Detectron2, and MMDetection.

## Scripts

### yolo_to_coco_converter.py

**Purpose**: Convert YOLO segmentation datasets to COCO format

**Key Features**:
- **YOLO Segmentation Support**: Handles YOLO format polygon annotations with normalized coordinates
- **Split Preservation**: Maintains train/val/test splits or merges into single dataset
- **Coordinate Conversion**: Converts normalized YOLO coordinates to absolute pixel coordinates
- **COCO JSON Generation**: Creates standardized annotations.json with:
  - Image metadata (dimensions, filenames, IDs)
  - Category definitions from YOLO class names
  - Instance annotations with bounding boxes and polygon segmentations
  - Area calculations for each instance
- **Image Copying**: Automatically copies images to output directories

**Requirements**:
```bash
pip install pillow pyyaml
```

**Usage**:
```bash
# Convert with existing splits (train/val/test folders)
python3 yolo_to_coco_converter.py --input-dir /path/to/yolo/dataset --output-dir ./coco-output

# Merge all splits into single folder
python3 yolo_to_coco_converter.py --input-dir /path/to/yolo/dataset --output-dir ./coco-merged --merge
```

**Input Structure**:
```
dataset/
├── dataset.yaml           # Contains class names (or classes.txt)
├── labels/
│   ├── train/            # Training labels (.txt files)
│   ├── val/              # Validation labels (.txt files)
│   └── test/             # Test labels (.txt files, optional)
└── images/
    ├── train/            # Training images
    ├── val/              # Validation images
    └── test/             # Test images (optional)
```

**Output Structure** (Without --merge):
```
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
```

**Output Structure** (With --merge):
```
output-dir/
├── annotations.json
└── [all images]
```

**YOLO Annotation Format**:
```
class_id x1 y1 x2 y2 ... xn yn  # Normalized coordinates (0-1)
```

**COCO Annotations Format**:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "class_name",
      "supercategory": "object"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ]
}
```

**Features**:
- **Automatic Class Loading**: Reads class names from classes.txt or data.yaml files
- **Polygon Conversion**: Converts YOLO polygon coordinates to COCO segmentation format
- **Bounding Box Calculation**: Automatically computes bounding boxes from segmentation polygons
- **Area Calculation**: Approximates area using bounding box dimensions
- **Multi-format Support**: Handles both split-based and merged dataset structures

### tfrecord_to_coco.py

**Purpose**: Convert TensorFlow TFRecord files to COCO dataset format

**Key Features**:
- **TFRecord Parsing**: Extracts images and instance segmentation masks from TFRecord files
- **COCO JSON Generation**: Creates standardized annotations.json file with:
  - Image metadata (dimensions, filenames, IDs)
  - Category definitions with unique identifiers
  - Instance annotations with bounding boxes and polygon segmentations
  - Area calculations for each instance
- **Mask Visualization**: Generates colored composite instance masks using matplotlib's tab20 colormap
- **Contour Extraction**: Uses OpenCV to extract polygon contours from binary masks
- **Progress Tracking**: Displays conversion progress with tqdm progress bar

**Requirements**:
```bash
pip install tensorflow opencv-python pillow numpy matplotlib tqdm
```

**Usage**:
```bash
# Basic conversion
python3 tfrecord_to_coco.py <tfrecord_path> <output_root>

# Example
python3 tfrecord_to_coco.py data/train.tfrecord output/coco_dataset
```

**Input Format**:
The TFRecord file must contain the following features:
- `image/encoded`: Encoded image data (JPEG/PNG bytes)
- `image/filename`: Original filename
- `image/height`: Image height in pixels
- `image/width`: Image width in pixels
- `image/object/mask`: Instance segmentation masks (encoded as images)
- `image/object/mask/class/text`: Class names for each instance
- `image/object/mask/class/label`: Numeric class labels

**Output Structure**:
```
output_root/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── InstanceMasks/
│   ├── image1.png          # Colored composite mask
│   ├── image2.png
│   └── ...
└── annotations.json        # COCO format annotations
```

**COCO Annotations Format**:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080,
      "coco_url": "output_root/Images/image1.jpg"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "defect_type",
      "supercategory": "defect"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ]
}
```

**Features**:
- **Automatic Category Registration**: Dynamically discovers and registers categories from the dataset
- **Polygon Simplification**: Extracts simplified polygon contours from masks using OpenCV
- **Multi-instance Support**: Handles multiple instances per image with distinct colors
- **Statistics Reporting**: Reports the image with the most annotations after conversion

## Use Cases

- Converting TensorFlow datasets for use with PyTorch-based training pipelines
- Preparing datasets for frameworks that require COCO format (e.g., Detectron2)
- Creating standardized dataset exports for sharing and collaboration
- Enabling compatibility with COCO evaluation tools and metrics
- Generating visualization-ready instance masks

## Support and Resources

- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **COCO API**: [https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- **pycocotools**: [https://pypi.org/project/pycocotools/](https://pypi.org/project/pycocotools/)
