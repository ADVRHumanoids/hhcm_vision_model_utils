# COCO Format Conversion Tools

This directory contains utilities for converting datasets to COCO (Common Objects in Context) format, which is a widely-used standard for object detection and instance segmentation datasets.

## Overview

COCO format provides a standardized JSON structure for dataset annotations, making it compatible with numerous computer vision frameworks and tools including PyTorch, TensorFlow, Detectron2, and MMDetection.

## Scripts

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
python tfrecord_to_coco.py <tfrecord_path> <output_root>

# Example
python tfrecord_to_coco.py data/train.tfrecord output/coco_dataset
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
