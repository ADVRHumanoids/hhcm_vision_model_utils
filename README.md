# HHCM Vision Model Workbench

A comprehensive toolkit for computer vision model development, specializing in object detection and instance segmentation. This workbench provides end-to-end workflows from dataset preparation through model training, with support for multiple frameworks including YOLO and Mask RCNN.

## Overview

This repository contains production-ready tools for:
- **Dataset Preparation**: Format conversion, augmentation, and quality control
- **Model Training**: State-of-the-art detection and segmentation models
- **Hyperparameter Optimization**: Automated tuning with Ray Tune and Optuna
- **Visualization**: Interactive tools for dataset exploration and debugging

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd hhcm-vision-model-workbench

# Install core dependencies
pip install torch torchvision opencv-python numpy pyyaml matplotlib pillow

# Install framework-specific dependencies (choose based on your needs)
pip install ultralytics ray[tune] optuna tensorboard  # For YOLO
pip install detectron2 optuna  # For Mask R-CNN
pip install labelme2yolo shapely ndjson requests  # For dataset tools
```

### Basic Workflow

```bash
# 1. Prepare dataset (Labelbox → YOLO)
cd dataset_tools/labelme/
python ndjson_to_labelme.py --config config.yaml --ndjson export.ndjson --image-folder dataset/

pip install labelme2yolo
labelme2yolo --json_dir dataset/ --val_size 0.2 --output_format polygon --segmentation polygon

# 2. Train YOLO segmentation model
cd ../../training/segmentation/yolo/
python train_yolo11_seg.py --data path/to/data.yaml --model yolo11m-seg.pt --epochs 100
```

## Repository Structure

```
hhcm-vision-model-workbench/
├── training/              # Model training pipelines
│   ├── detection/         # Object detection models
│   │   └── multimodel/    # Multi-model comparison (Faster R-CNN, RetinaNet, SSD)
│   └── segmentation/      # Instance segmentation models
│       ├── yolo/          # YOLO11-seg training with Ray Tune
│       └── mask_rcnn/     # Mask R-CNN with Detectron2
├── dataset_tools/         # Dataset preparation and conversion
│   ├── yolo/              # YOLO format tools (9 scripts)
│   ├── labelme/           # LabelMe format tools (5 scripts)
│   ├── tensorflow/        # TFRecord tools (2 scripts)
│   └── preprocessing/     # Image enhancement (1 script)
├── utils/                 # Utility scripts
│   ├── clearCuda.py       # GPU memory management
│   └── modelInfo.py       # Model inspection
├── notebooks/             # Jupyter notebooks
│   ├── coco_viewer.ipynb                      # COCO dataset visualization
│   ├── ExtractCOCOExportYOLOSegmentation.ipynb  # COCO subset extraction
│   └── fiftyOneGetDataset.ipynb                 # FiftyOne dataset downloader
└── legacy/                # Deprecated code (YOLOv5)
```

## Features

### Training Frameworks

#### YOLO Segmentation
- **Framework**: Ultralytics YOLO11-seg
- **Speed**: 30-60 FPS inference
- **Best For**: Production deployment, real-time applications
- **Features**: Multi-GPU, AMP, Ray Tune optimization
- [Documentation](training/segmentation/yolo/)

#### Mask R-CNN
- **Framework**: Detectron2
- **Accuracy**: State-of-the-art precision
- **Best For**: Research, complex scenes
- **Features**: Optuna tuning, top-K checkpointing
- [Documentation](training/segmentation/mask_rcnn/)

#### Multi-Model Detection
- **Frameworks**: Faster R-CNN, RetinaNet, SSD
- **Purpose**: Architecture comparison and benchmarking
- [Documentation](training/detection/multimodel/)

### Dataset Tools

#### Format Conversion
| From | To | Tool |
|------|-----|------|
| Labelbox NDJSON | LabelMe | `dataset_tools/labelme/ndjson_to_labelme.py` |
| LabelMe | YOLO | `labelme2yolo` (pip package) |
| COCO | YOLO | `dataset_tools/yolo/extract_coco_export_yolo.py` |
| Labelbox NDJSON | TFRecord | `dataset_tools/tensorflow/ndjson_to_tfrecord.py` |

#### Quality Control Tools
- **Interactive Annotation Filtering**: `dataset_tools/yolo/polygon_filter_gui.py`
- **Dataset Visualization**: `dataset_tools/yolo/display_dataset_gui.py`
- **Statistics Analysis**: `dataset_tools/yolo/stats_yolo_dataset.py`
- **Class Balancing**: `dataset_tools/yolo/balance_dataset.py`

#### Augmentation
- **Tiling with Annotation Preservation**: `dataset_tools/labelme/tiling_augmentation.py`
- **Image Enhancement**: `dataset_tools/preprocessing/bw_converter.py`



## Dependencies

### Core Requirements
```bash
pip install torch torchvision
pip install opencv-python numpy pyyaml
pip install matplotlib pillow
```

### Framework-Specific

**YOLO**:
```bash
pip install ultralytics>=8.0.0
pip install ray[tune] optuna
pip install tensorboard
```

**Detectron2**:
```bash
pip install detectron2
pip install optuna
```

**Dataset Tools**:
```bash
pip install labelme2yolo  # LabelMe → YOLO conversion
pip install shapely ndjson requests  # LabelMe tools
pip install tensorflow pycocotools  # TFRecord tools
pip install fiftyone  # Dataset downloading
```

## Documentation

Each directory contains comprehensive README.md files:

- [Training Documentation](training/) - Model training guides
  - [YOLO Segmentation](training/segmentation/yolo/)
  - [Mask R-CNN](training/segmentation/mask_rcnn/)
  - [Multi-Model Detection](training/detection/multimodel/)
- [Dataset Tools](dataset_tools/) - Conversion and preparation
  - [YOLO Tools](dataset_tools/yolo/)
  - [LabelMe Tools](dataset_tools/labelme/)
  - [TensorFlow Tools](dataset_tools/tensorflow/)
  - [Preprocessing](dataset_tools/preprocessing/)
- [Utilities](utils/) - Helper scripts
- [Notebooks](notebooks/) - Interactive examples
- [Legacy](legacy/) - Deprecated code (YOLOv5)

## Resources

- **Ultralytics YOLO**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Detectron2**: [https://detectron2.readthedocs.io/](https://detectron2.readthedocs.io/)
- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **Labelbox**: [https://labelbox.com/](https://labelbox.com/)
- **LabelMe**: [https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)
- **labelme2yolo**: [https://pypi.org/project/labelme2yolo/](https://pypi.org/project/labelme2yolo/)
