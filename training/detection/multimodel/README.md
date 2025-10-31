# Object Detection Models Training

This folder contains training scripts and utilities for object detection models on COCO format datasets. Supports multiple architectures including Faster R-CNN, RetinaNet, FCOS, and SSD variants with automatic dataset splitting and comprehensive evaluation metrics.

## Scripts

### train_detection_models.py

Training script for multiple object detection architectures with unified interface.

**Arguments**:
- `data_path` (str, required) - Path to dataset directory containing 'images/' and 'annotations/instances_default.json'
- `data_name` (str, optional, default: "laserSpots") - Dataset identifier for output file naming
- `batch_size` (int, optional, default: 1) - Training batch size
- `num_epochs` (int, optional, default: 1) - Number of training epochs
- `model_type` (str, optional, default: "faster_rcnn_v1") - Model architecture. Options: 'faster_rcnn_v1', 'faster_rcnn_v2', 'fasterrcnn_mobilenet_high', 'fasterrcnn_mobilenet_low'
- `val_percentage` (float, optional, default: 0.20) - Fraction of data for validation (0.0-1.0)
- `test_percentage` (float, optional, default: 0.10) - Fraction of data for testing (0.0-1.0)

**Example**:
```python
from train_detection_models import run

run(
    data_path="/path/to/dataset",
    data_name="my_dataset",
    batch_size=4,
    num_epochs=50,
    model_type="faster_rcnn_v1",
    val_percentage=0.20,
    test_percentage=0.10
)
```

**Output**: Saves trained model as `{model_type}_e{epochs}_b{batch}_tvt{train}{val}{test}_{data_name}.pt` and training plot as corresponding `.png` file.

### CustomCocoDataset.py

Custom PyTorch Dataset class for loading COCO format annotations. Used internally by training script.

**Key Methods**:
- `__init__(root, annotation, transforms=None)` - Initialize dataset with images and annotations
- `__getitem__(index)` - Get image and annotations by index (converts COCO bbox format to PyTorch format)
- `__len__()` - Returns number of images in dataset

**Example**:
```python
from CustomCocoDataset import CustomCocoDataset
import torchvision

dataset = CustomCocoDataset(
    root='/path/to/images',
    annotation='/path/to/instances_default.json',
    transforms=torchvision.transforms.ToTensor()
)
```

### TestModel.py

Evaluation utilities for testing trained detection models with mAP metrics.

**Key Function**:
- `test_simple(data_loader_test, model, device, show_images=False)` - Evaluate model on test set

**Arguments**:
- `data_loader_test` (DataLoader) - PyTorch DataLoader with test data
- `model` (nn.Module) - Trained detection model
- `device` (torch.device) - Device for evaluation (cuda or cpu)
- `show_images` (bool, optional, default: False) - Display first 4 predictions with bounding boxes

**Returns**: Dictionary with mAP metrics (map, map_50, map_75, map_small, map_medium, map_large)

**Example**:
```python
from TestModel import test_simple
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('trained_model.pt')

results = test_simple(test_loader, model, device, show_images=True)
print(f"mAP: {results['map']:.3f}")
```

## Dependencies

- torch
- torchvision
- pycocotools
- torchmetrics
- numpy
- matplotlib
- opencv-python (cv2)
- PIL (Pillow)

## Supported Model Architectures

1. **Faster R-CNN ResNet50 FPN** (v1, v2) - Two-stage detector with Feature Pyramid Network
2. **Faster R-CNN MobileNetV3** (high, low) - Mobile-optimized detector for deployment
3. **FCOS ResNet50 FPN** - Fully Convolutional One-Stage detector (anchor-free)
4. **RetinaNet ResNet50 FPN** (v1, v2) - Single-stage detector with focal loss
5. **SSD300 VGG16** - Single Shot MultiBox Detector
6. **SSDLite320 MobileNetV3** - Lightweight SSD variant for mobile

Note: Some model variants (FCOS, RetinaNet, SSD) are commented out in the code and may require additional configuration.

## Dataset Format

Expected COCO dataset structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    └── instances_default.json
```

The annotation JSON must follow COCO format with 'images', 'annotations', and 'categories' fields.

## Training Workflow

1. **Data Loading**: CustomCocoDataset loads images and COCO annotations
2. **Train/Val/Test Split**: Automatic random splitting based on specified percentages
3. **Data Augmentation**: Applied to training set (flips, sharpness, autocontrast)
4. **Training**: SGD optimizer with step LR scheduler over specified epochs
5. **Evaluation**: Validation after each epoch, test evaluation at end
6. **Output**: Saves model checkpoint and generates loss/mAP plots
