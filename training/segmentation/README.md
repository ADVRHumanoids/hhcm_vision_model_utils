# Instance Segmentation Training

This directory contains training pipelines for instance segmentation models. Instance segmentation combines object detection (finding objects) with semantic segmentation (pixel-level classification), producing both bounding boxes and precise pixel masks for each detected object.

## Overview

Instance segmentation is essential for applications requiring precise object localization and shape understanding. These tools support training state-of-the-art models with hyperparameter optimization, comprehensive logging, and production-ready configurations.

## Available Frameworks

### [YOLO Segmentation](yolo/)

Modern, fast instance segmentation using Ultralytics YOLO11-seg models.


---

### [Mask R-CNN](mask_rcnn/)

Precise instance segmentation using Mask R-CNN with Detectron2 backend.


---

## Quick Comparison

| Feature | YOLO11-seg | Mask R-CNN |
|---------|-----------|------------|
| **Speed** | Fast (30-60 FPS) | Moderate (5-15 FPS) |
| **Accuracy** | High | Very High |
| **Framework** | Ultralytics | Detectron2 |
| **Ease of Use** | Easier | More Complex |
| **Deployment** | Excellent (ONNX, TensorRT) | Good |
| **GPU Memory** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Best For** | Production/Real-time | Research/Precision |

## Choosing the Right Framework

**Choose YOLO11-seg when**:
- Speed is critical (real-time or near real-time)
- Deploying to edge devices or production systems
- Working with clear, well-defined object boundaries
- Need quick experimentation and iteration
- Limited GPU resources

**Choose Mask R-CNN when**:
- Maximum accuracy is priority
- Working with complex, overlapping objects
- Research or academic project
- Fine-grained segmentation required
- Sufficient computational resources available

## Dataset Preparation:
See [../../dataset_tools/](../../dataset_tools/) for conversion and augmentation tools.

## Dependencies

### YOLO Dependencies
```bash
pip install ultralytics>=8.0.0
pip install ray[tune]
pip install optuna
pip install tensorboard
```

### Mask R-CNN Dependencies
```bash
pip install detectron2
pip install torch torchvision
pip install optuna
pip install opencv-python
pip install pyyaml
```

## Support and Resources

### YOLO Resources
- **Ultralytics Docs**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **YOLO GitHub**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Segmentation Guide**: [https://docs.ultralytics.com/tasks/segment/](https://docs.ultralytics.com/tasks/segment/)
- **LabelMe to YOLO Conversion**: [https://pypi.org/project/labelme2yolo/](https://pypi.org/project/labelme2yolo/)

### Mask R-CNN Resources
- **Detectron2 Docs**: [https://detectron2.readthedocs.io/](https://detectron2.readthedocs.io/)
- **Detectron2 GitHub**: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- **Model Zoo**: [https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

### General Resources
- **COCO Evaluation**: [https://cocodataset.org/#detection-eval](https://cocodataset.org/#detection-eval)
