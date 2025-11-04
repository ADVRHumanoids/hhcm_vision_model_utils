# Model Training

This directory contains training pipelines for object detection and instance segmentation models.

## Structure

### [Detection](detection/)

Object detection models that predict bounding boxes and class labels.

**Available Models**:
- Multi-model comparison framework (Faster R-CNN, RetinaNet, SSD)
- Performance benchmarking across architectures
- Automatic evaluation and metric logging

**Best For**:
- When only bounding boxes are needed
- Comparing multiple detection architectures
- Applications where segmentation masks are unnecessary

See [detection/README.md](detection/) for details.

---

### [Segmentation](segmentation/)

Instance segmentation models that predict both bounding boxes and pixel-precise masks.

**Available Frameworks**:
- **[YOLO11-seg](segmentation/yolo/)**: Fast, production-ready segmentation
- **[Mask R-CNN](segmentation/mask_rcnn/)**: High-accuracy research-grade segmentation

**Best For**:
- Precise object localization and shape understanding
- Applications requiring pixel-level masks
- Scenarios with overlapping or occluded objects

See [segmentation/README.md](segmentation/) for framework comparison and details.

---

## Quick Comparison

| Task | Output | Speed | Use Case |
|------|--------|-------|----------|
| **Detection** | Bounding boxes + labels | Fast | Object localization, counting, tracking |
| **Segmentation** | Boxes + pixel masks | Moderate | Shape analysis, precise localization, robotics |

## Choosing Detection vs Segmentation

**Choose Detection when**:
- Only need to know "where" objects are (boxes)
- Speed is critical
- Boxes are sufficient for your task (counting, tracking, etc.)
- Training data has only bounding box annotations

**Choose Segmentation when**:
- Need precise object boundaries and shapes
- Working with overlapping objects
- Require pixel-level understanding
- Have polygon/mask annotations in dataset


## Dataset Requirements

All training scripts require properly formatted datasets:

**YOLO Format** (detection and segmentation):
```
YOLODataset/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**COCO Format** (Mask R-CNN automatically converts from YOLO):
- Detection: Boxes only
- Segmentation: Boxes + polygon coordinates

See [../dataset_tools/](../dataset_tools/) for conversion tools.

>**NOTE:**
Before implementing your own format conversion package, check if there is an available pip package (for example labelme2yolo, ecc.)

## Dependencies

```bash
# For detection
pip install torch torchvision

# For YOLO segmentation
pip install ultralytics ray[tune] optuna

# For Mask R-CNN
pip install detectron2 optuna
```

## Support and Resources

- **Ultralytics**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Detectron2**: [https://detectron2.readthedocs.io/](https://detectron2.readthedocs.io/)
- **COCO**: [https://cocodataset.org/](https://cocodataset.org/)
