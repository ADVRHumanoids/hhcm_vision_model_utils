# Dataset Tools

This directory contains tools for dataset preparation, format conversion, augmentation, and visualization. These tools form a comprehensive pipeline for preparing vision datasets from annotation to training-ready formats.

## Tools by Category

### Format Conversion

**[YOLO Tools](yolo/)** - YOLO format utilities
- LabelMe → YOLO conversion
- COCO → YOLO conversion
- Dataset statistics and balancing
- Interactive annotation filtering and cleanup
- Dataset splitting (train/val/test)

**[LabelMe Tools](labelme/)** - LabelMe format utilities
- Labelbox → LabelMe conversion
- Tiling augmentation with annotation preservation
- Interactive dataset visualization
- Polygon-based annotation tools

**[COCO Tools](coco/)** - COCO format utilities
- YOLO → COCO conversion
- TFRecord → COCO conversion

**[TensorFlow Tools](tensorflow/)** - TFRecord format utilities
- Labelbox → TFRecord conversion
- TFRecord visualization and debugging

### Preprocessing

**[Preprocessing Tools](preprocessing/)** - Image enhancement
- Grayscale conversion
- Automatic contrast/brightness adjustment

---

## Quick Format Guide

| Format | Structure | Tools |
|--------|-----------|-------|
| **YOLO** | `.txt` normalized coordinates | [yolo/](yolo/) |
| **LabelMe** | `.json` polygon annotations | [labelme/](labelme/) |
| **COCO** | Single `.json` with images | [coco/](coco/) |
| **TFRecord** | Binary TensorFlow format | [tensorflow/](tensorflow/) |

## Common Workflows

### 1. Labelbox → YOLO Pipeline

```bash
# Step 1: Labelbox → LabelMe
cd labelme/
python3 ndjson_to_labelme.py \
    --config config.yaml \
    --ndjson export.ndjson \
    --image-folder dataset/

# Step 2: LabelMe → YOLO (using labelme2yolo package)
pip install labelme2yolo
labelme2yolo \
    --json_dir ../labelme/dataset/ \
    --output_format polygon \
    --segmentation polygon
    --val_size 0.2

# Step 3: Balance and split (if needed)
cd yolo/
python3 balance_dataset.py \
    --yolo-dir ../YOLODataset/ \
    --train-ratio 0.7 \
    --val-ratio 0.2
```

## Format Conversion Matrix

| From | To | Tool | Command |
|------|-----|------|---------|
| Labelbox NDJSON | LabelMe | `labelme/ndjson_to_labelme.py` | [docs](labelme/) |
| LabelMe | YOLO | `labelme2yolo` (pip package) | `pip install labelme2yolo` |
| COCO | YOLO Det | `yolo/coco_to_yolo.py` | [docs](yolo/) |
| YOLO | COCO | `coco/yolo_to_coco_converter.py` | [docs](coco/) |
| TFRecord | COCO | `coco/tfrecord_to_coco.py` | [docs](coco/) |
| Labelbox NDJSON | TFRecord | `tensorflow/ndjson_to_tfrecord.py` | [docs](tensorflow/) |


## Dependencies

### Core Dependencies
```bash
pip install opencv-python3 numpy pyyaml
```

### YOLO Tools
```bash
pip install ultralytics
pip install labelme2yolo  # For LabelMe to YOLO conversion
```

### LabelMe Tools
```bash
pip install shapely ndjson requests
```

### TensorFlow Tools
```bash
pip install tensorflow pycocotools
```

### Preprocessing Tools
```bash
pip install opencv-python3 numpy
```

## Support and Resources

- **YOLO Format**: [https://docs.ultralytics.com/datasets/](https://docs.ultralytics.com/datasets/)
- **LabelMe**: [https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)
- **Labelbox**: [https://labelbox.com/](https://labelbox.com/)
- **TFRecord**: [https://www.tensorflow.org/tutorials/load_data/tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- **COCO Format**: [https://cocodataset.org/#format-data](https://cocodataset.org/#format-data)
