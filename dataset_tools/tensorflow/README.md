# TensorFlow Dataset Tools

This directory contains tools for converting datasets to TensorFlow TFRecord format and debugging TFRecord files for instance segmentation tasks with TensorFlow Object Detection API.

## Overview

[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) is TensorFlow's binary file format for storing large datasets efficiently. These tools help convert Labelbox annotations to TFRecord format and provide visualization utilities for debugging and verification.

## Scripts

### 1. ndjson_to_tfrecord.py

**Purpose**: Convert Labelbox NDJSON exports to TensorFlow TFRecord format

**Key Features**:
- Downloads images from Labelbox URLs
- Downloads segmentation masks via Labelbox API
- Automatically computes bounding boxes from masks
- Supports ImageBoundingBox and ImageSegmentationMask
- Normalizes coordinates to [0-1] range
- Configurable class name to ID mapping
- Progress tracking with emoji indicators
- Handles missing annotations gracefully

**Requirements**:
```bash
pip install tensorflow requests ndjson pyyaml opencv-python numpy
```

**Usage**:
```bash
# Basic conversion with default settings
python3 ndjson_to_tfrecord.py \\
    --config config.yaml \\
    --ndjson labelbox_export.ndjson \\
    --image_folder images \\
    --tfrecord output.tfrecord

# With specific paths
python3 ndjson_to_tfrecord.py \\
    --config /path/to/config.yaml \\
    --ndjson /path/to/export.ndjson \\
    --image_folder /path/to/images \\
    --tfrecord /path/to/output.tfrecord
```

**Config File** (config.yaml):
```yaml
api_key: "YOUR_LABELBOX_API_KEY_HERE"
```

**Class Mapping Configuration**:
Edit the `class_indices` dictionary in the script:
```python
class_indices = {
    "defect": 1,
    "normal": 2,
    "scratch": 3,
    "dent": 4
}
```

**TFRecord Schema**:
The generated TFRecord contains the following features:

| Feature Path | Type | Description |
|--------------|------|-------------|
| `image/height` | int64 | Image height in pixels |
| `image/width` | int64 | Image width in pixels |
| `image/filename` | bytes | Original filename |
| `image/source_id` | bytes | Source identifier (filename) |
| `image/encoded` | bytes | JPEG encoded image data |
| `image/format` | bytes | Image format ('jpeg') |
| `image/object/bbox/xmin` | float list | Normalized bbox x-min (0-1) |
| `image/object/bbox/xmax` | float list | Normalized bbox x-max (0-1) |
| `image/object/bbox/ymin` | float list | Normalized bbox y-min (0-1) |
| `image/object/bbox/ymax` | float list | Normalized bbox y-max (0-1) |
| `image/object/bbox/class/text` | bytes list | Class names |
| `image/object/bbox/class/label` | int64 list | Class IDs |
| `image/object/mask` | bytes list | PNG encoded binary masks |
| `image/object/mask/class/text` | bytes list | Mask class names |
| `image/object/mask/class/label` | int64 list | Mask class IDs |

**Output**:
```
images/
├── image001.jpg
├── image002.jpg
└── ...

output.tfrecord    # Binary TFRecord file
```

**Progress Output Example**:
```
[1/150] ✅ Image already downloaded: image001.jpg
[2/150] ⬇️ Downloaded reference image: image002.jpg
[3/150] ✅ Image already downloaded: image003.jpg
...
TFRecord saved to: output.tfrecord
```

---

### 2. debug_tfrecord.py

**Purpose**: Visualize and debug TFRecord files interactively

**Key Features**:
- Parses TFRecord with complete schema
- Displays reference images with bbox overlays
- Shows instance masks in grayscale subplots
- Interactive matplotlib navigation
- Configurable record limit
- Prints detailed metadata
- ESC key to navigate between records

**Requirements**:
```bash
pip install tensorflow matplotlib pillow numpy
```

**Usage**:
```bash
# Visualize all records
python3 debug_tfrecord.py dataset.tfrecord

# Visualize first 5 records only
python3 debug_tfrecord.py dataset.tfrecord --max-records 5

# Visualize specific TFRecord
python3 debug_tfrecord.py /path/to/file.tfrecord --max-records 10
```

**Controls**:
- **ESC**: Close current visualization and move to next record
- Window close button also advances to next record

---

## Dependencies

### For Conversion
```bash
pip install tensorflow>=2.4.0
pip install requests
pip install ndjson
pip install pyyaml
pip install opencv-python
pip install numpy
```

### For Debugging
```bash
pip install tensorflow>=2.4.0
pip install matplotlib
pip install pillow
pip install numpy
```

### Optional (for GPU acceleration)
```bash
# If you have CUDA-capable GPU
pip install tensorflow-gpu>=2.4.0
```

---

## Support and Resources

- **TensorFlow**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **TFRecord Guide**: [https://www.tensorflow.org/tutorials/load_data/tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- **Labelbox**: [https://labelbox.com/](https://labelbox.com/)
- **Mask R-CNN**: [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
