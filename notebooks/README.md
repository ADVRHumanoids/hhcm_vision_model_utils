# Jupyter Notebooks

This directory contains interactive Jupyter notebooks for dataset exploration, visualization, and conversion workflows.

## Notebooks

### coco_viewer.ipynb

Interactive COCO dataset viewer with comprehensive visualization capabilities.

**Purpose**: View and analyze COCO-format datasets with rich annotations including bounding boxes and segmentation masks.

**Key Features**:
- Load COCO annotation files using pycocotools
- Visualize individual images or random samples
- Display bounding boxes and segmentation polygons with color coding
- Show category labels on annotations
- Interactive exploration with customizable image selection

**Requirements**:
```bash
pip install pycocotools matplotlib opencv-python numpy
```

**Configuration**:
```python
dataset_dir = "path/to/coco/dataset"
coco_annotation_path = f"{dataset_dir}/annotations.json"
image_dir = f"{dataset_dir}/images/"

# View specific image
select_specific_image = True
specific_image_id = 178

# Or view random samples
select_specific_image = False  # Loads 10 random images
```

**Output**:
- Matplotlib figures showing images with overlaid annotations
- Color-coded bounding boxes and segmentation masks
- Category labels above each annotation

---

### ExtractCOCOExportYOLOSegmentation.ipynb

Extract custom subsets from COCO dataset and convert to YOLO segmentation format.

**Purpose**: Create domain-specific YOLO segmentation datasets from COCO with precise control over classes and sample sizes.

**Pipeline**:
1. **Configuration**: Define target classes and sample sizes
2. **Sampling**: Extract images with specified classes from COCO
3. **Balancing**: Ensure no train/val overlap, handle class imbalance
4. **Conversion**: Convert bbox and polygon annotations to YOLO format
5. **Download**: Fetch images with progress tracking
6. **Package**: Create ZIP archive with complete YOLO dataset

**Requirements**:
```bash
pip install pycocotools matplotlib numpy pyyaml requests progress
```

**Configuration Example**:
```python
# Define target classes (COCO class name → YOLO class ID)
my_categories = {
    "person": 0,
    "sports ball": 1,
    "bottle": 2,
    "cup": 3,
}

# Sampling parameters
N_sample_train = 400  # Min samples per class for training
N_sample_valid = 100  # Min samples per class for validation

output_name = "./coco_subset"  # Output directory base name
```

---

### fiftyOneGetDataset.ipynb

Download COCO dataset subsets using FiftyOne Zoo.

**Purpose**: Quick access to COCO datasets with simple filtering using FiftyOne's high-level API.

**Key Features**:
- Direct access to COCO via FiftyOne Zoo
- Filter by specific classes
- Select train, val, or test splits
- Limit sample count for fast prototyping
- Automatic download and caching
- Native FiftyOne dataset format

**Requirements**:
```bash
pip install fiftyone
```

**Usage Example**:
```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",                              # train, val, or test
    label_types=["detections", "segmentations"],
    classes=["bear", "dog", "cat"],             # Target classes
    only_matching=True,                          # Only matching images
    max_samples=10,                              # Limit samples
)
```

**FiftyOne Dataset Location**:
Downloads are cached in `~/fiftyone/coco-2017/` by default.

**Additional Capabilities**:
```python
# Download with custom directory
fo.utils.coco.download_coco_dataset_split(
    dataset_dir="/path/to/save",
    split="train",
    classes=["stop sign"],
    max_samples=5
)
```
--- 
## Support and Resources

- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **pycocotools**: [https://github.com/cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)
- **FiftyOne**: [https://docs.voxel51.com/](https://docs.voxel51.com/)
- **YOLO Format**: [https://docs.ultralytics.com/datasets/segment/](https://docs.ultralytics.com/datasets/segment/)
- **Jupyter**: [https://jupyter.org/](https://jupyter.org/)
