# YOLO Dataset Tools

This directory contains a comprehensive suite of tools for creating, managing, and curating YOLO format datasets for object detection and instance segmentation tasks.

## Overview

YOLO (You Only Look Once) format is a popular standard for training object detection and segmentation models. These tools help with dataset creation, conversion, quality control, and maintenance throughout the model development lifecycle.

## Scripts

### 1. extract_coco_export_yolo.py

**Purpose**: Extract subsets from COCO dataset and convert to YOLO format

**Key Features**:
- **Category-based Sampling**: Guarantee minimum instances per class
- **Dual Format Support**: Export as detection (bbox) or segmentation (polygons)
- **Parallel Processing**: Multi-threaded image downloading with progress bars
- **Smart Split Management**: Automatically fills validation set from training when needed
- **YAML Generation**: Creates YOLO-compatible dataset configuration

**Usage**:
```python
# Edit configuration variables in script:
my_categories = {"person": 0, "car": 1, ...}  # Category mapping
N_sample_train = 500                           # Min instances per class
N_sample_valid = 110
model_type = "seg"                            # "det" or "seg"

python3 extract_coco_export_yolo.py
```

**Output Structure**:
```
output_dir/
├── train/
│   ├── images/       # Training images
│   └── labels/       # YOLO format labels
├── valid/
│   ├── images/       # Validation images
│   └── labels/       # YOLO format labels
└── data.yaml         # Dataset configuration
```

---

### 2. convert_seg_to_det.py

**Purpose**: Convert YOLO segmentation labels to detection format (bounding boxes)

**Key Features**:
- Computes tight bounding boxes from polygon points
- Preserves class IDs and file structure
- Batch processing of entire folders

**Usage**:
```bash
python3 convert_seg_to_det.py

# Edit paths in script:
input_folder = "/path/to/seg/labels"
output_folder = "/path/to/det/labels"
```

**Format Conversion**:
```
# Segmentation format:
class_id x1 y1 x2 y2 x3 y3 ... xn yn

# Detection format:
class_id center_x center_y width height
```

---

### 3. rename_class_ids.py

**Purpose**: Batch rename class IDs across multiple dataset folders

**Key Features**:
- Multi-folder batch processing
- Preserves label file structure
- Useful for dataset consolidation

**Usage**:
```python
# Edit configuration in script:
label_dirs = [
    '/path/to/dataset1/train/labels',
    '/path/to/dataset1/valid/labels',
    ...
]
old_class_id = '0'
new_class_id = 19

python3 rename_class_ids.py
```

---

### 4. analyze_balance.py

**Purpose**: Analyze class distribution and balance across dataset splits

**Key Features**:
- **Visual Distribution**: ASCII bar charts showing class balance
- **Split Comparison**: Compare train/val/test distributions side-by-side
- **Imbalance Detection**: Automatic warnings for severe imbalance (>3:1 ratio)
- **Detailed Statistics**: Files, instances, and per-class breakdowns
- **Cross-split Analysis**: View distribution of each class across all splits

**Arguments**:
```
Positional:
  dataset_root          Path to dataset root directory (default: current directory)
                        Must contain dataset.yaml and labels/ folder
```

**Usage Examples**:
```bash
# Analyze current directory (looks for dataset.yaml)
python3 analyze_balance.py

# Analyze specific dataset
python3 analyze_balance.py /path/to/dataset

# Analyze and save output to file
python3 analyze_balance.py > balance_report.txt
```

**Output Example**:
```
YOLO Dataset Balance Analysis
================================================================================
📋 Dataset: /path/to/dataset
📝 Classes defined: 2
   - Class 0: defect
   - Class 1: normal

────────────────────────────────────────────────────────────────────────────────
📁 TRAIN SET
────────────────────────────────────────────────────────────────────────────────
   Label files: 850
   Total instances: 1908

   Instance distribution:
   Class              │ Distribution                                       │ Count
   ───────────────────┼────────────────────────────────────────────────────┼──────────────
   0: defect          │ ████████████████████████████████████████           │  1250 (65.5%)
   1: normal          │ █████████████████████                              │   658 (34.5%)

   ⚖️  Imbalance ratio: 1.90:1 (max/min)
   ℹ️  Moderate class imbalance

SPLIT COMPARISON
================================================================================
Split      │    Files │  Instances │   Avg/File
───────────┼──────────┼────────────┼───────────
train      │      850 │       1908 │       2.25
val        │      212 │        476 │       2.25
test       │      106 │        238 │       2.25

Class              │    Train │      Val │     Test │    Total
───────────────────┼──────────┼──────────┼──────────┼─────────
0: defect          │     1250 │      312 │      156 │     1718
1: normal          │      658 │      164 │       82 │      904
```

**Interpreting Results**:
- **Imbalance Ratio**: Ratio of most common to least common class
  - ≤1.5:1 = Well balanced ✓
  - 1.5-3.0:1 = Moderate imbalance ℹ️
  - >3.0:1 = Severe imbalance ⚠️
- **Avg/File**: Average annotations per image (helpful for detecting empty files)

---

### 5. balance_dataset.py

**Purpose**: Balance datasets by intelligently removing overrepresented class instances

**Key Features**:
- **Smart File Selection**: Prioritizes files with highest overrepresentation
- **Target Ratio Control**: Adjustable balance target (default 2.0:1)
- **Preview Mode**: Shows proposed changes with detailed statistics before applying
- **Safe Backup**: Moves removed files to backup folder (never deletes permanently)
- **Cross-class Awareness**: Handles multi-class annotations intelligently
- **Greedy Algorithm**: Efficiently selects files to remove for optimal balance

**Arguments**:
```
Positional:
  dataset_root          Path to dataset root directory (default: current directory)
                        Must contain dataset.yaml with class names

Options:
  --ratio FLOAT         Target imbalance ratio (max/min classes)
                        Lower = more balanced but more files removed
                        Higher = less balanced but fewer files removed
                        Default: 2.0
                        Recommended range: 1.5-3.0
```

**Usage Examples**:
```bash
# Use default 2.0:1 ratio in current directory
python3 balance_dataset.py

# Custom ratio and path
python3 balance_dataset.py /path/to/dataset --ratio 1.5

# More aggressive balancing (close to 1:1)
python3 balance_dataset.py --ratio 1.2
```

---

### 6. polygon_filter_gui.py

**Purpose**: Interactive GUI tool for removing polygons with too few points

This tool provides a visual interface to review and remove such polygons 
while allowing manual override for cases where automatic filtering is too 
aggressive.

**Key Features**:
- **Interactive Visualization**: Click polygons to toggle deletion state
- **Multi Istance Togglr**: Hold right-click mouse button and drag to toggle multiple items
- **Automatic Marking**: Auto-marks polygons below minimum point threshold
- **Color-Coded Display**:
  - Green polygons: Will be kept (≥ min points)
  - Red polygons: Marked for deletion (< min points)
  - Yellow outline: Currently selected polygon
- **Undo Support**: Undo last 10 actions with full state restoration
- **Keyboard Navigation**: Efficient review workflow
- **Multi-split Support**: Processes train/val/test splits automatically
- **Progress Tracking**: Shows current position and completion percentage
- **Statistics Panel**: Real-time count of kept vs deleted annotations

**Arguments**:
```
Required:
  --dataset PATH        Path to YOLO dataset.yaml file
                        Must contain 'names' and dataset paths

Optional:
  --min-points INT      Minimum points threshold for polygons
                        Polygons with fewer points are auto-marked for deletion
                        Default: 15
                        Recommended: 15-20 for segmentation quality
                        Lower values: Keep more polygons but may include low-quality
                        Higher values: Remove more polygons but may lose valid ones
```

**Keyboard Controls**:
| Key | Action | Description |
|-----|--------|-------------|
| **Left Click** | Toggle deletion | Click on polygon to toggle keep/delete state |
| **Right Click** | Multiple selection | Hold and move the cursor to create a toggling area |
| **Right Arrow** | Next image | Save current changes and move to next image |
| **Enter** | Next image | Alternative to Right Arrow |
| **Left Arrow** | Previous image | Go back to previous image (changes saved) |
| **Spacebar** | Keep all | Unmark all annotations on current image |
| **Z** | Undo | Undo last 10 actions |
| **S** | Save | Manually save current image annotations |
| **Q** | Quit | Exit program (prompts to save if unsaved changes) |
| **ESC** | Quit | Alternative to Q |


**Usage Examples**:
```bash
# Default threshold of 15 points
python3 polygon_filter_gui.py --dataset dataset.yaml

# Stricter filtering (20 points minimum)
python3 polygon_filter_gui.py --dataset dataset.yaml --min-points 20

# More lenient (10 points minimum)
python3 polygon_filter_gui.py --dataset dataset.yaml --min-points 10

# Process specific dataset
python3 polygon_filter_gui.py --dataset /path/to/dataset/dataset.yaml --min-points 15
```

---

### 7. manual_filter.py

**Purpose**: Interactive command-line tool for manual dataset review and filtering

This tool provides a streamlined OpenCV-based interface for quick manual review of the annotated images in the dataset, allowing to keep or delete them.

**Key Features**:
- **Visual Review**: Full-screen display of images with annotations overlaid
- **Simple Controls**: Just 3 keys - spacebar (keep), backspace (delete), Q (quit)
- **Segmentation Visualization**:
  - Filled transparent polygons (30% opacity)
  - Colored outlines (unique color per class)
  - Class labels at polygon centroids
- **Progress Tracking**: Shows "Image X/Total" in window title
- **Auto-cleanup**: Removes both image and corresponding label file
- **OpenCV-based**: Fast rendering, works over SSH with X11 forwarding

**Arguments**:
```
Required:
  --dataset PATH        Path to YOLO dataset.yaml file

Optional:
  --splits [SPLIT ...]  Which dataset splits to review
                        Options: train, val, test
                        Default: all available splits
                        Examples:
                          --splits train        (only train)
                          --splits train val    (train and val)
                          --splits test         (only test)
```

**Keyboard Controls**:
| Key | Action | Result |
|-----|--------|--------|
| **Spacebar** | Keep image | Move to next image, file unchanged |
| **Backspace** | Delete image | Permanently delete image + label, move to next |
| **Q** | Quit | Exit review process immediately |

**Usage Examples**:
```bash
# Review all splits in current directory
python3 manual_filter.py --dataset dataset.yaml

# Review only training set
python3 manual_filter.py --dataset dataset.yaml --splits train

# Review train and validation sets
python3 manual_filter.py --dataset dataset.yaml --splits train val

# Review specific dataset
python3 manual_filter.py --dataset /path/to/dataset/dataset.yaml --splits train
```

**When to Use This Tool**:
- **Quality Control**: Remove blurry, dark, or corrupted images
- **Annotation Check**: Remove images with wrong/missing annotations
- **Outlier Removal**: Remove images that don't fit your use case
- **Post-augmentation**: Review generated/augmented images
- **Final Cleanup**: Last pass before training

**Safety Notes**:
- ⚠️ **DESTRUCTIVE OPERATION**: Backspace permanently deletes files
- ⚠️ **NO UNDO**: Deleted files cannot be recovered
- ✅ **Recommendation**: Backup dataset before running
- ✅ **Alternative**: Copy dataset to temp folder for review

---

### 8. label_editor_gui.py

**Purpose**: Interactive GUI for fixing YOLO segmentation label classes

Sometimes annotations have the wrong class assigned (mislabeled during initial annotation),
or you need to merge/rename classes. This GUI tool allows quick visual correction of
class labels without re-annotating the entire polygon.

**Key Features**:
- **Class Cycling**: Left-click polygon to cycle through all available classes
- **Quick Deletion**: Right-click to remove annotation entirely
- **Visual Feedback**:
  - Color-coded classes (up to 10 distinct colors)
  - Class legend panel showing all classes
  - Highlighted polygon when clicked
- **Point-in-Polygon Detection**: Accurate click detection using ray-casting algorithm
- **Save Management**: Manual save or auto-save on next/previous
- **Keyboard Navigation**: Rapid movement through dataset
- **Multi-split Support**: Works across train/val/test splits

**Arguments**:
```
Required:
  --dataset PATH        Path to YOLO dataset.yaml file
                        Must contain class names under 'names' key
```

**Mouse Controls**:
| Action | Click | Effect |
|--------|-------|--------|
| **Change Class** | Left Click on polygon | Cycles class: 0→1→2→...→0 |
| **Delete Annotation** | Right Click on polygon | Removes polygon entirely |

**Keyboard Controls**:
| Key | Action | Description |
|-----|--------|-------------|
| **Enter** | Next image | Save changes (if any) and advance |
| **N** | Next image | Alternative to Enter |
| **Backspace** | Previous image | Save and go back one image |
| **P** | Previous image | Alternative to Backspace |
| **S** | Manual save | Save current annotations without advancing |
| **Q** | Quit | Exit program (prompts for unsaved changes) |

**Class Colors** (automatically assigned):
```
Class 0: Red         Class 5: Cyan
Class 1: Green       Class 6: Purple
Class 2: Blue        Class 7: Orange
Class 3: Yellow      Class 8: Light Blue
Class 4: Magenta     Class 9: Lime
```

**Usage Examples**:
```bash
# Basic usage - current directory
python3 label_editor_gui.py --dataset dataset.yaml

# Specific dataset path
python3 label_editor_gui.py --dataset /path/to/dataset/dataset.yaml

# Over SSH with X11 forwarding
ssh -X user@host
python3 label_editor_gui.py --dataset dataset.yaml
```

---

### 9. upload_to_labelbox.py

**Purpose**: Upload YOLO segmentation dataset to Labelbox for annotation/review

**Key Features**:
- **Format Conversion**: YOLO polygons → Labelbox mask annotations
- **Batch Upload**: Processes entire datasets with progress tracking
- **Project Organization**: Creates/uses Labelbox projects and datasets
- **Ontology Mapping**: Maps YOLO classes to Labelbox ontology

**Requirements**:
```bash
pip install labelbox pillow
```

**Usage**:
```bash
python3 upload_to_labelbox.py \
    --dataset dataset.yaml \
    --api-key YOUR_LABELBOX_API_KEY \
    --project-name "Dataset Review" \
    --split train
```

---

## Requirements

### Core Dependencies
```bash
pip install numpy opencv-python pillow pyyaml tqdm
```

### Additional (Feature-specific)
```bash
# COCO extraction
pip install pycocotools matplotlib requests

# GUI tools
pip install tkinter  # Usually pre-installed with Python

# Labelbox upload
pip install labelbox
```

## Support and Resources

- **YOLO Documentation**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **Labelbox**: [https://labelbox.com/docs/](https://labelbox.com/docs/)
- **LabelMe to YOLO Conversion**: [https://pypi.org/project/labelme2yolo/](https://pypi.org/project/labelme2yolo/)
