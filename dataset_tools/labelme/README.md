# LabelMe Dataset Tools

This directory contains tools for working with LabelMe format datasets, including conversion from Labelbox, tiling for augmentation, visualization, and quality control utilities.

## Overview

[LabelMe](http://labelme.csail.mit.edu/) is a popular annotation tool for computer vision that stores annotations as JSON files alongside images. These tools help manage, convert, augment, and validate LabelMe datasets throughout the dataset preparation pipeline.

## Scripts

### 1. ndjson_to_labelme.py

**Purpose**: Convert Labelbox NDJSON exports to LabelMe format

**Key Features**:
- Downloads images from Labelbox URLs
- Downloads segmentation masks via Labelbox API
- Converts binary masks to polygon contours using OpenCV
- Preserves RGB colors from Labelbox
- Generates LabelMe v3.21.1 compatible JSON
- Optional mask PNG retention
- Auto-cleanup on failures

**Requirements**:
```bash
pip install ndjson requests pyyaml opencv-python numpy
```

**Usage**:
```bash
# Basic conversion with config file
python3 ndjson_to_labelme.py \\
    --config config.yaml \\
    --ndjson export.ndjson \\
    --image-folder output/labelme_dataset

# Save mask PNG files for inspection
python3 ndjson_to_labelme.py \\
    --config config.yaml \\
    --ndjson export.ndjson \\
    --image-folder output/labelme_dataset \\
    --save-masks
```

**Config File Format** (config.yaml):
```yaml
api_key: "YOUR_LABELBOX_API_KEY_HERE"
```

**Output Structure**:
```
output/labelme_dataset/
├── image001.jpg
├── image001.json      # LabelMe annotations
├── image002.jpg
├── image002.json
└── ...
```

---

### 2. tiling_augmentation.py

**Purpose**: Split large images into overlapping tiles with annotation preservation

**Key Features**:
- Configurable tile size and overlap
- Polygon clipping using Shapely library
- Automatic polygon validation and repair
- Handles complex geometries (MultiPolygon, GeometryCollection)
- Optional border padding
- Optional zoom-out scaled versions
- Creates tiling log for reconstruction
- Preserves colors and labels

**Requirements**:
```bash
pip install shapely opencv-python numpy colorama
```

**Usage**:
```bash
# Basic tiling (640x640, no overlap)
python3 tiling_augmentation.py \\
    --input-dir labelme_dataset \\
    --output-dir tiled_dataset \\
    --width 640 \\
    --height 640

# With 20% overlap
python3 tiling_augmentation.py \\
    --input-dir labelme_dataset \\
    --output-dir tiled_dataset \\
    --width 640 \\
    --height 640 \\
    --overlap 0.2

# With border padding instead of overlap
python3 tiling_augmentation.py \\
    --input-dir labelme_dataset \\
    --output-dir tiled_dataset \\
    --width 640 \\
    --height 640 \\
    --pad-border

# Include zoomed-out full images
python3 tiling_augmentation.py \\
    --input-dir labelme_dataset \\
    --output-dir tiled_dataset \\
    --width 640 \\
    --height 640 \\
    --overlap 0.1 \\
    --zoom-out
```

**Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | required | Path to input LabelMe dataset |
| `--output-dir` | str | required | Path to output tiled dataset |
| `--width` | int | 640 | Tile width in pixels |
| `--height` | int | 640 | Tile height in pixels |
| `--overlap` | float | 0.0 | Overlap ratio (0.0-1.0) |
| `--zoom-out` | flag | False | Create scaled full-image versions |
| `--pad-border` | flag | False | Pad border tiles instead of overlapping |

**Output**:
```
tiled_dataset/
├── image001_0.jpg
├── image001_0.json
├── image001_1.jpg
├── image001_1.json
├── ...
├── image001_scaled_full.jpg    # If --zoom-out used
├── image001_scaled_full.json
└── tiling_log.txt              # Reconstruction metadata
```

**Tiling Log Format**:
```
image001: grid 3x3, tile 640x640, orig 1920x1080
image002: grid 2x2, tile 640x640, orig 1280x720
```

---

### 3. tile_grid_viewer.py

**Purpose**: Interactive viewer for quality control of tiled datasets

**Key Features**:
- Reconstructs original image grid from tiles
- Overlays tile numbers for identification
- Displays annotations with transparency
- Interactive tile selection
- Delete problematic tiles
- Logs all deletions
- Maximized window for better viewing

**Usage**:
```bash
python3 tile_grid_viewer.py \\
    --tile-folder tiled_dataset \\
    --tiling-log tiled_dataset/tiling_log.txt
```

**Controls**:
| Key/Action | Function |
|------------|----------|
| **Left Click** | Toggle tile selection (highlighted in red) |
| **D** | Delete selected tiles and JSON files |
| **N** / **→** | Next base image |
| **P** / **←** | Previous base image |
| **Q** / **ESC** | Quit viewer |

**Workflow**:
1. Viewer reconstructs grid from tiles using tiling log
2. Each tile shows its number and annotations
3. Click tiles to mark for deletion (turns red)
4. Press 'D' to delete marked tiles
5. Navigate through base images with N/P keys
6. Deletions logged to `deleted_files.log`

---

### 4. display_dataset.py

**Purpose**: Interactive viewer for LabelMe annotated datasets

**Key Features**:
- Auto-resizes images to fit screen
- Transparent filled polygons (50% alpha)
- Colored borders from LabelMe JSON
- Supports polygon and rectangle annotations
- Keyboard navigation
- Displays annotation labels
- Processes subset or all images

**Usage**:
```bash
# View all images in folder
python3 display_dataset.py --folder labelme_dataset

# View first 50 images
python3 display_dataset.py --folder labelme_dataset --images 50

# View specific dataset
python3 display_dataset.py --folder /path/to/dataset --images 100
```

**Controls**:
| Key | Action |
|-----|--------|
| **← Arrow** / **Q** | Previous image |
| **→ Arrow** / **E** | Next image |
| **ESC** | Exit viewer |
| **Any other key** | Next image |

**Display Features**:
- Window title shows: `[5/100] image_name.jpg`
- Auto-detection of screen size
- Image scaling maintains aspect ratio
- Annotations use colors from JSON `line_color` and `fill_color` fields
- Labels displayed at polygon/rectangle corners

---

### 5. check_missing_pairs.py

**Purpose**: Validate dataset integrity by checking for missing pairs

**Key Features**:
- Scans for .jpg/.jpeg and .json files
- Reports images without annotations
- Reports annotations without images
- Case-insensitive matching
- Clear, organized output

**Usage**:
```bash
# Check folder
python3 check_missing_pairs.py --folder labelme_dataset

# Check multiple folders
python3 check_missing_pairs.py --folder dataset/train
python3 check_missing_pairs.py --folder dataset/val
```

**Output Example**:
```
Checked folder: labelme_dataset
JPEG files missing JSON:
  image045.jpg
  image078.jpg
JSON files missing JPEG:
  temp_annotation.json
```

---

## LabelMe JSON Format

### Structure
```json
{
  "version": "3.21.1",
  "flags": {},
  "shapes": [
    {
      "label": "defect",
      "line_color": [255, 0, 0],
      "fill_color": [255, 0, 0],
      "points": [
        [100.5, 200.3],
        [150.2, 200.1],
        [150.8, 250.9],
        [100.1, 250.5]
      ],
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "image001.jpg",
  "imageData": "",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

### Shape Types
- **polygon**: List of [x, y] points (minimum 3 points)
- **rectangle**: Two points [top-left, bottom-right]

### Colors
- Format: RGB list `[R, G, B]` where each value is 0-255
- Used for visualization in display tools

---

## Dependencies

### Core
```bash
pip install opencv-python numpy
```

### For Labelbox Conversion
```bash
pip install ndjson requests pyyaml
```

### For Tiling
```bash
pip install shapely colorama
```

### For Display (usually pre-installed)
```bash
pip install tkinter
```

---

## Support and Resources

- **LabelMe**: [http://labelme.csail.mit.edu/](http://labelme.csail.mit.edu/)
- **LabelMe GitHub**: [https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)
- **Labelbox**: [https://labelbox.com/](https://labelbox.com/)
- **Shapely Docs**: [https://shapely.readthedocs.io/](https://shapely.readthedocs.io/)
