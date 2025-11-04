#!/usr/bin/env python3
"""
Manual YOLO Dataset Filter - Interactive Review Tool.

Interactive command-line tool to review and filter YOLO dataset images.
Displays each image with its annotations overlaid as colored transparent masks
and allows user to keep or delete using simple keyboard controls.

Author: Alessio Lovato

Arguments:
    --dataset: Path to YOLO dataset.yaml file
    --splits: Which splits to review (default: all)

Controls:
    Spacebar: Keep image and move to next
    Backspace: Delete image and move to next
    Q: Quit
"""

import os
import sys
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_dataset_yaml(yaml_path: str) -> dict:
    """Load YOLO dataset configuration"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def read_yolo_annotations(label_path: str) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Read YOLO polygon format annotations.
    Returns list of (class_id, [(x1, y1), (x2, y2), ...]) in normalized coords
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Parse polygon points
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((coords[i], coords[i + 1]))
            
            if len(points) >= 3:  # Valid polygon needs at least 3 points
                annotations.append((cls_id, points))
    
    return annotations


def draw_annotations(image: np.ndarray, 
                     annotations: List[Tuple[int, List[Tuple[float, float]]]], 
                     class_names: Dict[int, str],
                     colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """
    Draw YOLO polygon annotations on image.
    Annotations are in normalized coordinates (0-1).
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    for cls_id, points in annotations:
        # Convert normalized coords to pixel coords
        pixel_points = []
        for x, y in points:
            pixel_points.append((int(x * w), int(y * h)))
        
        if len(pixel_points) < 3:
            continue
        
        # Get class info
        class_name = class_names.get(cls_id, f'class_{cls_id}')
        color = colors.get(cls_id, (0, 255, 0))
        
        # Draw filled polygon with transparency
        overlay = result.copy()
        pts = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        # Draw polygon outline
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
        
        # Draw label with background
        centroid_x = int(np.mean([p[0] for p in pixel_points]))
        centroid_y = int(np.mean([p[1] for p in pixel_points]))
        
        label = f'{class_name}'
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(result, 
                     (centroid_x - 5, centroid_y - label_h - 5),
                     (centroid_x + label_w + 5, centroid_y + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(result, label, (centroid_x, centroid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    return result


def generate_class_colors(class_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """Generate distinct colors for each class using golden ratio"""
    import colorsys
    
    colors = {}
    golden_ratio = 0.618033988749895
    h = 0.0
    
    for cls_id in sorted(class_names.keys()):
        h = (h + golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(h, 0.8, 0.95)
        # Convert to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors[cls_id] = bgr
    
    return colors


def find_image_files(dataset_root: str, splits: List[str]) -> List[Tuple[str, str, str]]:
    """
    Find all images with their label paths.
    Returns list of (image_path, label_path, split_name)
    """
    image_files = []
    
    for split in splits:
        images_dir = os.path.join(dataset_root, 'images', split)
        labels_dir = os.path.join(dataset_root, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"⚠️  Warning: Images directory not found: {images_dir}")
            continue
        
        if not os.path.exists(labels_dir):
            print(f"⚠️  Warning: Labels directory not found: {labels_dir}")
            continue
        
        # Find all images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in Path(images_dir).glob(ext):
                base_name = img_path.stem
                label_path = os.path.join(labels_dir, base_name + '.txt')
                
                # Only include if label file exists
                if os.path.exists(label_path):
                    image_files.append((str(img_path), label_path, split))
    
    return sorted(image_files)


def delete_image_and_label(image_path: str, label_path: str):
    """Delete both image and label files"""
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        return True
    except Exception as e:
        print(f"❌ Error deleting files: {e}")
        return False


def create_display_image(image: np.ndarray, 
                         annotations: List[Tuple[int, List[Tuple[float, float]]]],
                         class_names: Dict[int, str],
                         colors: Dict[int, Tuple[int, int, int]],
                         current_idx: int,
                         total: int,
                         split: str,
                         filename: str,
                         kept_count: int,
                         deleted_count: int) -> np.ndarray:
    """Create display image with annotations and info overlay"""
    
    # Draw annotations on image
    display = draw_annotations(image, annotations, class_names, colors)
    
    # Add info panel at top
    header_height = 120
    h, w = display.shape[:2]
    
    # Create header
    header = np.zeros((header_height, w, 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    
    # Add text info
    y_offset = 30
    line_height = 25
    
    # Title
    title = f"YOLO Dataset Manual Filter - [{current_idx + 1}/{total}]"
    cv2.putText(header, title, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += line_height
    
    # File info
    info = f"Split: {split}  |  File: {filename}  |  Annotations: {len(annotations)}"
    cv2.putText(header, info, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y_offset += line_height
    
    # Statistics
    stats = f"Kept: {kept_count}  |  Deleted: {deleted_count}  |  Remaining: {total - current_idx - 1}"
    cv2.putText(header, stats, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)
    y_offset += line_height
    
    # Controls
    controls = "SPACE: Keep & Next  |  BACKSPACE: Delete & Next  |  Q/ESC: Quit"
    cv2.putText(header, controls, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1, cv2.LINE_AA)
    
    # Combine header and image
    result = np.vstack([header, display])
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Interactive YOLO dataset filter - Review and delete unwanted images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  SPACE       : Keep current image and move to next
  BACKSPACE   : Delete current image and annotation, move to next
  Q / ESC     : Quit and save progress

Examples:
  # Review all splits
  python3 manual_filter.py --dataset ./dataset/dataset.yaml
  
  # Review only training set
  python3 manual_filter.py --dataset ./dataset/dataset.yaml --splits train
  
  # Review validation and test sets
  python3 manual_filter.py --dataset ./dataset/dataset.yaml --splits val test

Notes:
  - Deleted files are permanently removed (not moved to trash)
  - Progress is saved automatically as you go
  - You can quit anytime and resume later
        """
    )
    
    parser.add_argument('--dataset', required=True,
                       help='Path to YOLO dataset.yaml file')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='Which splits to review (default: all)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Load dataset configuration
    try:
        config = load_dataset_yaml(args.dataset)
        dataset_root = Path(args.dataset).parent
        
        # Get class names
        class_names = config.get('names', {})
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        
        print(f"📝 Loaded dataset: {config.get('path', 'Unknown')}")
        print(f"📂 Root directory: {dataset_root}")
        print(f"🏷️  Classes ({len(class_names)}): {', '.join(class_names.values())}")
        
    except Exception as e:
        print(f"❌ Error loading dataset configuration: {e}")
        sys.exit(1)
    
    # Generate colors for classes
    colors = generate_class_colors(class_names)
    
    # Find all images
    print(f"\n🔍 Searching for images in splits: {', '.join(args.splits)}")
    image_files = find_image_files(str(dataset_root), args.splits)
    
    if not image_files:
        print(f"❌ Error: No images found in specified splits")
        sys.exit(1)
    
    print(f"📸 Found {len(image_files)} images to review\n")
    
    # Statistics
    kept_count = 0
    deleted_count = 0
    current_idx = 0
    
    # Create window
    window_name = "YOLO Dataset Filter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    
    print("🎮 Controls:")
    print("   SPACE       : Keep current image and move to next")
    print("   BACKSPACE   : Delete current image and annotation, move to next")
    print("   Q / ESC     : Quit")
    print()
    
    # Review loop
    while current_idx < len(image_files):
        img_path, label_path, split = image_files[current_idx]
        filename = Path(img_path).name
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Warning: Could not load image: {img_path}")
            current_idx += 1
            continue
        
        # Load annotations
        annotations = read_yolo_annotations(label_path)
        
        # Create display
        display = create_display_image(
            image, annotations, class_names, colors,
            current_idx, len(image_files), split, filename,
            kept_count, deleted_count
        )
        
        # Show image
        cv2.imshow(window_name, display)
        
        # Wait for keypress
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # Quit
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                print(f"\n👋 Quitting...")
                print(f"📊 Final Statistics:")
                print(f"   Kept: {kept_count}")
                print(f"   Deleted: {deleted_count}")
                print(f"   Reviewed: {current_idx}/{len(image_files)}")
                cv2.destroyAllWindows()
                sys.exit(0)
            
            # Keep (Space)
            elif key == 32:  # Space
                kept_count += 1
                print(f"✓ [{current_idx + 1}/{len(image_files)}] Kept: {filename}")
                current_idx += 1
                break
            
            # Delete (Backspace)
            elif key == 8 or key == 127:  # Backspace or Delete
                print(f"🗑️  [{current_idx + 1}/{len(image_files)}] Deleting: {filename}")
                if delete_image_and_label(img_path, label_path):
                    deleted_count += 1
                    print(f"   ✓ Deleted successfully")
                else:
                    print(f"   ✗ Failed to delete")
                current_idx += 1
                break
    
    # Finished reviewing all images
    print(f"\n🎉 Review Complete!")
    print(f"📊 Final Statistics:")
    print(f"   Total reviewed: {len(image_files)}")
    print(f"   Kept: {kept_count}")
    print(f"   Deleted: {deleted_count}")
    print(f"   Remaining: {kept_count}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
