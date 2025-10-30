# /bin/bash/env python3
"""
@brief Script to visualize YOLOv11-seg ground truth masks with random sampling.
Shows ground truth masks (colored, semi-transparent) in a separate window,
while predictions are displayed automatically with result.show().
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import yaml
import random
from glob import glob
from ultralytics import YOLO

def load_yolo_segmentation(label_path, img_shape):
    """
    Load YOLO segmentation annotations (txt format).
    Returns a list of (class_id, polygon_points).
    """
    h, w = img_shape[:2]
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            coords = np.array(list(map(float, parts[1:])))
            # Denormalize
            coords[0::2] *= w
            coords[1::2] *= h
            polygon = coords.reshape(-1, 2).astype(int)
            polygons.append((cls, polygon))
    return polygons

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def visualize_gt_pred(image_path, label_path, class_names, result):
    """
    Show ground truth segmentation masks (semi-transparent + labels)
    alongside YOLO predictions (from Ultralytics result).
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    polygons = load_yolo_segmentation(label_path, img.shape)

    # Create subplot with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: Ground Truth ---
    axes[0].imshow(img)
    for cls, poly in polygons:
        rng = np.random.default_rng(cls)
        color = rng.uniform(0, 1, 3)

        # Fill mask
        axes[0].fill(poly[:, 0], poly[:, 1], color=color, alpha=0.4)

        # Contour
        poly_closed = np.vstack([poly, poly[0]])
        axes[0].plot(poly_closed[:, 0], poly_closed[:, 1], color=color, linewidth=2)

        # Label text (first vertex)
        class_name = class_names.get(cls, str(cls))
        x_text, y_text = poly[0]
        axes[0].text(
            x_text,
            y_text,
            class_name,
            color="white",
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2")
        )

    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # --- Right: Prediction ---
    pred_img = result.plot()  # numpy BGR image
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    axes[1].imshow(pred_img)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    # Press ESC to close window
    def on_key(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the trained YOLO model file (e.g., best.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to YOLO data YAML file')
    parser.add_argument('--images', type=int, default=1, help='Number of random images to select')
    parser.add_argument('--threshold', type=float, default=0.4, help='Confidence threshold for predictions')
    parser.add_argument('--random', action='store_true', default=False, help='Randomize image order (default: sequential)')
    args = parser.parse_args()

    # Resolve paths
    model_path = os.path.abspath(os.path.expanduser(args.model))
    data_path = os.path.abspath(os.path.expanduser(args.data))
    threshold = args.threshold

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' does not exist.")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1:
        print("Error: Threshold must be between 0 and 1.")
        sys.exit(1)
    
    if args.images < 1:
        print("Error: Number of images must be at least 1.")
        sys.exit(1)

    # Load model
    model = YOLO(model_path)

    # Load YAML
    with open(data_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    # Prefer test → val
    img_dir = data_yaml.get('test') or data_yaml.get('val')
    if not img_dir:
        print("Error: No 'test' or 'val' dataset found in YAML.")
        sys.exit(1)

    base_path = data_yaml.get('path', '')
    if base_path and not os.path.isabs(img_dir):
        img_dir = os.path.join(base_path, img_dir)

    img_dir = os.path.abspath(os.path.expanduser(img_dir))
    image_files = []
    for ext in ('*.jpg', '*.png', '*.jpeg'):
        image_files.extend(glob(os.path.join(img_dir, ext)))

    if len(image_files) == 0:
        print(f"Error: No images found in directory '{img_dir}'.")
        sys.exit(1)

    # Random selection
    num_images = min(args.images, len(image_files))
    random_images = random.sample(image_files, num_images) if args.random else image_files[:num_images]

    # Annotation paths
    annotation_files = []
    for img_path in random_images:
        ann_path = img_path.replace('/images/', '/labels/')
        ann_path = os.path.splitext(ann_path)[0] + '.txt'
        annotation_files.append(ann_path)

    # Class names
    names = data_yaml['names']
    if isinstance(names, dict):
        class_names = {int(k): v for k, v in names.items()}
    else:
        class_names = {i: name for i, name in enumerate(names)}
    print(f"Class names: {class_names}")

    # Run model and show predictions
    results = model(random_images, imgsz=480, conf=threshold, device='0')

    for i, result in enumerate(results):
        print(f"\nResults for image {i+1}/{num_images}: {random_images[i]}")
        visualize_gt_pred(random_images[i], annotation_files[i], class_names, result)
