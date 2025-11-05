#!/usr/bin/env python3
"""
Interactive viewer for LabelMe annotated datasets.

Displays images with annotations from LabelMe JSON files in an interactive OpenCV window.
Supports polygon and rectangle annotations with transparency, colored overlays based on
LabelMe's line_color and fill_color fields, and keyboard navigation for reviewing datasets.

Arguments:
    --folder: Path to folder containing LabelMe JSON files and images
    --images: Number of images to display (default: 0 = all)

Supported Image Formats:
    .jpg, .jpeg, .png

Controls:
    ← / Q: Previous image
    → / E: Next image
    ESC: Exit viewer
    Any other key: Next image

Author: Alessio Lovato
"""

import os
import json
import cv2
import numpy as np
import argparse
import tkinter as tk

# Add colorama for colored terminal output
from colorama import Fore

# === CONFIG ===
SUPPORTED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

def get_screen_size():
    """Get screen resolution for window fitting."""
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        # Use more of the screen for better visibility
        return int(screen_width * 0.95), int(screen_height * 0.9)
    except:
        # Fallback to larger resolution if tkinter fails
        return 1600, 1000

def resize_image_to_fit_screen(image, max_width, max_height):
    """Resize image to fit within screen dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    
    return image, 1.0

def draw_labelme_annotations(image, data, scale=1.0):
    """Draw annotations on image with optional scaling."""
    # Create an overlay for transparency
    overlay = image.copy()
    
    for shape in data.get('shapes', []):
        points = np.array(shape['points'], dtype=np.float32)
        # Scale points if image was resized
        if scale != 1.0:
            points = points * scale
        points = points.astype(np.int32)
        
        label = shape.get('label', '')
        
        # Get colors from shape or use defaults
        line_color = shape.get('line_color', None)
        fill_color = shape.get('fill_color', None)
        
        # Convert colors from various formats to BGR
        if line_color and line_color != "null":
            if isinstance(line_color, list) and len(line_color) == 3:
                # RGB to BGR
                border_color = (int(line_color[2]), int(line_color[1]), int(line_color[0]))
            else:
                border_color = (0, 255, 0)  # Default green
        else:
            border_color = (0, 255, 0)  # Default green
            
        if fill_color and fill_color != "null":
            if isinstance(fill_color, list) and len(fill_color) == 3:
                # RGB to BGR
                fill_bgr = (int(fill_color[2]), int(fill_color[1]), int(fill_color[0]))
            else:
                fill_bgr = (0, 255, 0)  # Default green
        else:
            fill_bgr = border_color  # Use same as border

        if shape['shape_type'] == 'polygon':
            # Fill the polygon on overlay
            if len(points) > 2:
                cv2.fillPoly(overlay, [points], fill_bgr)
            
            # Draw border on original image
            cv2.polylines(image, [points], isClosed=True, color=border_color, thickness=2)
            
            # Put label near first point
            if len(points) > 0:
                cv2.putText(image, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        elif shape['shape_type'] == 'rectangle':
            if len(points) >= 2:
                pt1, pt2 = tuple(points[0]), tuple(points[1])
                
                # Fill rectangle on overlay
                cv2.rectangle(overlay, pt1, pt2, fill_bgr, -1)
                
                # Draw border on original image
                cv2.rectangle(image, pt1, pt2, border_color, 2)
                cv2.putText(image, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend overlay with original image for transparency (50% alpha)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image

# === MAIN LOOP ===
arg_parser = argparse.ArgumentParser(description="Display LabelMe annotated images.")
arg_parser.add_argument('--folder', type=str, required=True, help='Path to LabelMe dataset folder')
arg_parser.add_argument('--images', type=str, default='0', help='Number of images to display (default: 0 = all)')
args = arg_parser.parse_args()

# Sanity checks
if not os.path.exists(args.folder):
    print(Fore.RED + f"Folder does not exist: {args.folder}" + Fore.RESET)
    exit(1)

if not args.images.isdigit() or int(args.images) < 0:
    print(Fore.RED + "--images must be a positive integer or 0 for all." + Fore.RESET)
    exit(1)

json_files = [f for f in os.listdir(args.folder) if f.endswith('.json')]
print(f"Found {len(json_files)} annotated files.")

if len(json_files) == 0:
    print(Fore.YELLOW + "No JSON files found in the specified folder." + Fore.RESET)
    exit(0)

# Get screen dimensions for window fitting
max_width, max_height = get_screen_size()
print(f"Screen size detected: {max_width}x{max_height}")
print()
print("Navigation:")
print("  ← Left Arrow  : Previous image")
print("  → Right Arrow : Next image")
print("  Q key         : Previous image (alternative)")
print("  E key         : Next image (alternative)")
print("  ESC           : Exit")
print("  Any other key : Next image")
print()

# Limit number of images if specified
total_images = len(json_files)
if args.images.isdigit() and int(args.images) > 0:
    total_images = min(total_images, int(args.images))
    json_files = json_files[:total_images]

current_idx = 0
window_title = "LabelMe Dataset Viewer"

# Create the window with proper flags for resizing and keyboard control
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_title, max_width, max_height)

while current_idx < total_images:
    json_file = json_files[current_idx]
    json_path = os.path.join(args.folder, json_file)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(Fore.RED + f"Failed to load JSON {json_file}: {e}" + Fore.RESET)
        current_idx += 1
        continue

    image_path = os.path.join(args.folder, data['imagePath'])
    if not os.path.exists(image_path):
        print(Fore.RED + f"Image not found for {json_file}: {data['imagePath']}" + Fore.RESET)
        current_idx += 1
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(Fore.RED + f"Failed to load image: {image_path}" + Fore.RESET)
        current_idx += 1
        continue

    # Resize image to fit screen if necessary
    original_size = f"{image.shape[1]}x{image.shape[0]}"
    resized_image, scale = resize_image_to_fit_screen(image, max_width, max_height)
    
    if scale < 1.0:
        print(f"Resizing {data['imagePath']} from {original_size} to {resized_image.shape[1]}x{resized_image.shape[0]} (scale: {scale:.2f})")
    
    # Draw annotations with proper scaling
    annotated = draw_labelme_annotations(resized_image.copy(), data, scale)
    
    # Update window title with navigation info
    display_title = f"[{current_idx + 1}/{total_images}] {data['imagePath']}"
    cv2.setWindowTitle(window_title, display_title)
    
    # Update the image content in the same window
    cv2.imshow(window_title, annotated)
    
    # Ensure the window has focus for keyboard input
    cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 0)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    
    # Handle navigation
    if key == 27:  # ESC key
        print("Exiting...")
        break
    elif key in [81, 2, 113]:  # Left arrow (Linux: 81, some systems: 2, q key: 113)
        if current_idx == 0:
            current_idx = total_images - 1
        else:
            current_idx = max(0, current_idx - 1)
    elif key in [83, 3, 101]:  # Right arrow (Linux: 83, some systems: 3, e key: 101)
        if current_idx == total_images - 1:
            current_idx = 0
        else:
            current_idx = min(total_images - 1, current_idx + 1)
    else:
        # Any other key moves to next image
        current_idx = min(total_images - 1, current_idx + 1)

cv2.destroyAllWindows()
print("Visualization completed.")
