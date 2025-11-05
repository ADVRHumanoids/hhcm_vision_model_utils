#!/usr/bin/env python3
"""
Convert YOLO segmentation format labels to detection format (bounding boxes).

Reads YOLO polygon segmentation annotations and converts them to bounding box format
by computing the min/max extent of polygon points. Useful for creating detection
datasets from segmentation annotations or training detection models on segmentation data.

Author: tori

Arguments:
    input_folder: Folder with YOLO segmentation .txt labels
    output_folder: Folder to save YOLO detection .txt labels

Output Format:
    Detection: class_id center_x center_y width height (normalized 0-1)
"""

import os

def convert_seg_line_to_bbox(line):
    """
    Convert a YOLO segmentation line to bounding box format.

    Args:
        line (str): YOLO segmentation format line (class_id x1 y1 x2 y2 ...)

    Returns:
        str: YOLO detection format (class_id center_x center_y width height)
    """
    values = list(map(float, line.strip().split()))
    class_id = int(values[0])
    points = list(zip(values[1::2], values[2::2]))  # (x, y) pairs

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_center = (max(xs) + min(xs)) / 2
    y_center = (max(ys) + min(ys)) / 2
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def convert_folder(input_folder, output_folder):
    """
    Convert all segmentation label files in a folder to detection format.

    Args:
        input_folder (str): Path to folder containing segmentation labels
        output_folder (str): Path where detection labels will be saved
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        try:
                            bbox_line = convert_seg_line_to_bbox(line)
                            outfile.write(bbox_line + '\n')
                        except Exception as e:
                            print(f"Error processing line in {filename}: {e}")

    print(f"Conversion complete. Output saved to: {output_folder}")

# === Usage ===
if __name__ == "__main__":
    input_folder = "/home/tori/YOLO/data/seg_laser_nicla_320x320/valid/labels"     # Folder with YOLO segmentation .txt labels
    output_folder = "/home/tori/YOLO/data/det_laser_nicla_320x320/valid/labels"    # Folder to save YOLO detection .txt labels

    convert_folder(input_folder, output_folder)
