#!/usr/bin/env python3
"""
Rename YOLO class IDs across multiple dataset folders.

This script batch-updates YOLO label files by replacing old class IDs with new ones.
Useful for consolidating datasets with different class numbering schemes or adding
new classes to existing datasets.

Author: tori

Usage:
    Edit the label_dirs list with paths to your label folders
    Edit old_class_id and new_class_id variables
    Run: python3 rename_class_ids.py
"""

import os
from pathlib import Path

# Directory containing YOLO label files
label_dirs = [
    '/home/tori/YOLO/data/det_laser_yolo_1280/train/labels',
    '/home/tori/YOLO/data/det_laser_yolo_1280/valid/labels',    
    '/home/tori/YOLO/data/seg_laser_nicla_640/train/labels',
    '/home/tori/YOLO/data/seg_laser_nicla_640/valid/labels',    
    '/home/tori/YOLO/data/seg_laser_yolo_320/train/labels',
    '/home/tori/YOLO/data/seg_laser_yolo_320/valid/labels',    
    '/home/tori/YOLO/data/seg_laser_yolo_640/train/labels',
    '/home/tori/YOLO/data/seg_laser_yolo_640/valid/labels',    
    '/home/tori/YOLO/data/seg_laser_yolo_1280/train/labels',
    '/home/tori/YOLO/data/seg_laser_yolo_1280/valid/labels',    
    ]

# New class ID to replace '0'
new_class_id = 19
old_class_id = '0'

# Update each label file
for label_dir in label_dirs:
    for label_file in Path(label_dir).glob('*.txt'):
        updated_lines = []
        with open(label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = parts[0]
                    if class_id == old_class_id:  # Check if the class ID is 0
                        parts[0] = str(new_class_id)  # Replace with new class ID
                    updated_lines.append(' '.join(parts))

        # Write updated content back to the file
        with open(label_file, 'w') as file:
            file.write('\n'.join(updated_lines) + '\n')

print("Updated all label files.")
