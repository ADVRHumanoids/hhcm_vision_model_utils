#!/usr/bin/env python3
"""
TFRecord to COCO Format Converter.

Converts TensorFlow TFRecord files containing instance segmentation data to COCO dataset format.
This script reads TFRecord files with images and segmentation masks, extracts the data, and
converts it to COCO JSON format with separate directories for images and instance masks.

The conversion process:
1. Parses TFRecord files to extract images, masks, and labels
2. Saves images to Images/ directory
3. Creates colored composite instance masks and saves to InstanceMasks/
4. Generates COCO format annotations.json with:
   - Image metadata (dimensions, filenames)
   - Category definitions with unique IDs
   - Instance annotations with bounding boxes and polygon segmentations
   - Area calculations and contour extraction

Useful for converting TensorFlow-based datasets to a format compatible with other
frameworks, annotation tools, and training pipelines that expect COCO format.

Author: Alessio Lovato
Modified by: Alessio Lovato, 03-11-2025

Arguments:
    tfrecord_path: Path to input TFRecord file
    output_root: Root directory for COCO format output

Output Structure:
    output_root/
    ├── Images/             # Extracted RGB images
    ├── InstanceMasks/      # Colored composite instance segmentation masks
    └── annotations.json    # COCO format annotations with bbox and polygons
"""

import os
import io
import cv2
import json
import matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf


def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example to extract image and annotation features.

    Defines the feature schema for TFRecord parsing, including image data, metadata,
    and instance segmentation masks with their corresponding class labels.

    Args:
        example_proto: Serialized TFRecord example proto

    Returns:
        dict: Parsed features containing:
            - image/encoded: Encoded image bytes (JPEG/PNG)
            - image/filename: Original filename string
            - image/height: Image height in pixels
            - image/width: Image width in pixels
            - image/object/mask: Sparse tensor of encoded mask images
            - image/object/mask/class/text: Sparse tensor of class name strings
            - image/object/mask/class/label: Sparse tensor of class label integers
    """
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/mask': tf.io.VarLenFeature(tf.string),
        'image/object/mask/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/mask/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def tfrecord_to_coco(tfrecord_path, output_root):
    """
    Convert TFRecord dataset to COCO format with images, masks, and annotations.

    Processes each record in the TFRecord file, extracting images and instance masks,
    generating COCO format annotations with bounding boxes and polygon segmentations,
    and creating colored composite masks for visualization.

    The function:
    1. Creates output directory structure (Images/, InstanceMasks/)
    2. Iterates through TFRecord dataset
    3. Saves each image to disk
    4. Processes instance masks:
       - Assigns unique colors (using tab20 colormap)
       - Computes bounding boxes from masks
       - Extracts polygon contours using OpenCV
       - Calculates instance areas
    5. Builds COCO format data structure with images, categories, and annotations
    6. Saves composite colored masks for visualization
    7. Writes annotations.json file

    Args:
        tfrecord_path (str): Path to input TFRecord file
        output_root (str): Root directory where COCO dataset will be created

    Output Files:
        - {output_root}/Images/*.jpg: Extracted images
        - {output_root}/InstanceMasks/*.png: Colored composite instance masks
        - {output_root}/annotations.json: COCO format annotations

    Returns:
        None: Outputs are written to disk
    """
    os.makedirs(f"{output_root}/Images", exist_ok=True)
    os.makedirs(f"{output_root}/InstanceMasks", exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  # text → category_id
    ann_id = 1
    image_id = 1

    max_ann_count = 0
    max_ann_image_id = None

    for record in tqdm(dataset):
        # Get image metadata
        image_data = record['image/encoded'].numpy()
        filename = record['image/filename'].numpy().decode()
        height = int(record['image/height'].numpy())
        width = int(record['image/width'].numpy())

        # Save image
        img_path = os.path.join(output_root, 'Images', filename)
        with open(img_path, 'wb') as f:
            f.write(image_data)

        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "coco_url": f"{output_root}/Images/{filename}"
        }
        coco["images"].append(image_info)

        # Prepare mask canvas
        composite_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Get masks and labels
        masks = tf.sparse.to_dense(record['image/object/mask']).numpy()
        class_texts = tf.sparse.to_dense(record['image/object/mask/class/text']).numpy()
        class_labels = tf.sparse.to_dense(record['image/object/mask/class/label']).numpy()

        this_image_ann_count = 0

        for i, (mask_bytes, label_text, label_id) in enumerate(zip(masks, class_texts, class_labels)):
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_np = np.array(mask)

            # Assign unique color
            # Use a qualitative colormap for distinct colors
            tab20 = matplotlib.colormaps['tab20']
            rgba = tab20(i % 20)  # 20 distinct colors
            color = [int(255 * c) for c in rgba[:3]]
            for c in range(3):
                composite_mask[:, :, c][mask_np > 0] = color[c]

            # Category registration
            label_text_decoded = label_text.decode()
            if label_text_decoded not in category_map:
                category_map[label_text_decoded] = int(label_id)
                coco["categories"].append({
                    "id": int(label_id),
                    "name": label_text_decoded,
                    "supercategory": "defect"
                })

            # Compute bbox
            y_indices, x_indices = np.where(mask_np > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x_min, y_min = int(np.min(x_indices)), int(np.min(y_indices))
            x_max, y_max = int(np.max(x_indices)), int(np.max(y_indices))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = int(np.sum(mask_np > 0))

            # Ensure binary mask (0 or 1)
            binary_mask = (mask_np > 0).astype(np.uint8)

            # Use OpenCV to find external contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) >= 6:  # valid polygon has at least 3 points (6 coords)
                    segmentation.append(contour)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label_id),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation
            })
            ann_id += 1
            this_image_ann_count += 1

        if this_image_ann_count > max_ann_count:
            max_ann_count = this_image_ann_count
            max_ann_image_id = image_id

        # Save composite mask
        mask_path = os.path.join(output_root, 'InstanceMasks', filename.replace('.jpg', '.png'))
        Image.fromarray(composite_mask).save(mask_path)
        image_id += 1

    print(f"Image with most annotations: ID {max_ann_image_id} ({max_ann_count} annotations)")

    # Save annotations
    with open(os.path.join(output_root, 'annotations.json'), 'w') as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecord_path", help="Path to TFRecord")
    parser.add_argument("output_root", help="Output folder root")
    args = parser.parse_args()

    tfrecord_to_coco(args.tfrecord_path, args.output_root)
