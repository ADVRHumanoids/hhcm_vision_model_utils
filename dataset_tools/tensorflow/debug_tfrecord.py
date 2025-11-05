#!/usr/bin/env python3
"""
Visualize and debug TensorFlow TFRecord files for instance segmentation.

Reads TFRecord files, decodes images, masks, and bounding boxes, and displays them
in an interactive matplotlib subplot for visual inspection. Useful for verifying
data correctness before training Mask R-CNN or other instance segmentation models.

Author: Alessio Lovato

Arguments:
    tfrecord_path: Path to .tfrecord file (positional)
    --max-records: Maximum number of records to visualize (default: -1 for all)

Controls:
    ESC: Close current visualization and move to next record

Requirements:
    pip install tensorflow matplotlib pillow numpy
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io


def parse_tfrecord(example_proto):
    """
    @brief Parses a single TFRecord example using a feature schema.
    @param example_proto A serialized TFRecord example.
    @return A dictionary with parsed features including image, bounding boxes, and masks.
    """
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),

        # Bounding boxes
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/bbox/class/label': tf.io.VarLenFeature(tf.int64),

        # Instance masks
        'image/object/mask': tf.io.VarLenFeature(tf.string),
        'image/object/mask/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/mask/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def load_and_plot_tfrecord(tfrecord_path, max_records=-1):
    """
    @brief Loads and plots each TFRecord entry including image, masks, and bounding boxes.
    @param tfrecord_path Path to the TFRecord file.
    @param max_records Maximum number of records to visualize. If -1, visualize all.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    dataset_list = list(dataset.take(max_records)) if max_records != -1 else list(dataset)
    print("Number of elements in the dataset:", len(dataset_list))

    for i, record in enumerate(dataset_list):
        print(f"\n🔍 Processing record {i + 1}")

        # Decode image
        image = tf.io.decode_image(record['image/encoded'], channels=3)
        image_np = image.numpy()

        # Metadata
        height = record['image/height'].numpy()
        width = record['image/width'].numpy()
        filename = record['image/filename'].numpy().decode('utf-8')

        # Bounding boxes
        xmin = tf.sparse.to_dense(record['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(record['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(record['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(record['image/object/bbox/ymax']).numpy()
        bbox_classes = tf.sparse.to_dense(record['image/object/bbox/class/text']).numpy()
        bbox_classes = [c.decode('utf-8') for c in bbox_classes]

        # Masks
        mask_bytes_list = tf.sparse.to_dense(record['image/object/mask']).numpy()
        mask_classes = tf.sparse.to_dense(record['image/object/mask/class/text']).numpy()
        mask_classes = [c.decode('utf-8') for c in mask_classes]

        print(f"Image: {filename}, Size: {width}x{height}, BBoxes: {len(xmin)}, Masks: {len(mask_bytes_list)}")
        print("Mask Classes:", mask_classes)

        num_masks = len(mask_bytes_list)
        fig, axs = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
        axs = axs if num_masks > 0 else [axs]

        # Plot image with bounding boxes
        axs[0].imshow(image_np)
        axs[0].set_title(f"Image: {filename}")
        axs[0].axis('off')
        for x0, x1, y0, y1, cls in zip(xmin, xmax, ymin, ymax, bbox_classes):
            if cls != "lines":
                continue
            x0_abs = int(x0 * width)
            x1_abs = int(x1 * width)
            y0_abs = int(y0 * height)
            y1_abs = int(y1 * height)
            axs[0].add_patch(plt.Rectangle((x0_abs, y0_abs), x1_abs - x0_abs, y1_abs - y0_abs,
                                           edgecolor='red', facecolor='none', lw=2))
            axs[0].text(x0_abs, max(y0_abs - 5, 0), cls, color='red', fontsize=10, backgroundcolor='white')

        # Plot each mask in grayscale
        for j, mask_bytes in enumerate(mask_bytes_list):
            mask_img = Image.open(io.BytesIO(mask_bytes))
            mask_np = np.array(mask_img)
            axs[j + 1].imshow(mask_np, cmap='gray', vmin=0, vmax=255)
            cls_name = mask_classes[j] if j < len(mask_classes) else "Unknown"
            axs[j + 1].set_title(f"Mask {j + 1}: {cls_name}")
            axs[j + 1].axis('off')

        def on_key(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize TFRecord for debugging.")
    parser.add_argument("tfrecord_path", help="Path to the .tfrecord file")
    parser.add_argument("--max-records", type=int, default=-1, help="Max records to visualize")
    args = parser.parse_args()

    load_and_plot_tfrecord(args.tfrecord_path, max_records=args.max_records)
