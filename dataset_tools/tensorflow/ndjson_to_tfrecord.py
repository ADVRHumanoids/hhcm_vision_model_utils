#  @brief Script to convert Labelbox-exported NDJSON annotations to TensorFlow TFRecord format.
# 
#  This script downloads reference images and segmentation masks, parses the annotations,
#  and generates a TFRecord file suitable for training instance segmentation models (e.g., Mask R-CNN).
# 
#  @details
#  The NDJSON file must be exported from a Labelbox project with image annotations.
#  It should contain references to image URLs and segmentation masks in Labelbox's format.
#  You must provide a Labelbox API key for downloading masks if required.
# 
#  Usage:
#  @code
#    python3 script.py --ndjson data.ndjson --config config.yaml --image_folder images/ --tfrecord output.tfrecord
#  @endcode

#!/usr/bin/env python3

import tensorflow as tf
import requests
import ndjson
import yaml
import numpy as np
import cv2
import os
import argparse

def download_mask(url, headers=None):
    """
    @brief Download the binary segmentation mask from a URL.

    @param url The URL pointing to the mask.
    @param headers Optional headers, e.g., for authorization.
    @return The mask as bytes, or None if the download fails.
    """
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def get_bbox_from_mask_bytes(mask_bytes):
    """
    @brief Extract bounding box from binary PNG mask.

    @param mask_bytes Raw bytes of the mask image (PNG format).
    @return Bounding box as [x, y, width, height] or None if no foreground.
    """
    image_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.count_nonzero(mask) == 0:
        return None
    ys, xs = np.where(mask > 0)
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]


def download_reference_image(row_data_url, external_id, dest_folder):
    """
    @brief Download the reference image and save it locally.

    @param row_data_url URL of the original image.
    @param external_id Filename for saving the image.
    @param dest_folder Directory to save the image.
    @return Local file path of the saved image or None if download fails.
    """
    os.makedirs(dest_folder, exist_ok=True)
    save_path = os.path.join(dest_folder, external_id)
    if os.path.exists(save_path):
        print(f"✅ Image already downloaded: {external_id}")
        return save_path
    try:
        r = requests.get(row_data_url, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"⬇️ Downloaded reference image: {external_id}")
        return save_path
    except Exception as e:
        print(f"❌ Failed to download image {external_id}: {e}")
        return None


def create_tf_record(image_path, image_encoded, width, height,
                      bbox_boxes, bbox_class_texts, bbox_class_ids,
                      mask_bytes_list, mask_class_texts, mask_class_ids):
    """
    @brief Create a TensorFlow Example for image, bounding boxes, and masks.

    @param image_path Path to the image file.
    @param image_encoded Raw image bytes.
    @param width Image width.
    @param height Image height.
    @param bbox_boxes List of bounding boxes.
    @param bbox_class_texts List of class names for boxes.
    @param bbox_class_ids List of class IDs for boxes.
    @param mask_bytes_list List of raw bytes for each mask.
    @param mask_class_texts List of class names for masks.
    @param mask_class_ids List of class IDs for masks.
    @return A serialized tf.train.Example.
    """
    xmins = [box[0] / width for box in bbox_boxes]
    xmaxs = [(box[0] + box[2]) / width for box in bbox_boxes]
    ymins = [box[1] / height for box in bbox_boxes]
    ymaxs = [(box[1] + box[3]) / height for box in bbox_boxes]

    feature_dict = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode()])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),

        # Bounding Boxes
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/bbox/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c.encode() for c in bbox_class_texts])),
        'image/object/bbox/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox_class_ids)),

        # Segmentation Masks
        'image/object/mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=mask_bytes_list)),
        'image/object/mask/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c.encode() for c in mask_class_texts])),
        'image/object/mask/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=mask_class_ids)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def process_ndjson_to_tfrecord(ndjson_path, image_folder, tfrecord_path, api_key, class_indices):
    """
    @brief Process NDJSON annotations and convert them to TFRecord format.

    @param ndjson_path Path to the NDJSON file.
    @param image_folder Folder to store and read reference images.
    @param tfrecord_path Path to output TFRecord file.
    @param api_key Labelbox API key for downloading masks.
    @param class_indices Dictionary mapping class names to integer IDs.
    """
    with open(ndjson_path, 'r') as f:
        data = ndjson.load(f)

    writer = tf.io.TFRecordWriter(tfrecord_path)
    headers = {'Authorization': api_key}

    for i, item in enumerate(data):
        print(f"[{i + 1}/{len(data)}] ", end="")
        external_id = item['data_row']['external_id']
        download_reference_image(
            row_data_url=item['data_row']['row_data'],
            external_id=external_id,
            dest_folder=image_folder
        )

        image_path = os.path.join(image_folder, external_id)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        with open(image_path, 'rb') as img_f:
            image_encoded = img_f.read()

        height = item['media_attributes']['height']
        width = item['media_attributes']['width']

        bbox_boxes = []
        bbox_class_texts = []
        bbox_class_ids = []
        mask_bytes_list = []
        mask_class_texts = []
        mask_class_ids = []

        objects = item['projects'][list(item['projects'].keys())[0]]['labels'][0]['annotations']['objects']

        for obj in objects:
            name = obj['name']
            annotation_kind = obj['annotation_kind']
            class_id = class_indices.get(name, 0)

            if annotation_kind == 'ImageBoundingBox':
                bbox = obj['bounding_box']
                box = [bbox['left'], bbox['top'], bbox['width'], bbox['height']]
                bbox_boxes.append(box)
                bbox_class_texts.append(name)
                bbox_class_ids.append(class_id)
            elif annotation_kind == 'ImageSegmentationMask':
                mask_url = obj.get('mask', {}).get('url')
                if mask_url:
                    mask_bytes = download_mask(mask_url, headers=headers)
                    if mask_bytes:
                        # Add mask
                        mask_bytes_list.append(mask_bytes)
                        mask_class_texts.append(name)
                        mask_class_ids.append(class_id)

                        # Compute bounding box from mask
                        box = get_bbox_from_mask_bytes(mask_bytes)
                        if box:
                            bbox_boxes.append(box)
                            bbox_class_texts.append(name)
                            bbox_class_ids.append(class_id)

        if len(mask_class_ids) == 0 and len(bbox_class_ids) == 0:
            print(f"No annotations found for image {external_id}, saving image only.")

        tf_records = create_tf_record(
            external_id,
            image_encoded,
            width,
            height,
            bbox_boxes,
            bbox_class_texts,
            bbox_class_ids,
            mask_bytes_list,
            mask_class_texts,
            mask_class_ids
        )
        writer.write(tf_records.SerializeToString())

    writer.close()
    print(f"TFRecord saved to: {tfrecord_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NDJSON to TFRecord.")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config YAML file')
    parser.add_argument('--ndjson', type=str, default='./test3.ndjson', help='Path to NDJSON file')
    parser.add_argument('--image_folder', type=str, default='./dataset_1', help='Folder to save images')
    parser.add_argument('--tfrecord', type=str, default='./dataset_1.tfrecord', help='Output TFRecord file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    api_key = config['api_key']

    class_indices = {
        "positive defect": 1,
        "negative defect": 2,
        "lines": 3,
    }

    process_ndjson_to_tfrecord(
        ndjson_path=args.ndjson,
        image_folder=args.image_folder,
        tfrecord_path=args.tfrecord,
        api_key=api_key,
        class_indices=class_indices
    )