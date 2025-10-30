#!/usr/bin/env python3
"""
Convert Labelbox NDJSON annotations to LabelMe JSON format with line_color.
@brief Converts NDJSON annotations from Labelbox to LabelMe JSON format.
@details This script downloads images and masks, converts them to polygons,
         and saves them in the LabelMe JSON format with specified line colors.

Arguments:
--config: Path to config YAML file containing the API key.
--ndjson: Path to the NDJSON file with annotations.
--image_folder: Folder to save images and JSON files.
--save_masks: Flag to save individual masks as PNG files in the image folder.
"""
import os
import cv2
import ndjson
import json
import yaml
import argparse
import requests
import numpy as np

def download_file(url, save_path, headers=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        return save_path
    try:
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return None

def mask_to_polygons(mask_bytes):
    """
    Convert binary mask (PNG bytes) to polygons (list of points).
    """
    image_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.count_nonzero(mask) == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float).tolist()
        if len(points) >= 3:  # valid polygon
            polygons.append(points)
    return polygons

def process_ndjson_to_labelme(ndjson_path, image_folder, api_key, save_masks=False):
    """
    Convert Labelbox NDJSON annotations to LabelMe JSON format (with line_color).
    """
    with open(ndjson_path, 'r') as f:
        data = ndjson.load(f)

    headers = {'Authorization': api_key}
    total = len(data)

    for i, item in enumerate(data):
        external_id = item['data_row']['external_id']
        image_url = item['data_row']['row_data']
        img_path = download_file(image_url, os.path.join(image_folder, external_id))

        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read {img_path}")
            continue
        height, width = img.shape[:2]

        # Create LabelMe JSON
        labelme_json = {
            "version": "3.21.1",
            "flags": {},
            "shapes": [],
            "imagePath": external_id,
            "imageData": "",
            "imageHeight": height,
            "imageWidth": width
        }

        labels = item['projects'][list(item['projects'].keys())[0]]['labels']
        if not labels:
            print(f"⚠️ No labels for {external_id}")
        else:
            objects = item['projects'][list(item['projects'].keys())[0]]['labels'][0]['annotations']['objects']

            for obj in objects:
                if obj['annotation_kind'] == 'ImageSegmentationMask':
                    mask_url = obj.get('mask', {}).get('url')
                    comp_mask = obj.get('composite_mask', {})
                    color_rgb = comp_mask.get('color_rgb', None)

                    if mask_url:
                        mask_file = os.path.join(image_folder, f"{obj['feature_id']}.png")
                        mask_path = download_file(mask_url, mask_file, headers=headers)
                        if mask_path and os.path.exists(mask_path):
                            with open(mask_path, 'rb') as f:
                                mask_content = f.read()
                            polygons = mask_to_polygons(mask_content)

                            for poly in polygons:
                                shape = {
                                    "label": obj['name'],
                                    "line_color": color_rgb if color_rgb else "null",
                                    "fill_color": color_rgb if color_rgb else "null",
                                    "points": poly,
                                    "shape_type": "polygon",
                                    "flags": {}
                                }
                                labelme_json["shapes"].append(shape)
                        elif mask_path is None:
                            print(f"❌ Failed to download mask for {external_id}")
                            os.remove(img_path)  # Remove image if mask download fails
                            continue
                        # Remove mask file if not saving
                        if not save_masks and os.path.exists(mask_file):
                            os.remove(mask_file)

        # Save JSON next to image
        json_path = os.path.join(image_folder, os.path.splitext(external_id)[0] + ".json")
        with open(json_path, 'w') as jf:
            json.dump(labelme_json, jf, indent=4)

        print(f"[{i+1}/{total}] Processed {external_id} -> {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Labelbox NDJSON to LabelMe JSON dataset (with line_color).")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config YAML with api_key')
    parser.add_argument('--ndjson', type=str, default='./data.ndjson', help='Path to NDJSON file')
    parser.add_argument('--image-folder', type=str, default='./dataset_labelme', help='Folder to save images + JSONs')
    parser.add_argument('--save-masks', action='store_true', help='Save masks as PNG files in the image folder')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    api_key = config['api_key']

    process_ndjson_to_labelme(
        ndjson_path=args.ndjson,
        image_folder=args.image_folder,
        api_key=api_key,
        save_masks=args.save_masks
    )
