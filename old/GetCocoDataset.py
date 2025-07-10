from pycocotools.coco import COCO
import os
import shutil

# Paths to COCO files
ann_file = '/home/tori/YOLO/annotations_trainval2017/annotations/instances_train2017.json'
img_dir = 'trial/'

# Output directories
output_dir = 'path/to/output'
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(ann_file)

# Define the categories you want
category_ids = coco.getCatIds(catNms=['cat','dog','skateboard'])

# Get all image IDs containing the desired categories
img_ids = coco.getImgIds(catIds=category_ids)
selected_images = coco.loadImgs(img_ids)

# Copy selected images
for img in selected_images:
    src_path = os.path.join(img_dir, img['file_name'])
    dest_path = os.path.join(output_dir, img['file_name'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
print(f"Copied {len(selected_images)} images.")


import json
from pathlib import Path

# Output paths for YOLO labels
yolo_labels_dir = Path(output_dir) / 'labels'
yolo_labels_dir.mkdir(parents=True, exist_ok=True)

# Function to convert COCO format to YOLO
def convert_coco_to_yolo(coco, img_ids, output_path):
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)

        # Prepare YOLO format label
        label_file = (yolo_labels_dir / f"{Path(img_info['file_name']).stem}.txt")
        with open(label_file, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id'] - 1  # YOLO uses 0-indexed class IDs
                bbox = ann['bbox']
                segmentation = ann['segmentation']

                # Convert bbox to YOLO format
                x, y, w, h = bbox
                cx = x + w / 2
                cy = y + h / 2
                yolo_bbox = [cx / img_info['width'], cy / img_info['height'], w / img_info['width'], h / img_info['height']]

                # Write detection line
                f.write(f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n")

                # If segmentation exists, append the polygons (optional)
                if isinstance(segmentation, list):  # Polygons
                    for segment in segmentation:
                        segment = [str(coord / (img_info['width'] if i % 2 == 0 else img_info['height'])) for i, coord in enumerate(segment)]
                        f.write(f"# Seg: {' '.join(segment)}\n")
        print(f"Saved labels for {img_info['file_name']}")

# Convert annotations
convert_coco_to_yolo(coco, img_ids, yolo_labels_dir)

