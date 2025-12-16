#!/usr/bin/env python3
"""Interactive COCO viewer (single window)

Shows original image on the left and annotated (original + mask overlay)
on the right in a single window. Use 'd' to advance, 'a' to go back, and 'q'
to quit. Displays a counter (i/N).

Requires: pycocotools, opencv-python, numpy

Usage:
  python demo/coco_gallery.py --coco annotations.json --images-root /path/to/images
"""
import argparse
import os
import sys
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
]


def color_for_category(cat_id: int):
    # deterministic color from palette
    c = PALETTE[cat_id % len(PALETTE)]
    # palette given as RGB; convert to BGR for OpenCV
    return (int(c[2]), int(c[1]), int(c[0]))


def overlay_annotations(image_bgr, anns, coco, alpha=0.45):
    """Return annotated image (BGR) with segmentation filled and bboxes/labels drawn.

    image_bgr: numpy array HxWx3 (BGR)
    anns: list of annotation dicts (COCO format)
    coco: COCO object (to resolve category names)
    """
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    for ann_idx, ann in enumerate(anns):
        cat_id = ann.get("category_id", 0)
        color = color_for_category(cat_id)

        # segmentation handling
        seg = ann.get("segmentation", None)
        if seg:
            # polygon list
            if isinstance(seg, list):
                for poly in seg:
                    if not poly:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color)
            else:
                # RLE or compressed RLE
                try:
                    m = maskUtils.decode(seg)
                    if m is None:
                        continue
                    if m.ndim == 3:
                        # if multiple channels, take first
                        m = m[:, :, 0]
                    mask_u8 = (m > 0).astype('uint8') * 255
                    contours_info = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
                    for c in contours:
                        if c is None or c.size < 6:
                            continue
                        pts = c.reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(overlay, [pts], color)
                except Exception:
                    # ignore segmentation errors
                    pass

        # draw bbox
        bbox = ann.get("bbox", None)
        if bbox and len(bbox) >= 4:
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = int(round(x)), int(round(y)), int(round(x + bw)), int(round(y + bh))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2)

        # draw label
        try:
            cat = coco.loadCats(cat_id)[0]["name"]
        except Exception:
            cat = str(cat_id)
        label = cat
        # background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = max(0, int(bbox[0]) if bbox else 5), max(th + 4, 12)
        cv2.rectangle(overlay, (tx, ty - th - 4), (tx + tw + 4, ty), color, -1)
        cv2.putText(overlay, label, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # blend
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, overlay, alpha, 0)
    return blended


def load_image(path):
    img = cv2.imread(path)
    return img


def run_interactive(coco_path, images_root):
    coco = COCO(coco_path)
    img_ids = coco.getImgIds()
    if not img_ids:
        print("No images found in COCO file.")
        return

    imgs = coco.loadImgs(img_ids)
    N = len(imgs)
    idx = 0

    winname = "COCO Viewer (a/d to navigate, q to quit)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        info = imgs[idx]
        fname = info.get("file_name")
        if not os.path.isabs(fname):
            img_path = os.path.join(images_root, fname)
        else:
            img_path = fname

        img = load_image(img_path)
        if img is None:
            # show placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Missing: {os.path.basename(img_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            combined = np.concatenate([placeholder, placeholder], axis=1)
        else:
            # get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=info["id"]) if "id" in info else []
            anns = coco.loadAnns(ann_ids) if ann_ids else []

            annotated = overlay_annotations(img, anns, coco)
            # ensure same height
            h = max(img.shape[0], annotated.shape[0])
            # no resize: images should match
            combined = np.concatenate([img, annotated], axis=1)

        # overlay counter text
        counter = f"{idx+1}/{N}"
        cv2.rectangle(combined, (5, 5), (140, 34), (0, 0, 0), -1)
        cv2.putText(combined, counter, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(winname, combined)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('d'):
            idx = (idx + 1) % N
        elif key == ord('a'):
            idx = (idx - 1) % N
        elif key == ord('q'):
            break
        else:
            # ignore other keys
            continue

    cv2.destroyWindow(winname)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive COCO single-window viewer")
    p.add_argument("--coco", required=True, help="COCO annotations.json path")
    p.add_argument("--images-root", required=True, help="Path to folder containing images referenced in COCO file")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.coco):
        print(f"COCO file not found: {args.coco}")
        sys.exit(1)
    if not os.path.isdir(args.images_root):
        print(f"Images root not found or not a directory: {args.images_root}")
        sys.exit(1)
    run_interactive(args.coco, args.images_root)


if __name__ == '__main__':
    main()