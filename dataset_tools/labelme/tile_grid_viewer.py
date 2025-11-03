#!/usr/bin/env python3
"""
Interactive tile grid viewer for LabelMe tiled datasets.

Reconstructs and displays original image grids from tiled datasets, showing tile numbers
and annotation masks overlaid. Allows interactive selection and deletion of problematic
tiles with logging. Useful for quality control after tiling augmentation.

Arguments:
    --tile-folder: Path to folder containing tiled images
    --tiling-log: Path to tiling log file (created by tiling_augmentation.py)

Controls:
    Left Click: Toggle tile selection for deletion
    D: Delete selected tiles and their JSON files
    N / Right Arrow: Next base image
    P / Left Arrow: Previous base image
    Q / ESC: Quit viewer

Author: Alessio Lovato
"""
import os
import cv2
import numpy as np
import glob
import json
import argparse
from pathlib import Path

PADDING = 20
TITLE_HEIGHT = 80
WINDOW_NAME = "Tile Grid Viewer"

# Helper: get all base names
def get_base_names(tile_folder):
    tile_files = glob.glob(os.path.join(tile_folder, "*.jpg"))
    base_names = set()
    for f in tile_files:
        fname = os.path.basename(f)
        if any(s in fname for s in ["_q0", "_q1", "_q2", "_q3", "_scaled_full"]):
            continue
        parts = fname.split("_")
        if len(parts) > 1 and parts[-1].split(".")[0].isdigit():
            base = "_".join(parts[:-1])
            base_names.add(base)
    return sorted(list(base_names))

# Helper: parse tiling log file
def parse_tiling_log(log_path):
    info = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(':')
            base_name = parts[0]
            grid_part = [p for p in parts[1].split(',') if 'grid' in p][0]
            tile_part = [p for p in parts[1].split(',') if 'tile' in p][0]
            orig_part = [p for p in parts[1].split(',') if 'orig' in p][0]
            n_rows, n_cols = map(int, grid_part.replace('grid','').strip().split('x'))
            tile_w, tile_h = map(int, tile_part.replace('tile','').strip().split('x'))
            orig_w, orig_h = map(int, orig_part.replace('orig','').strip().split('x'))
            info[base_name] = {
                'n_rows': n_rows,
                'n_cols': n_cols,
                'tile_w': tile_w,
                'tile_h': tile_h,
                'orig_w': orig_w,
                'orig_h': orig_h
            }
    return info

# Helper: draw masks and tile number, using color from JSON
def draw_masks_and_number(tile_img, json_path, tile_number, alpha=0.4):
    overlay = tile_img.copy()
    if os.path.exists(json_path):
        with open(json_path) as f:
            anns = json.load(f)
        for shape in anns.get("shapes", []):
            pts = np.array(shape.get("points", []), dtype=np.int32)
            color = shape.get("color", [0, 255, 0])
            # If color is a string, convert to tuple
            if isinstance(color, str):
                try:
                    color = tuple(map(int, color.strip('[]()').split(',')))
                except Exception:
                    color = (0, 255, 0)
            if pts.shape[0] >= 3:
                cv2.fillPoly(overlay, [pts], color)
    out = cv2.addWeighted(overlay, alpha, tile_img, 1-alpha, 0)
    cv2.putText(out, str(tile_number), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
    return out

# Helper: delete tile and JSON file
def delete_tile_and_json(tile_path, deleted_log):
    json_path = os.path.splitext(tile_path)[0] + ".json"
    if os.path.exists(tile_path):
        os.remove(tile_path)
        deleted_log.append(str(tile_path))
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted_log.append(str(json_path))

# Helper: log deleted files
def log_deleted_files(deleted_log, log_path):
    with open(log_path, 'w') as f:
        for path in deleted_log:
            f.write(f"{path}\n")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Tile Grid Viewer for LabelMe Tiled Dataset")
    parser.add_argument('--folder', type=str, required=True, help='Path to tiled dataset folder')
    parser.add_argument('--info', type=str, default="tiling_log.txt", help='Path to dataset info file (tiling log)')
    parser.add_argument('--log', type=str, default="deleted_files.txt", help='Path to save deleted files log. (Default: <folder>/deleted_files.txt)')
    args = parser.parse_args()

    tile_folder = args.folder
    info_path = os.path.join(tile_folder, args.info)
    log_path = args.log if args.log else os.path.join(tile_folder, "deleted_files.txt")

    if args.log and not os.path.exists(log_path):
        print("Error: Specified log path does not exist.")
        return
    if not os.path.exists(info_path):
        if not args.info:
            print("Error: Default info path '<folder>/tiling_log.txt' does not exist.")
        else:
            print("Error: Custom info path does not exist.")
        return
    if not os.path.exists(tile_folder):
        print("Error: Specified tile folder does not exist.")
        return

    info = parse_tiling_log(info_path)
    base_names = get_base_names(tile_folder)
    print(f"Found {len(base_names)} original images.")
    all_deleted_log = []
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    idx = 0
    while idx < len(base_names):
        base_name = base_names[idx]
        if base_name not in info:
            print(f"No log info for {base_name}, skipping.")
            idx += 1
            continue
        params = info[base_name]
        n_rows = params['n_rows']
        n_cols = params['n_cols']
        tile_w = params['tile_w']
        tile_h = params['tile_h']
        grid_size = n_rows * n_cols
        tile_imgs = [np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 180 for _ in range(grid_size)]
        tile_paths = [os.path.join(tile_folder, f"{base_name}_{i+1}.jpg") for i in range(grid_size)]
        for f in range(grid_size):
            image = tile_paths[f]
            if os.path.exists(image):
                img = cv2.imread(image)
                mask_img = draw_masks_and_number(img, os.path.splitext(image)[0]+'.json', f+1)
                tile_imgs[f] = mask_img
        selected = [False] * grid_size
        def redraw_canvas():
            display_imgs = []
            for i in range(grid_size):
                img = tile_imgs[i].copy()
                if selected[i]:
                    cv2.rectangle(img, (0,0), (tile_w-1, tile_h-1), (0,0,255), 5)
                display_imgs.append(img)
            H = n_rows * tile_h + (n_rows + 1) * PADDING + TITLE_HEIGHT
            W = n_cols * tile_w + (n_cols + 1) * PADDING
            canvas = np.ones((H, W, 3), dtype=np.uint8) * 220
            tidx = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    y = r * tile_h + (r + 1) * PADDING + TITLE_HEIGHT
                    x = c * tile_w + (c + 1) * PADDING
                    canvas[y:y+tile_h, x:x+tile_w] = display_imgs[tidx]
                    tidx += 1
            # Center image name above tiles
            text = f"Image: {base_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (W - text_size[0]) // 2
            text_y = TITLE_HEIGHT // 2 + text_size[1] // 2
            cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, canvas)
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                tidx = 0
                for r in range(n_rows):
                    for c in range(n_cols):
                        y0 = r * tile_h + (r + 1) * PADDING + TITLE_HEIGHT
                        x0 = c * tile_w + (c + 1) * PADDING
                        if x0 <= x < x0+tile_w and y0 <= y < y0+tile_h:
                            if os.path.exists(tile_paths[tidx]):
                                selected[tidx] = not selected[tidx]
                        tidx += 1
                redraw_canvas()
        redraw_canvas()
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            print(f"Showing {base_name}. Click to select tiles, Enter to delete selected or move to the next image, ESC to exit.")
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                print("Deleted files:")
                for f in all_deleted_log:
                    print(f)
                log_deleted_files(all_deleted_log, log_path)
                return
            elif key == 13:
                # Delete selected tiles
                for i, sel in enumerate(selected):
                    if sel and os.path.exists(tile_paths[i]):
                        print(f"Deleting {tile_paths[i]} and its JSON.")
                        delete_tile_and_json(tile_paths[i], all_deleted_log)
                break
        idx += 1
    cv2.destroyAllWindows()
    log_deleted_files(all_deleted_log, log_path)
    print("Done.")

if __name__ == "__main__":
    main()
