#!/usr/bin/env python3
"""
Tiling augmentation for LabelMe datasets with annotation preservation.

Splits large images into smaller overlapping tiles while accurately clipping and transforming
polygon annotations to match new tile coordinates. Uses Shapely for robust geometric operations
including polygon intersection, validation, and repair.

Arguments:
    --input-dir: Path to input LabelMe dataset folder
    --output-dir: Path to output folder for tiled dataset
    --width: Width of each tile in pixels (default: 640)
    --height: Height of each tile in pixels (default: 640)
    --overlap: Overlap percentage between tiles 0.0-1.0 (default: 0.0)
    --zoom-out: Create zoomed-out full-image versions (flag)
    --pad-border: Pad border tiles instead of overlapping (flag)

Requirements:
    pip install shapely opencv-python numpy colorama

Output:
    - Tiled images: {base_name}_{tile_number}.jpg
    - Tile annotations: {base_name}_{tile_number}.json
    - Tiling log: tiling_log.txt (for reconstruction)
    - Scaled versions (if --zoom-out): {base_name}_scaled_full.jpg

Author: Alessio Lovato
"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from colorama import Fore # For colored terminal output
from shapely.geometry import Polygon, box


def valid_polygon(p):
    """Check if a polygon is valid."""
    return p.is_valid and not p.is_empty

def clip_points(pts, w, h):
    """Clip points to be within tile boundaries [0, w] x [0, h]."""
    pts[:, 0] = np.clip(pts[:, 0], 0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0, h)
    return pts

def generate_tiles(img, anns, out_dir, base_name, tile_w, tile_h, overlap, pad_border=False):
    """Generate tiles from a large image with annotations. Optionally pad border tiles."""
    H, W = img.shape[:2]
    tiles_info = []
    if pad_border:
        # Calculate grid size
        n_cols = int(np.ceil(W / tile_w))
        n_rows = int(np.ceil(H / tile_h))
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * tile_w
                y0 = row * tile_h
                x1 = min(x0 + tile_w, W)
                y1 = min(y0 + tile_h, H)
                crop = img[y0:y1, x0:x1]
                # Pad if needed
                pad_bottom = tile_h - (y1 - y0)
                pad_right = tile_w - (x1 - x0)
                crop_padded = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
                # Adjust annotations
                new_shapes = []
                crop_box = box(0, 0, x1-x0, y1-y0)
                for shape in anns["shapes"]:
                    pts = np.array(shape["points"], dtype=np.float32)
                    
                    # Create polygon and validate/repair if needed
                    try:
                        mask = Polygon([(px - x0, py - y0) for px, py in pts])
                        if not mask.is_valid:
                            # Try to repair invalid polygon with buffer(0) trick
                            mask = mask.buffer(0)
                            if not mask.is_valid or mask.is_empty:
                                print(Fore.YELLOW + f"Warning: Skipping invalid polygon in {base_name}" + Fore.RESET)
                                continue
                    except Exception as e:
                        print(Fore.YELLOW + f"Warning: Failed to create polygon in {base_name}: {e}" + Fore.RESET)
                        continue
                    
                    inter = crop_box.intersection(mask)
                    if inter.is_empty or not inter.is_valid:
                        continue
                    polygons_to_save = []
                    if inter.geom_type == "Polygon":
                        if valid_polygon(inter):
                            polygons_to_save.append(inter)
                    elif inter.geom_type == "MultiPolygon":
                        for geom in inter.geoms:
                            if geom.geom_type == "Polygon" and valid_polygon(geom):
                                polygons_to_save.append(geom)
                    elif inter.geom_type == "GeometryCollection":
                        for geom in inter.geoms:
                            if geom.geom_type == "Polygon" and valid_polygon(geom):
                                polygons_to_save.append(geom)
                            elif geom.geom_type == "MultiPolygon":
                                for sub_geom in geom.geoms:
                                    if sub_geom.geom_type == "Polygon" and valid_polygon(sub_geom):
                                        polygons_to_save.append(sub_geom)
                    for poly in polygons_to_save:
                        try:
                            coords = list(poly.exterior.coords)
                            if len(coords) > 1 and coords[0] == coords[-1]:
                                coords = coords[:-1]
                            if len(coords) < 3:
                                continue
                            if poly.area < 1.0:
                                continue
                            clipped_shape = shape.copy()
                            clipped_shape["points"] = [[float(x), float(y)] for x, y in coords]
                            new_shapes.append(clipped_shape)
                        except Exception as e:
                            print(Fore.YELLOW + f"Warning: Failed to process polygon in {base_name}: {e}" + Fore.RESET)
                            continue
                if not new_shapes:
                    continue
                tile_id = row * n_cols + col + 1
                new_anns = {
                    "version": anns.get("version", "5.0.1"),
                    "flags": {},
                    "shapes": new_shapes,
                    "imagePath": f"{base_name}_{tile_id}.jpg",
                    "imageHeight": tile_h,
                    "imageWidth": tile_w
                }
                cv2.imwrite(str(out_dir / f"{base_name}_{tile_id}.jpg"), crop_padded)
                with open(out_dir / f"{base_name}_{tile_id}.json", "w") as f:
                    json.dump(new_anns, f, indent=2)
        return n_rows, n_cols
    else:
        # step size with overlap
        step_x = int(tile_w * (1 - overlap))
        step_y = int(tile_h * (1 - overlap))

        # start from bottom-left (y from bottom to top, x from left to right)
        y_starts = list(range(0, H - tile_h, step_y))
        x_starts = list(range(0, W - tile_w, step_x))

        # ensure last row and column included
        if not y_starts or y_starts[-1] != H - tile_h:
            y_starts.append(H - tile_h)
        if not x_starts or x_starts[-1] != W - tile_w:
            x_starts.append(W - tile_w)
        
        tile_id = 0
        crop_box = box(0, 0, tile_w, tile_h)
        for y in y_starts:
            for x in x_starts:
                tile_id += 1
                crop_img = img[y:y+tile_h, x:x+tile_w]
                

                new_shapes = []
                for i, shape in enumerate(anns["shapes"]):
                    pts = np.array(shape["points"], dtype=np.float32) # Original points in image coords

                    # Create polygon and validate/repair if needed
                    try:
                        mask = Polygon([(px - x, py - y) for px, py in pts]) # Convert in polygon in tile coords
                        if not mask.is_valid:
                            # Try to repair invalid polygon with buffer(0) trick
                            mask = mask.buffer(0)
                            if not mask.is_valid or mask.is_empty:
                                print(Fore.YELLOW + f"Warning: Skipping invalid polygon in {base_name}, shape {i}" + Fore.RESET)
                                continue
                    except Exception as e:
                        print(Fore.YELLOW + f"Warning: Failed to create polygon in {base_name}, shape {i}: {e}" + Fore.RESET)
                        continue
                    
                    inter = crop_box.intersection(mask)

                    if inter.is_empty or not inter.is_valid:
                        continue

                    # Extract polygon coordinates from intersection result
                    polygons_to_save = []
                    
                    if inter.geom_type == "Polygon":
                        if valid_polygon(inter):
                            polygons_to_save.append(inter)
                            
                    elif inter.geom_type == "MultiPolygon":
                        # Extract all valid polygons from MultiPolygon
                        for geom in inter.geoms:
                            if geom.geom_type == "Polygon" and valid_polygon(geom):
                                polygons_to_save.append(geom)
                                
                    elif inter.geom_type == "GeometryCollection":
                        # Extract polygons from GeometryCollection, ignore other geometry types
                        for geom in inter.geoms:
                            if geom.geom_type == "Polygon" and valid_polygon(geom):
                                polygons_to_save.append(geom)
                            elif geom.geom_type == "MultiPolygon":
                                for sub_geom in geom.geoms:
                                    if sub_geom.geom_type == "Polygon" and valid_polygon(sub_geom):
                                        polygons_to_save.append(sub_geom)
                    
                    # Convert valid polygons to LabelMe format
                    for poly in polygons_to_save:
                        try:
                            # Get exterior coordinates (main polygon boundary)
                            coords = list(poly.exterior.coords)
                            
                            # Remove duplicate last point if exists
                            if len(coords) > 1 and coords[0] == coords[-1]:
                                coords = coords[:-1]
                            
                            # Need at least 3 points for a valid polygon
                            if len(coords) < 3:
                                continue
                                
                            # Check if polygon area is meaningful
                            if poly.area < 1.0:  # Skip very small polygons
                                continue
                            
                            clipped_shape = shape.copy()
                            clipped_shape["points"] = [[float(x), float(y)] for x, y in coords]
                            new_shapes.append(clipped_shape)
                            
                        except Exception as e:
                            print(Fore.YELLOW + f"Warning: Failed to process polygon in {base_name}: {e}" + Fore.RESET)
                            continue

                if not new_shapes:
                    continue

                new_anns = {
                    "version": anns.get("version", "5.0.1"),
                    "flags": {},
                    "shapes": new_shapes,
                    "imagePath": f"{base_name}_{tile_id}.jpg",
                    "imageHeight": tile_h,
                    "imageWidth": tile_w
                }

                cv2.imwrite(str(out_dir / f"{base_name}_{tile_id}.jpg"), crop_img)
                with open(out_dir / f"{base_name}_{tile_id}.json", "w") as f:
                    json.dump(new_anns, f, indent=2)
        # Estimate grid size for logging
        n_cols = len(x_starts)
        n_rows = len(y_starts)
        return n_rows, n_cols

def generate_scaled_quadrants(img, anns, out_dir, base_name, tile_w, tile_h):
    """Divide image into 4 quadrants, resize, clip & scale annotations."""
    H, W = img.shape[:2]
    quads = [
        (0, 0, W // 2, H // 2, "q0"),
        (W // 2, 0, W, H // 2, "q1"),
        (0, H // 2, W // 2, H, "q2"),
        (W // 2, H // 2, W, H, "q3")
    ]
    saved = 0
    for (x0, y0, x1, y1, suffix) in quads:
        crop = img[y0:y1, x0:x1]
        sx, sy = tile_w / float(x1 - x0), tile_h / float(y1 - y0)

        new_shapes = []
        for i, shape in enumerate(anns.get("shapes", [])):
            pts = np.array(shape.get("points", []), dtype=np.float32)
            if pts.size < 6:  # Need at least 3 points (6 coordinates)
                continue

            # Create polygon in quadrant coordinates
            shifted_pts = [(px - x0, py - y0) for px, py in pts]
            
            # Validate and repair polygon before intersection
            try:
                mask = Polygon(shifted_pts)
                if not mask.is_valid:
                    # Try to repair invalid polygon with buffer(0) trick
                    mask = mask.buffer(0)
                    if not mask.is_valid or mask.is_empty:
                        print(Fore.YELLOW + f"Warning: Skipping invalid polygon in {base_name}, shape {i}" + Fore.RESET)
                        continue
            except Exception as e:
                print(Fore.YELLOW + f"Warning: Failed to create polygon in {base_name}, shape {i}: {e}" + Fore.RESET)
                continue
            
            # Create quadrant box and find intersection
            quadrant_box = box(0, 0, x1 - x0, y1 - y0)
            inter = mask.intersection(quadrant_box)
            
            if inter.is_empty or not inter.is_valid:
                continue

            # Extract polygon coordinates from intersection result (same as generate_tiles)
            polygons_to_save = []
            
            if inter.geom_type == "Polygon":
                if valid_polygon(inter):
                    polygons_to_save.append(inter)
                    
            elif inter.geom_type == "MultiPolygon":
                # Extract all valid polygons from MultiPolygon
                for geom in inter.geoms:
                    if geom.geom_type == "Polygon" and valid_polygon(geom):
                        polygons_to_save.append(geom)
                        
            elif inter.geom_type == "GeometryCollection":
                # Extract polygons from GeometryCollection, ignore other geometry types
                for geom in inter.geoms:
                    if geom.geom_type == "Polygon" and valid_polygon(geom):
                        polygons_to_save.append(geom)
                    elif geom.geom_type == "MultiPolygon":
                        for sub_geom in geom.geoms:
                            if sub_geom.geom_type == "Polygon" and valid_polygon(sub_geom):
                                polygons_to_save.append(sub_geom)
            
            # Convert valid polygons to LabelMe format with scaling
            for poly in polygons_to_save:
                try:
                    # Get exterior coordinates (main polygon boundary)
                    coords = list(poly.exterior.coords)
                    
                    # Remove duplicate last point if exists
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    
                    # Need at least 3 points for a valid polygon
                    if len(coords) < 3:
                        continue
                        
                    # Check if polygon area is meaningful before scaling
                    if poly.area < 1.0:  # Skip very small polygons
                        continue
                    
                    # Scale coordinates to match resized quadrant
                    scaled_coords = [[float(vx * sx), float(vy * sy)] for vx, vy in coords]
                    
                    clipped_shape = shape.copy()
                    clipped_shape["points"] = scaled_coords
                    new_shapes.append(clipped_shape)
                    
                except Exception as e:
                    print(Fore.YELLOW + f"Warning: Failed to process polygon in {base_name}_{suffix}: {e}" + Fore.RESET)
                    continue

        new_shapes = [s for s in new_shapes if s.get("points") and len(s["points"]) >= 3]
        if not new_shapes:
            print(Fore.YELLOW + f"Skipping {base_name}_{suffix}: no annotations" + Fore.RESET)
            continue

        resized = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{base_name}_{suffix}.jpg"), resized)
        with open(out_dir / f"{base_name}_{suffix}.json", "w") as f:
            json.dump({
                "version": anns.get("version", "5.0.1"),
                "flags": anns.get("flags", {}),
                "shapes": new_shapes,
                "imagePath": f"{base_name}_{suffix}.jpg",
                "imageHeight": tile_h,
                "imageWidth": tile_w
            }, f, indent=2)
        saved += 1
    return saved


def save_resized_with_annotations(img, anns, out_path, base_name, suffix, out_w, out_h):
    """
        @brief Resize image and scale annotations accordingly.
        @details Resize the entire image to (out_w, out_h) and scale all annotations.
        @return The number of saved images (1 if successful, 0 otherwise).
    """
    H, W = img.shape[:2]
    sx, sy = out_w / float(W), out_h / float(H)

    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    new_shapes = []
    
    for shape in anns.get("shapes", []):
        pts = np.array(shape.get("points", []), dtype=np.float32)
        
        if pts.size < 6:  # Need at least 3 points (6 coordinates)
            continue

        # Scale the points directly without creating intermediate polygon
        scaled_pts = [(float(px * sx), float(py * sy)) for px, py in pts]
        
        # Remove duplicate last point if exists
        if len(scaled_pts) > 1 and scaled_pts[0] == scaled_pts[-1]:
            scaled_pts = scaled_pts[:-1]

        if len(scaled_pts) < 3:
            continue

        # Only validate if we can create a proper polygon
        try:
            test_poly = Polygon(scaled_pts)
            
            # Repair if invalid
            if not test_poly.is_valid:
                test_poly = test_poly.buffer(0)
                if not test_poly.is_valid or test_poly.is_empty:
                    print(Fore.YELLOW + f"Warning: Skipping resized full image due to invalid polygon in {base_name}" + Fore.RESET)
                    continue
            
            # Use a much smaller area threshold for scaled images
            if test_poly.area < 0.1:  # Reduced from 1.0 to 0.1
                continue
                
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Failed to create polygon in resized full image {base_name}: {e}" + Fore.RESET)
            continue

        clipped_shape = shape.copy()
        clipped_shape["points"] = scaled_pts
        new_shapes.append(clipped_shape)

    if not new_shapes:
        print(Fore.YELLOW + f"Skipping {base_name}_{suffix}: no valid annotations after scaling" + Fore.RESET)
        return 0

    cv2.imwrite(str(out_path / f"{base_name}_{suffix}.jpg"), resized)
    with open(out_path / f"{base_name}_{suffix}.json", "w") as f:
        json.dump({
            "version": anns.get("version", "5.0.1"),
            "flags": anns.get("flags", {}),
            "shapes": new_shapes,
            "imagePath": f"{base_name}_{suffix}.jpg",
            "imageHeight": out_h,
            "imageWidth": out_w
        }, f, indent=2)
    return 1

def process_dataset(input_dir, output_dir, tile_w, tile_h, overlap, zoom_out=False, pad_border=False):
    """Process entire LabelMe dataset by tiling all images and annotations."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    # Check if input directory has .jpg images
    processed_count = 0
    images = list(input_dir.glob("*.jpg"))
    if not images:
        print(Fore.RED + f"No .jpg images found in input directory: {input_dir}" + Fore.RESET)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tiling_log.txt"
    with open(log_path, "w") as logf:
        for img_file in images:
            base_name = img_file.stem
            json_file = input_dir / f"{base_name}.json"
            if not json_file.exists():
                print(Fore.YELLOW + f"Skipping {img_file.name}: no corresponding JSON file found" + Fore.RESET)
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                print(Fore.RED + f"Failed to load image: {img_file}" + Fore.RESET)
                continue
            with open(json_file) as f:
                anns = json.load(f)
            print(f"Processing {img_file.name} ({img.shape[1]}x{img.shape[0]})")
            # n_rows, n_cols = generate_tiles(img, anns, output_dir, base_name, tile_w, tile_h, overlap, pad_border)
            # logf.write(f"{base_name}: grid {n_rows}x{n_cols}, tile {tile_w}x{tile_h}, orig {img.shape[1]}x{img.shape[0]}\n")
            if zoom_out:
                n_saved = generate_scaled_quadrants(img, anns, output_dir, base_name, tile_w, tile_h)
                n_saved += save_resized_with_annotations(img, anns, output_dir, base_name, "scaled_full", tile_w, tile_h)
            processed_count += 1
            # print(f"  Grid for {base_name}: {n_rows}x{n_cols}")
    print(f"Successfully processed {processed_count} images. Log written to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile LabelMe dataset with overlapping regions")
    parser.add_argument('--input-dir', type=str, required=True, 
                       help='Path to input LabelMe dataset folder')
    parser.add_argument('--output-dir', type=str, default='dataset_tiled',
                       help='Path to output folder for tiled dataset (default: dataset_tiled)')
    parser.add_argument('--width', type=int, default=848,
                       help=f'Width of each tile in pixels (default: 848)')
    parser.add_argument('--height', type=int, default=480,
                       help=f'Height of each tile in pixels (default: 480)')
    parser.add_argument('--overlap', type=float, default=0.0,
                       help=f'Overlap percentage between tiles (0.0-1.0, default: 0.0)')
    parser.add_argument('--pad-border', action='store_true',
                        help='If set, pad border tiles instead of overlapping them')
    parser.add_argument('--zoom-out', action='store_true',
                        help='If set, create zoomed-out versions of images')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(Fore.RED + f"Input directory does not exist: {args.input_dir}" + Fore.RESET)
        exit(1)

    if args.width <= 0 or args.height <= 0:
        print(Fore.RED + "Tile width and height must be positive integers" + Fore.RESET)
        exit(1)
        
    if not (0.0 <= args.overlap < 1.0):
        print(Fore.RED + "Overlap must be between 0.0 and 1.0 (exclusive)" + Fore.RESET)
        exit(1)
    
    print(f"   Configuration:")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Tile size: {args.width}x{args.height}")
    print(f"   Overlap: {args.overlap:.1%}")
    print(f"   Zoom out: {args.zoom_out}")
    print(f"   Pad border: {args.pad_border}")
    print()

    process_dataset(args.input_dir, args.output_dir, args.width, args.height, args.overlap, args.zoom_out, args.pad_border)
