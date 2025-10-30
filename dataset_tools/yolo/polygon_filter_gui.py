#!/usr/bin/env python3
"""
YOLO Dataset Small Polygon Filter - GUI Version
Interactive tool for removing annotations with too few points.

Controls:
- Left Click: Toggle annotation delete state
- Right Arrow / Enter: Save and move to next image
- Left Arrow: Previous image
- Spacebar: Unmark all annotations (keep all)
- Z: Undo last 10 actions
- S: Save current changes
- Q / ESC: Quit
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import yaml
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import copy


class Annotation:
    """Represents a single annotation (polygon)."""
    
    def __init__(self, class_id: int, points: List[Tuple[float, float]], index: int):
        self.class_id = class_id
        self.points = points  # Normalized coordinates [0-1]
        self.index = index
        self.marked_for_deletion = False
        
    def num_points(self) -> int:
        """Get number of points in polygon."""
        return len(self.points)
        
    def to_absolute(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        abs_points = []
        for x, y in self.points:
            abs_points.append((int(x * img_width), int(y * img_height)))
        return abs_points
    
    def contains_point(self, x: int, y: int, img_width: int, img_height: int) -> bool:
        """Check if a point is inside this annotation's polygon."""
        abs_points = self.to_absolute(img_width, img_height)
        
        if len(abs_points) < 3:
            return False
        
        # Ray casting algorithm
        n = len(abs_points)
        inside = False
        p1x, p1y = abs_points[0]
        for i in range(1, n + 1):
            p2x, p2y = abs_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def to_yolo_string(self) -> str:
        """Convert annotation to YOLO format string."""
        coords = []
        for x, y in self.points:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        return f"{self.class_id} " + " ".join(coords)


class PolygonFilterGUI:
    """Main GUI class for filtering small polygons."""
    
    def __init__(self, root, dataset_yaml_path: str, min_points: int = 15):
        self.root = root
        self.root.title(f"YOLO Polygon Filter - Auto-mark < {min_points} Points")
        self.root.geometry("1800x900")
        
        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.dataset_dir = self.dataset_yaml_path.parent
        self.min_points = min_points
        
        # Load dataset configuration
        with open(dataset_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config.get('names', {})
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}
        
        # Load all images from train, val, and test sets
        self.images = []
        self._load_images()
        
        # Use all images (not just those with small polygons)
        # Small polygons will be auto-marked for deletion
        
        self.current_idx = 0
        self.current_image_pil = None
        self.current_annotations = []
        self.image_path = None
        self.label_path = None
        self.img_height = 0
        self.img_width = 0
        self.modified = False
        
        # Display properties
        self.display_scale = 1.0
        self.hovered_annotation_idx = None
        
        # Selection rectangle for area selection
        self.selection_active = False
        self.selection_start = None
        self.selection_rect = None
        
        # Undo history (store up to 10 states)
        self.undo_history = []
        self.max_undo = 10
        
        # Statistics
        self.total_deleted = 0
        self.total_kept = 0
        
        # Setup GUI
        self._setup_gui()
        
        # Load first image
        if self.images:
            self.load_image(0)
        else:
            messagebox.showinfo("No Images", "No images found in dataset!")
            self.root.quit()
            return
        
        # Key bindings
        self.root.bind('<Return>', lambda e: self.save_and_next())
        self.root.bind('<Right>', lambda e: self.save_and_next())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<space>', lambda e: self.unmark_all())
        self.root.bind('s', lambda e: self.save_annotations())
        self.root.bind('z', lambda e: self.undo())
        self.root.bind('<Delete>', lambda e: self.delete_current_image())
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        
        print(f"Loaded {len(self.images)} total images")
        print(f"Annotations with < {min_points} points will be auto-marked for deletion")
        print(f"Classes: {self.class_names}")
        
    def _load_images(self):
        """Load all image paths from train, val, and test sets."""
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / 'images' / split
            labels_dir = self.dataset_dir / 'labels' / split
            
            if images_dir.exists() and labels_dir.exists():
                # Get all image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in images_dir.glob(ext):
                        # Determine corresponding label path
                        label_path = labels_dir / (img_path.stem + '.txt')
                        if label_path.exists():
                            self.images.append({
                                'image': img_path,
                                'label': label_path,
                                'split': split
                            })
    
    def _setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Info labels
        self.info_label = ttk.Label(control_frame, text="", font=('Arial', 11, 'bold'))
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="← Previous", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save & Next →", command=self.save_and_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Unmark All (Space)", command=self.unmark_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save (S)", command=self.save_annotations).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Undo (Z)", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Quit (Q)", command=self.quit_app).pack(side=tk.LEFT, padx=2)
        
        # Statistics
        self.stats_label = ttk.Label(control_frame, text="", foreground="blue", font=('Arial', 10))
        self.stats_label.pack(side=tk.RIGHT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="", foreground="green")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Original image canvas
        original_container = ttk.LabelFrame(image_frame, text="Original Image")
        original_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.original_canvas = tk.Canvas(original_container, bg='black')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Annotated image canvas
        annotated_container = ttk.LabelFrame(image_frame, text="Annotations (Click to Toggle | Red=Delete, Green=Keep)")
        annotated_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.annotated_canvas = tk.Canvas(annotated_container, bg='black')
        self.annotated_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events to annotated canvas
        self.annotated_canvas.bind('<Motion>', self.on_mouse_move)
        self.annotated_canvas.bind('<Button-1>', self.on_left_click)
        self.annotated_canvas.bind('<Button-3>', self.on_right_click_start)
        self.annotated_canvas.bind('<B3-Motion>', self.on_right_drag)
        self.annotated_canvas.bind('<ButtonRelease-3>', self.on_right_click_end)
        
        # Bottom info panel
        info_panel = ttk.Frame(main_frame)
        info_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Annotation details
        details_frame = ttk.LabelFrame(info_panel, text="Current Image Annotations")
        details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.details_text = tk.Text(details_frame, height=6, width=50, state='disabled')
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(info_panel, text="Controls")
        instructions_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=2)
        
        instructions = """
Left Click: Toggle annotation delete state
Right Click & Drag: Select area to toggle multiple
Enter/→: SAVE & move to next (deletions happen here!)
←: Previous image
Space: Unmark all (keep all)
S: SAVE changes (deletions happen here!)
Z: Undo
Delete: Remove image & label from dataset
Q/ESC: Quit

NOTE: Annotations are PERMANENTLY DELETED only when
you press S (Save) or Enter (Save & Next).
Red = Will be deleted | Green = Will be kept
Auto-marks annotations with < {} points as red.
        """.format(self.min_points)
        instr_label = ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT, font=('Arial', 8))
        instr_label.pack(padx=5, pady=5)
    
    def save_undo_state(self):
        """Save current state to undo history."""
        # Create a deep copy of current annotations
        state = copy.deepcopy(self.current_annotations)
        self.undo_history.append(state)
        
        # Keep only last 10 states
        if len(self.undo_history) > self.max_undo:
            self.undo_history.pop(0)
    
    def undo(self):
        """Undo last action."""
        if not self.undo_history:
            self.status_label.config(text="Nothing to undo", foreground="orange")
            return
        
        # Restore previous state
        self.current_annotations = self.undo_history.pop()
        self.modified = True
        self._update_display()
        self._update_info()
        self.status_label.config(text="Undone", foreground="blue")
    
    def load_image(self, idx: int):
        """Load image and its annotations."""
        if idx < 0 or idx >= len(self.images):
            return False
        
        self.current_idx = idx
        img_info = self.images[idx]
        self.image_path = img_info['image']
        self.label_path = img_info['label']
        
        # Load image
        try:
            self.current_image_pil = Image.open(str(self.image_path))
            if self.current_image_pil.mode != 'RGB':
                self.current_image_pil = self.current_image_pil.convert('RGB')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
            return False
        
        self.img_width, self.img_height = self.current_image_pil.size
        
        # Load annotations
        self.current_annotations = []
        if self.label_path.exists():
            with open(self.label_path, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Parse coordinate pairs
                    points = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            points.append((coords[i], coords[i + 1]))
                    
                    if points:
                        ann = Annotation(class_id, points, idx)
                        # Automatically mark for deletion if too few points
                        if ann.num_points() < self.min_points:
                            ann.marked_for_deletion = True
                        self.current_annotations.append(ann)
        
        self.modified = False
        self.undo_history = []  # Clear undo history when loading new image
        self._update_display()
        self._update_info()
        
        return True
    
    def _update_display(self):
        """Update both canvas displays."""
        if self.current_image_pil is None:
            return
        
        # Calculate scale to fit canvas
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 800
            canvas_height = 800
        
        scale_w = canvas_width / self.img_width
        scale_h = canvas_height / self.img_height
        self.display_scale = min(scale_w, scale_h)  # Scale to fill available space
        
        display_width = int(self.img_width * self.display_scale)
        display_height = int(self.img_height * self.display_scale)
        
        # Display original image
        original_resized = self.current_image_pil.resize((display_width, display_height), Image.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(original_resized)
        
        self.original_canvas.delete('all')
        self.original_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.original_photo, anchor=tk.CENTER
        )
        
        # Display annotated image
        annotated_img = self._draw_annotations()
        annotated_resized = annotated_img.resize((display_width, display_height), Image.LANCZOS)
        self.annotated_photo = ImageTk.PhotoImage(annotated_resized)
        
        self.annotated_canvas.delete('all')
        self.annotated_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.annotated_photo, anchor=tk.CENTER
        )
        
        # Draw selection rectangle if active
        if self.selection_active and self.selection_start and self.selection_rect:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_rect
            self.annotated_canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='yellow', width=3, dash=(5, 5)
            )
    
    def _draw_annotations(self) -> Image.Image:
        """Draw all annotations on the image."""
        annotated = self.current_image_pil.copy()
        draw = ImageDraw.Draw(annotated, 'RGBA')
        
        for ann in self.current_annotations:
            abs_points = ann.to_absolute(self.img_width, self.img_height)
            
            if len(abs_points) < 3:
                continue
            
            # Color based on deletion state
            if ann.marked_for_deletion:
                color = (255, 0, 0)  # Red for delete
                alpha = 100
                outline_width = 3
            else:
                color = (0, 255, 0)  # Green for keep
                alpha = 80
                outline_width = 2
            
            # Highlight if hovered
            if ann.index == self.hovered_annotation_idx:
                alpha = 150
                outline_width = 4
            
            # Draw filled polygon with transparency
            fill_color = color + (alpha,)
            outline_color = color + (255,)
            draw.polygon(abs_points, fill=fill_color, outline=outline_color, width=outline_width)
        
        return annotated
    
    def _update_info(self):
        """Update info labels and details."""
        # Title info
        to_delete = sum(1 for ann in self.current_annotations if ann.marked_for_deletion)
        to_keep = len(self.current_annotations) - to_delete
        
        info_text = f"Image {self.current_idx + 1}/{len(self.images)} | {self.image_path.name}"
        self.info_label.config(text=info_text)
        
        # Statistics
        stats_text = f"Total Deleted: {self.total_deleted} | Total Kept: {self.total_kept}"
        self.stats_label.config(text=stats_text)
        
        # Status
        if self.modified:
            self.status_label.config(text="[MODIFIED]", foreground="red")
        else:
            self.status_label.config(text="[SAVED]", foreground="green")
        
        # Details panel
        self.details_text.config(state='normal')
        self.details_text.delete(1.0, tk.END)
        
        self.details_text.insert(tk.END, f"Total Annotations: {len(self.current_annotations)}\n")
        self.details_text.insert(tk.END, f"Marked for Deletion: {to_delete} (red)\n")
        self.details_text.insert(tk.END, f"To Keep: {to_keep} (green)\n\n")
        
        # List annotations
        for ann in self.current_annotations:
            class_name = self.class_names.get(ann.class_id, f"Class {ann.class_id}")
            status = "DELETE" if ann.marked_for_deletion else "KEEP"
            color = "red" if ann.marked_for_deletion else "green"
            
            line_start = self.details_text.index(tk.END)
            self.details_text.insert(tk.END, f"  [{status}] {class_name}: {ann.num_points()} points\n")
            line_end = self.details_text.index(tk.END)
            
            tag_name = f"ann_{ann.index}"
            self.details_text.tag_add(tag_name, line_start, line_end)
            self.details_text.tag_config(tag_name, foreground=color)
        
        self.details_text.config(state='disabled')
    
    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """Convert canvas coordinates to image coordinates."""
        canvas_width = self.annotated_canvas.winfo_width()
        canvas_height = self.annotated_canvas.winfo_height()
        
        display_width = int(self.img_width * self.display_scale)
        display_height = int(self.img_height * self.display_scale)
        
        # Calculate offset (image is centered)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # Convert to image coordinates
        img_x = int((canvas_x - offset_x) / self.display_scale)
        img_y = int((canvas_y - offset_y) / self.display_scale)
        
        return img_x, img_y
    
    def find_annotation_at_point(self, x: int, y: int) -> Optional[int]:
        """Find annotation index at given point."""
        # Search in reverse order so top annotations are selected first
        for ann in reversed(self.current_annotations):
            if ann.contains_point(x, y, self.img_width, self.img_height):
                return ann.index
        return None
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Check if within image bounds
        if 0 <= img_x < self.img_width and 0 <= img_y < self.img_height:
            old_hover = self.hovered_annotation_idx
            self.hovered_annotation_idx = self.find_annotation_at_point(img_x, img_y)
            
            # Redraw if hover changed
            if old_hover != self.hovered_annotation_idx:
                self._update_display()
    
    def on_left_click(self, event):
        """Handle left click - toggle deletion state."""
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if 0 <= img_x < self.img_width and 0 <= img_y < self.img_height:
            ann_idx = self.find_annotation_at_point(img_x, img_y)
            if ann_idx is not None:
                self.toggle_annotation_deletion(ann_idx)
    
    def on_right_click_start(self, event):
        """Handle right click start - begin selection rectangle."""
        self.selection_active = True
        self.selection_start = (event.x, event.y)
        self.selection_rect = (event.x, event.y)
    
    def on_right_drag(self, event):
        """Handle right click drag - update selection rectangle."""
        if self.selection_active:
            self.selection_rect = (event.x, event.y)
            self._update_display()
    
    def on_right_click_end(self, event):
        """Handle right click release - toggle all annotations in selection."""
        if not self.selection_active:
            return
        
        self.selection_rect = (event.x, event.y)
        
        # Get selection bounds in image coordinates
        x1_canvas, y1_canvas = self.selection_start
        x2_canvas, y2_canvas = self.selection_rect
        
        # Normalize coordinates (handle drag in any direction)
        min_x_canvas = min(x1_canvas, x2_canvas)
        max_x_canvas = max(x1_canvas, x2_canvas)
        min_y_canvas = min(y1_canvas, y2_canvas)
        max_y_canvas = max(y1_canvas, y2_canvas)
        
        # Convert to image coordinates
        min_x_img, min_y_img = self.canvas_to_image_coords(min_x_canvas, min_y_canvas)
        max_x_img, max_y_img = self.canvas_to_image_coords(max_x_canvas, max_y_canvas)
        
        # Find all annotations with centroids in the selection
        selected_annotations = []
        for ann in self.current_annotations:
            abs_points = ann.to_absolute(self.img_width, self.img_height)
            if len(abs_points) > 0:
                # Calculate centroid
                centroid_x = sum(p[0] for p in abs_points) / len(abs_points)
                centroid_y = sum(p[1] for p in abs_points) / len(abs_points)
                
                # Check if centroid is in selection rectangle
                if (min_x_img <= centroid_x <= max_x_img and 
                    min_y_img <= centroid_y <= max_y_img):
                    selected_annotations.append(ann.index)
        
        # Toggle all selected annotations
        if selected_annotations:
            self.save_undo_state()
            for ann_idx in selected_annotations:
                for ann in self.current_annotations:
                    if ann.index == ann_idx:
                        ann.marked_for_deletion = not ann.marked_for_deletion
            
            self.modified = True
            self.status_label.config(
                text=f"Toggled {len(selected_annotations)} annotations in selection",
                foreground="blue"
            )
        
        # Clear selection
        self.selection_active = False
        self.selection_start = None
        self.selection_rect = None
        self._update_display()
        self._update_info()
    
    def toggle_annotation_deletion(self, ann_idx: int):
        """Toggle deletion state for annotation."""
        for ann in self.current_annotations:
            if ann.index == ann_idx:
                self.save_undo_state()
                ann.marked_for_deletion = not ann.marked_for_deletion
                self.modified = True
                self._update_display()
                self._update_info()
                
                state = "DELETE" if ann.marked_for_deletion else "KEEP"
                class_name = self.class_names.get(ann.class_id, f"Class {ann.class_id}")
                self.status_label.config(
                    text=f"Toggled {class_name} to {state}",
                    foreground="red" if ann.marked_for_deletion else "green"
                )
                break
    
    def unmark_all(self):
        """Unmark all annotations (keep all)."""
        self.save_undo_state()
        for ann in self.current_annotations:
            ann.marked_for_deletion = False
        self.modified = True
        self._update_display()
        self._update_info()
        self.status_label.config(text="All annotations unmarked (keeping all)", foreground="green")
    
    def delete_current_image(self):
        """Delete current image and label file from dataset."""
        if not self.image_path or not self.image_path.exists():
            self.status_label.config(text="No image to delete", foreground="orange")
            return
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Delete Image?",
            f"Permanently delete this image and label from dataset?\n\n"
            f"File: {self.image_path.name}\n"
            f"This cannot be undone!",
            icon='warning'
        )
        
        if not response:
            return
        
        # Delete files
        try:
            deleted_files = []
            
            if self.image_path.exists():
                os.remove(str(self.image_path))
                deleted_files.append(self.image_path.name)
            
            if self.label_path.exists():
                os.remove(str(self.label_path))
                deleted_files.append(self.label_path.name)
            
            self.status_label.config(
                text=f"✓ Deleted: {', '.join(deleted_files)}",
                foreground="red"
            )
            
            # Remove from list
            self.images.pop(self.current_idx)
            
            # Load next or previous image
            if len(self.images) == 0:
                messagebox.showinfo("Done", "No more images in dataset!")
                self.quit_app()
                return
            
            # Stay at same index (which now points to next image) or go back if at end
            if self.current_idx >= len(self.images):
                self.current_idx = len(self.images) - 1
            
            self.load_image(self.current_idx)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete files: {e}")
            self.status_label.config(text=f"Error deleting files", foreground="red")
    
    def save_annotations(self):
        """Save current annotations to label file (THIS IS WHEN DELETIONS ACTUALLY HAPPEN)."""
        if not self.modified:
            self.status_label.config(text="No changes to save", foreground="blue")
            return
        
        # Count deletions
        deleted_count = sum(1 for ann in self.current_annotations if ann.marked_for_deletion)
        kept_count = len(self.current_annotations) - deleted_count
        
        # Filter out deleted annotations - THIS IS THE ACTUAL DELETION STEP
        kept_annotations = [ann for ann in self.current_annotations if not ann.marked_for_deletion]
        
        # Update global statistics
        self.total_deleted += deleted_count
        self.total_kept += kept_count
        
        # Write to file - ANNOTATIONS NOT IN THIS LIST ARE PERMANENTLY DELETED
        with open(self.label_path, 'w') as f:
            for ann in kept_annotations:
                f.write(ann.to_yolo_string() + '\n')
        
        # Update current annotations to reflect saved state
        self.current_annotations = kept_annotations
        
        self.modified = False
        self._update_info()
        self.status_label.config(
            text=f"✓ SAVED: Permanently deleted {deleted_count}, kept {kept_count}",
            foreground="green"
        )
    
    def save_and_next(self):
        """Save current changes and move to next image."""
        self.save_annotations()
        self.next_image()
    
    def next_image(self):
        """Load next image."""
        if self.current_idx < len(self.images) - 1:
            self.load_image(self.current_idx + 1)
        else:
            self.status_label.config(text="Reached last image", foreground="blue")
            messagebox.showinfo("Complete", 
                f"Finished reviewing all images!\n\n"
                f"Total Deleted: {self.total_deleted}\n"
                f"Total Kept: {self.total_kept}")
    
    def prev_image(self):
        """Load previous image."""
        if self.modified:
            response = messagebox.askyesnocancel("Save?", "Save changes before moving to previous image?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_annotations()
        
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)
        else:
            self.status_label.config(text="Already at first image", foreground="blue")
    
    def quit_app(self):
        """Quit the application."""
        if self.modified:
            response = messagebox.askyesnocancel("Save?", "Save changes before quitting?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_annotations()
        
        print(f"\nFinal Statistics:")
        print(f"  Total Deleted: {self.total_deleted}")
        print(f"  Total Kept: {self.total_kept}")
        
        self.root.quit()
        self.root.destroy()


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter YOLO annotations by minimum number of points",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, help='Path to dataset.yaml')
    parser.add_argument('--min-points', type=int, default=15,
                       help='Minimum number of points required (default: 15)')
    
    args = parser.parse_args()
    
    dataset_yaml = args.dataset
    
    if not dataset_yaml:
        # Open file dialog to select dataset.yaml
        root = tk.Tk()
        root.withdraw()
        dataset_yaml = filedialog.askopenfilename(
            title="Select dataset.yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not dataset_yaml:
            print("No file selected. Exiting.")
            sys.exit(0)
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset YAML file not found: {dataset_yaml}")
        sys.exit(1)
    
    root = tk.Tk()
    app = PolygonFilterGUI(root, dataset_yaml, args.min_points)
    root.mainloop()


if __name__ == "__main__":
    main()
