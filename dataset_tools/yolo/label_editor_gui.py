#!/usr/bin/env python3
"""
YOLO Dataset Label Editor - Interactive GUI Tool.

Interactive Tkinter-based tool for fixing YOLO segmentation dataset labels.
Allows quick class changes and annotation deletions with visual feedback
using color-coded polygons. Supports keyboard navigation and manual save.

Author: Alessio Lovato

Controls:
- Left Click on mask: Change class (cycles through classes)
- Right Click on mask: Delete annotation
- Enter or 'n': Next image
- Backspace or 'p': Previous image
- 's': Save current annotations
- 'q': Quit

Arguments:
    --dataset: Path to YOLO dataset.yaml file
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import yaml
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil


# Define colors for up to 10 classes (RGB format for PIL)
CLASS_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 128),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
]


class Annotation:
    """Represents a single annotation (polygon)."""
    
    def __init__(self, class_id: int, points: List[Tuple[float, float]]):
        self.class_id = class_id
        self.points = points  # Normalized coordinates [0-1]
        
    def to_absolute(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        abs_points = []
        for x, y in self.points:
            abs_points.append((int(x * img_width), int(y * img_height)))
        return abs_points
    
    def contains_point(self, x: int, y: int, img_width: int, img_height: int) -> bool:
        """Check if a point is inside this annotation's polygon."""
        abs_points = self.to_absolute(img_width, img_height)
        
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


class YOLOLabelEditorGUI:
    """Main editor GUI class for YOLO dataset."""
    
    def __init__(self, root, dataset_yaml_path: str):
        self.root = root
        self.root.title("YOLO Label Editor")
        self.root.geometry("1600x900")
        
        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.dataset_dir = self.dataset_yaml_path.parent
        
        # Load dataset configuration
        with open(dataset_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
        self.num_classes = len(self.class_names)
        
        # Load all images from train, val, and test sets
        self.images = []
        self._load_images()
        
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
        
        # Undo history (store up to 10 states)
        self.undo_history = []
        self.max_undo = 10
        
        # Setup GUI
        self._setup_gui()
        
        # Load first image
        if self.images:
            self.load_image(0)
        
        # Key bindings
        self.root.bind('<Return>', lambda e: self.next_image())
        self.root.bind('n', lambda e: self.next_image())
        self.root.bind('<BackSpace>', lambda e: self.prev_image())
        self.root.bind('p', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('s', lambda e: self.save_annotations())
        self.root.bind('z', lambda e: self.undo())
        self.root.bind('<Delete>', lambda e: self.delete_current_image())
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        
        print(f"Loaded {len(self.images)} images from dataset")
        print(f"Classes: {self.class_names}")
        
    def _load_images(self):
        """Load all image paths from train, val, and test sets."""
        for split in ['train', 'val', 'test']:
            split_path = self.config.get(split)
            if split_path and split_path.strip():
                images_dir = self.dataset_dir / split_path
                if images_dir.exists():
                    # Get all image files
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        for img_path in images_dir.glob(ext):
                            # Determine corresponding label path
                            label_dir = self.dataset_dir / 'labels' / split
                            label_path = label_dir / (img_path.stem + '.txt')
                            self.images.append({
                                'image': img_path,
                                'label': label_path,
                                'split': split
                            })
                    found = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
                    print(f"Found {found} images in {split} set")
    
    def _setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Info labels
        self.info_label = ttk.Label(control_frame, text="", font=('Arial', 10, 'bold'))
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="Previous (P)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Next (N)", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save (S)", command=self.save_annotations).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Quit (Q)", command=self.quit_app).pack(side=tk.LEFT, padx=2)
        
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
        annotated_container = ttk.LabelFrame(image_frame, text="Annotated Image (Click to Edit)")
        annotated_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.annotated_canvas = tk.Canvas(annotated_container, bg='black')
        self.annotated_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events to annotated canvas
        self.annotated_canvas.bind('<Motion>', self.on_mouse_move)
        self.annotated_canvas.bind('<Button-1>', self.on_left_click)
        self.annotated_canvas.bind('<Button-3>', self.on_right_click)
        
        # Legend and info panel
        info_panel = ttk.Frame(main_frame)
        info_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Class legend
        legend_frame = ttk.LabelFrame(info_panel, text="Class Legend")
        legend_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.legend_text = tk.Text(legend_frame, height=4, width=40, state='disabled')
        self.legend_text.pack(fill=tk.BOTH, expand=True)
        
        # Log panel (middle)
        log_frame = ttk.LabelFrame(info_panel, text="Activity Log")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.log_text = tk.Text(log_frame, height=4, width=50, state='disabled', wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(info_panel, text="Instructions")
        instructions_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        instructions = """
Left Click: Change class | Right Click: Delete annotation
Arrows/Enter/N: Next | Backspace/P: Previous
S: Save | Z: Undo | Delete: Delete image | Q/ESC: Quit
        """
        instr_label = ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT)
        instr_label.pack(padx=5, pady=5)
        
        # Update legend
        self._update_legend()
    
    def _update_legend(self):
        """Update the class legend display."""
        self.legend_text.config(state='normal')
        self.legend_text.delete(1.0, tk.END)
        
        line_num = 1
        for class_id, class_name in self.class_names.items():
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            
            self.legend_text.insert(tk.END, f"  {class_name}\n")
            self.legend_text.tag_add(f"class_{class_id}", f"{line_num}.0", f"{line_num}.end")
            self.legend_text.tag_config(f"class_{class_id}", background=color_hex, foreground="white")
            line_num += 1
        
        self.legend_text.config(state='disabled')
    
    def log_message(self, message: str):
        """Add a message to the log panel."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.config(state='disabled')
        print(message)  # Also print to console
    
    def save_undo_state(self):
        """Save current state to undo history."""
        # Create a deep copy of current annotations
        import copy
        state = copy.deepcopy(self.current_annotations)
        self.undo_history.append(state)
        
        # Keep only last 10 states
        if len(self.undo_history) > self.max_undo:
            self.undo_history.pop(0)
    
    def undo(self):
        """Undo last action."""
        if not self.undo_history:
            self.log_message("No actions to undo")
            return
        
        # Restore previous state
        self.current_annotations = self.undo_history.pop()
        self.modified = True
        self._update_display()
        self._update_info()
        self.log_message("Undone last action")
    
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
                for line in f:
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
                        self.current_annotations.append(Annotation(class_id, points))
        
        self.modified = False
        self.undo_history = []  # Clear undo history when loading new image
        self._update_display()
        self._update_info()
        self.log_message(f"Loaded image: {self.image_path.name} ({len(self.current_annotations)} annotations)")
        
        return True
    
    def _update_display(self):
        """Update both canvas displays."""
        if self.current_image_pil is None:
            return
        
        # Calculate scale to fit canvas
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 600
            canvas_height = 600
        
        scale_w = canvas_width / self.img_width
        scale_h = canvas_height / self.img_height
        self.display_scale = min(scale_w, scale_h)  # Scale to fit canvas (allow upscaling)
        
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
    
    def _draw_annotations(self) -> Image.Image:
        """Draw all annotations on the image."""
        annotated = self.current_image_pil.copy()
        draw = ImageDraw.Draw(annotated, 'RGBA')
        
        for idx, ann in enumerate(self.current_annotations):
            color = CLASS_COLORS[ann.class_id % len(CLASS_COLORS)]
            abs_points = ann.to_absolute(self.img_width, self.img_height)
            
            # Draw filled polygon with transparency
            alpha = 80 if idx != self.hovered_annotation_idx else 120
            fill_color = color + (alpha,)
            draw.polygon(abs_points, fill=fill_color, outline=color + (255,))
            
            # Draw class label at centroid
            if len(abs_points) > 0:
                centroid_x = sum(p[0] for p in abs_points) // len(abs_points)
                centroid_y = sum(p[1] for p in abs_points) // len(abs_points)
                
                class_name = self.class_names.get(ann.class_id, f"Class {ann.class_id}")
                
                # Draw text with background
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((centroid_x, centroid_y), class_name, font=font)
                draw.rectangle(bbox, fill=(255, 255, 255, 200))
                draw.text((centroid_x, centroid_y), class_name, fill=color + (255,), font=font)
        
        return annotated
    
    def _update_info(self):
        """Update info labels."""
        info_text = f"Image {self.current_idx + 1}/{len(self.images)} | {self.image_path.name} | Annotations: {len(self.current_annotations)}"
        self.info_label.config(text=info_text)
        
        if self.modified:
            self.status_label.config(text="[MODIFIED]", foreground="red")
        else:
            self.status_label.config(text="[SAVED]", foreground="green")
    
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
        for idx in range(len(self.current_annotations) - 1, -1, -1):
            ann = self.current_annotations[idx]
            if ann.contains_point(x, y, self.img_width, self.img_height):
                return idx
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
        """Handle left click - change class."""
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if 0 <= img_x < self.img_width and 0 <= img_y < self.img_height:
            ann_idx = self.find_annotation_at_point(img_x, img_y)
            if ann_idx is not None:
                self.cycle_annotation_class(ann_idx)
    
    def on_right_click(self, event):
        """Handle right click - delete annotation."""
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if 0 <= img_x < self.img_width and 0 <= img_y < self.img_height:
            ann_idx = self.find_annotation_at_point(img_x, img_y)
            if ann_idx is not None:
                self.delete_annotation(ann_idx)
    
    def cycle_annotation_class(self, ann_idx: int):
        """Cycle to next class for given annotation."""
        if 0 <= ann_idx < len(self.current_annotations):
            self.save_undo_state()  # Save state before modifying
            
            ann = self.current_annotations[ann_idx]
            old_class = self.class_names.get(ann.class_id, f"Class {ann.class_id}")
            ann.class_id = (ann.class_id + 1) % self.num_classes
            new_class = self.class_names.get(ann.class_id, ann.class_id)
            
            self.modified = True
            self.log_message(f"Changed annotation from '{old_class}' to '{new_class}'")
            self._update_display()
            self._update_info()
    
    def delete_annotation(self, ann_idx: int):
        """Delete annotation at given index."""
        if 0 <= ann_idx < len(self.current_annotations):
            self.save_undo_state()  # Save state before modifying
            
            ann = self.current_annotations[ann_idx]
            class_name = self.class_names.get(ann.class_id, f"Class {ann.class_id}")
            self.current_annotations.pop(ann_idx)
            
            self.modified = True
            self.log_message(f"Deleted '{class_name}' annotation")
            self._update_display()
            self._update_info()
    
    def save_annotations(self):
        """Save current annotations to label file."""
        if not self.modified:
            self.log_message("No changes to save")
            return
        
        # Create label directory if it doesn't exist
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write annotations
        with open(self.label_path, 'w') as f:
            for ann in self.current_annotations:
                f.write(ann.to_yolo_string() + '\n')
        
        self.modified = False
        self._update_info()
        self.log_message(f"Saved annotations to {self.label_path.name}")
    
    def delete_current_image(self):
        """Delete current image and label from dataset."""
        if not self.image_path or not self.image_path.exists():
            self.log_message("No image to delete")
            return
        
        # Create a backup directory
        backup_dir = self.dataset_dir / 'deleted_images'
        backup_dir.mkdir(exist_ok=True)
        
        # Move files to backup
        img_name = self.image_path.name
        img_backup = backup_dir / img_name
        shutil.move(str(self.image_path), str(img_backup))
        
        if self.label_path.exists():
            lbl_backup = backup_dir / self.label_path.name
            shutil.move(str(self.label_path), str(lbl_backup))
        
        self.log_message(f"Deleted {img_name} (moved to deleted_images/)")
        
        # Remove from list
        self.images.pop(self.current_idx)
        
        # Load next image (or previous if at end)
        if self.current_idx >= len(self.images):
            self.current_idx = len(self.images) - 1
        
        if len(self.images) > 0:
            self.load_image(self.current_idx)
        else:
            self.log_message("No more images in dataset!")
            messagebox.showinfo("Done", "All images processed!")
            self.quit_app()
    
    def next_image(self):
        """Load next image."""
        if self.modified:
            response = messagebox.askyesnocancel("Save?", "Save changes before moving to next image?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_annotations()
        
        if self.current_idx < len(self.images) - 1:
            self.load_image(self.current_idx + 1)
        else:
            self.log_message("Reached last image")
    
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
            self.log_message("Already at first image")
    
    def quit_app(self):
        """Quit the application."""
        if self.modified:
            response = messagebox.askyesnocancel("Save?", "Save changes before quitting?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_annotations()
        
        self.root.quit()
        self.root.destroy()


def main():
    import sys
    
    if len(sys.argv) < 2:
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
    else:
        dataset_yaml = sys.argv[1]
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset YAML file not found: {dataset_yaml}")
        sys.exit(1)
    
    root = tk.Tk()
    app = YOLOLabelEditorGUI(root, dataset_yaml)
    root.mainloop()


if __name__ == "__main__":
    main()
