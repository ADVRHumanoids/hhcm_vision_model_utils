#!/usr/bin/env python3
"""
Upload YOLO segmentation dataset to Labelbox.
Converts YOLO polygon annotations to Labelbox polygon annotations.
"""

import os
import sys
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    import labelbox as lb
    import labelbox.types as lb_types
    from PIL import Image
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Install with: pip install labelbox pillow")
    sys.exit(1)


def load_dataset_yaml(yaml_path: str) -> Tuple[Dict[int, str], str]:
    """Load class names and dataset path from dataset.yaml"""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {}), data.get("path", ".")


def read_yolo_polygons(label_path: str, img_width: int, img_height: int) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Read YOLO polygon format labels and convert to pixel coordinates.
    Returns list of (class_id, [(x1, y1), (x2, y2), ...])
    """
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixels
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_width
                y = coords[i + 1] * img_height
                points.append((x, y))
            
            polygons.append((cls_id, points))

    return polygons


def create_labelbox_annotation(polygons: List[Tuple[int, List[Tuple[float, float]]]], 
                                class_names: Dict[int, str],
                                ontology_index: Dict[str, str],
                                img_width: int,
                                img_height: int) -> List[lb_types.ObjectAnnotation]:
    """
    Convert YOLO polygons to Labelbox Mask annotations.
    Creates a separate mask annotation for each instance.
    
    Following Labelbox's format:
    - Each mask is a 2D numpy array (height x width) with 0/1 values
    - Color is always (1, 1, 1) to identify the mask region
    - Use MaskData.from_2D_arr() to create mask data
    """
    annotations = []
    
    if not polygons:
        return []
    
    # Identifying what values in the numpy array correspond to the mask annotation
    color = (1, 1, 1)
    
    # Create a separate mask for each polygon instance
    for cls_id, points in polygons:
        if len(points) < 3:
            continue
        
        # Create binary mask for this polygon
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=color)  # Fill with 1
        
        # Create mask annotation using Labelbox format
        mask_data = lb_types.MaskData.from_2D_arr(mask)
        mask_annotation = lb_types.ObjectAnnotation(
            name=class_names.get(cls_id, f"class_{cls_id}"),
            value=lb_types.Mask(mask=mask_data, color=color)
        )
        annotations.append(mask_annotation)

    return annotations


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except ImportError:
        # Fallback to cv2 if PIL not available
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        return w, h


def upload_dataset_to_labelbox(dataset_root: str, 
                                api_key: str,
                                project_name: str,
                                splits: List[str] = ['train', 'val', 'test'],
                                dataset_name: str = None,
                                create_ontology: bool = True):
    """
    Upload YOLO dataset to Labelbox with segmentation annotations.
    
    Args:
        dataset_root: Root directory containing dataset.yaml
        api_key: Labelbox API key
        project_name: Name for the Labelbox project
        splits: Which splits to upload (train, val, test)
        dataset_name: Name for Labelbox dataset (defaults to project_name)
        create_ontology: Whether to create ontology from dataset classes
    """
    
    # Initialize Labelbox client
    print("🔑 Connecting to Labelbox...")
    client = lb.Client(api_key=api_key)
    
    # Load dataset configuration
    yaml_path = os.path.join(dataset_root, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"❌ Error: dataset.yaml not found at {yaml_path}")
        return
    
    class_names, _ = load_dataset_yaml(yaml_path)
    print(f"📝 Found {len(class_names)} classes: {', '.join(class_names.values())}")
    
    # Create or get project
    print(f"\n📦 Creating/finding project: {project_name}")
    projects = client.get_projects(where=(lb.Project.name == project_name))
    projects_list = list(projects)
    
    if projects_list:
        project = projects_list[0]
        print(f"   ✓ Found existing project: {project.uid}")
    else:
        project = client.create_project(
            name=project_name,
            media_type=lb.MediaType.Image
        )
        print(f"   ✓ Created new project: {project.uid}")
    
    # Create ontology if needed
    if create_ontology:
        print(f"\n🏗️  Setting up ontology...")
        
        # Check if project already has an ontology
        existing_ontology = project.ontology()
        if existing_ontology:
            print(f"   ℹ️  Project already has ontology: {existing_ontology.name}")
            ontology = existing_ontology
        else:
            # Create ontology with polygon tools for each class
            tools = []
            for cls_id, cls_name in sorted(class_names.items()):
                tools.append(lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name=cls_name))
            
            ontology_builder = lb.OntologyBuilder(
                classifications=[],
                tools=tools
            )
            
            ontology = client.create_ontology(
                f"{project_name}_ontology",
                ontology_builder.asdict(),
                media_type=lb.MediaType.Image
            )
            
            # Connect ontology to project using connect_ontology method
            project.connect_ontology(ontology)
            print(f"   ✓ Created ontology with {len(class_names)} polygon tools")
    else:
        ontology = project.ontology()
        if not ontology:
            print("❌ Error: Project has no ontology. Set create_ontology=True")
            return
    
    # Build ontology index (class_name -> feature_schema_id)
    ontology_index = {}
    for tool in ontology.tools():
        ontology_index[tool.name] = tool.feature_schema_id
    
    print(f"   Ontology index: {ontology_index}")
    
    # Create dataset
    if dataset_name is None:
        dataset_name = f"{project_name}_dataset"
    
    print(f"\n📊 Creating/finding dataset: {dataset_name}")
    # Try to find existing dataset
    datasets = client.get_datasets(where=(lb.Dataset.name == dataset_name))
    datasets_list = list(datasets)
    
    if datasets_list:
        dataset = datasets_list[0]
        print(f"   ✓ Found existing dataset: {dataset.uid}")
    else:
        dataset = client.create_dataset(name=dataset_name)
        print(f"   ✓ Dataset created: {dataset.uid}")
    
    # Collect all images and prepare data rows and labels
    all_data_rows = []
    all_labels = []
    
    for split in splits:
        images_dir = os.path.join(dataset_root, 'images', split)
        labels_dir = os.path.join(dataset_root, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"\n⚠️  Split '{split}' not found at {images_dir}, skipping...")
            continue
        
        print(f"\n📁 Processing split: {split}")
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(images_dir).glob(ext))
        
        print(f"   Found {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"   Processing {split}"):
            base_name = img_path.stem
            label_path = os.path.join(labels_dir, base_name + '.txt')
            
            # Get image dimensions
            try:
                img_width, img_height = get_image_dimensions(str(img_path))
            except Exception as e:
                print(f"   ⚠️  Could not read {img_path}: {e}")
                continue
            
            # Read annotations
            polygons = read_yolo_polygons(label_path, img_width, img_height)
            
            # Create unique identifier
            global_key = f"{split}_{base_name}"
            
            # Create data row entry
            data_row_dict = {
                "row_data": str(img_path),
                "global_key": global_key,
                "external_id": global_key
            }
            all_data_rows.append(data_row_dict)
            
            # Create label with annotations if polygons exist
            if polygons:
                lb_annotations = create_labelbox_annotation(
                    polygons, class_names, ontology_index, img_width, img_height
                )
                
                if lb_annotations:
                    # Create Label with ImageData following Labelbox format
                    label = lb_types.Label(
                        data={"global_key": global_key},
                        annotations=lb_annotations
                    )
                    all_labels.append(label)
    
    # Upload data rows to dataset
    print(f"\n⬆️  Uploading {len(all_data_rows)} images to dataset...")
    
    try:
        task = dataset.create_data_rows(all_data_rows)
        print(f"   Waiting for upload to complete...")
        task.wait_till_done()
        
        print(f"   Task status: {task.status}")
        
        if task.status == "COMPLETE":
            print(f"   ✓ Successfully uploaded {len(all_data_rows)} images")
        elif task.status == "FAILED":
            print(f"   ❌ Upload failed!")
            # Try to get error details from various attributes
            if hasattr(task, 'errors') and task.errors:
                print(f"   Errors: {task.errors}")
            if hasattr(task, 'error_messages'):
                print(f"   Error messages: {task.error_messages}")
            if hasattr(task, 'result'):
                print(f"   Result: {task.result}")
            # Print all task attributes to debug
            print(f"   Task details: {vars(task)}")
            return
        else:
            print(f"   ⚠️  Upload finished with status: {task.status}")
            if hasattr(task, 'errors') and task.errors:
                print(f"   Errors: {task.errors}")
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create batch and add data rows to project
    print(f"\n🔗 Creating batch in project...")
    global_keys = [dr["global_key"] for dr in all_data_rows]
    
    try:
        batch = project.create_batch(
            f"{dataset_name}_batch",
            global_keys=global_keys,
            priority=1
        )
        print(f"   ✓ Batch created: {batch.uid}")
    except Exception as e:
        print(f"   ⚠️  Batch creation note: {e}")
        print(f"   Continuing with annotation upload...")
    
    # Upload annotations as pre-labels (Model-Assisted Labeling)
    if all_labels:
        print(f"\n📝 Uploading {len(all_labels)} annotations as pre-labels...")
        
        try:
            import uuid
            
            upload_job = lb.MALPredictionImport.create_from_objects(
                client=client,
                project_id=project.uid,
                name=f"mal_job_{str(uuid.uuid4())}",
                predictions=all_labels
            )
            
            print(f"   Waiting for annotation upload to complete...")
            upload_job.wait_till_done()
            
            print(f"   ✓ Annotations uploaded as pre-labels")
            print(f"   Upload job ID: {upload_job.uid}")
            print(f"   Errors: {upload_job.errors}")
            print(f"   Status of uploads: {upload_job.statuses}")
                    
        except Exception as e:
            print(f"   ❌ Annotation upload error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No annotations to upload")
    
    # Summary
    print("\n" + "=" * 80)
    print("UPLOAD COMPLETE")
    print("=" * 80)
    print(f"📦 Project: {project_name} ({project.uid})")
    print(f"📊 Dataset: {dataset_name} ({dataset.uid})")
    print(f"🖼️  Images: {len(all_data_rows)}")
    print(f"📝 Annotations: {len(all_labels)}")
    print(f"🔗 Project URL: https://app.labelbox.com/projects/{project.uid}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Upload YOLO dataset to Labelbox with segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with API key from environment variable
  export LABELBOX_API_KEY="your_api_key_here"
  python upload_to_labelbox.py --dir ./big-images-core --project "My YOLO Project"
  
  # Upload only train and val splits
  python upload_to_labelbox.py --dir ./dataset --project "Training Data" --splits train val
  
  # Upload with explicit API key
  python upload_to_labelbox.py --dir ./dataset --project "My Project" --api-key "your_key"
  
  # Use existing ontology (don't create new one)
  python upload_to_labelbox.py --dir ./dataset --project "My Project" --no-create-ontology
  
Requirements:
  pip install labelbox pillow
  
Get your API key from: https://app.labelbox.com/account/api-keys
        """
    )
    
    parser.add_argument('--dir', required=True, dest='directory',
                       help='Dataset root directory (contains dataset.yaml)')
    parser.add_argument('--project', required=True,
                       help='Labelbox project name')
    parser.add_argument('--api-key', dest='api_key',
                       help='Labelbox API key (or set LABELBOX_API_KEY env var)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='Splits to upload (default: all)')
    parser.add_argument('--dataset-name', dest='dataset_name',
                       help='Labelbox dataset name (default: project_name + "_dataset")')
    parser.add_argument('--no-create-ontology', dest='create_ontology', 
                       action='store_false', default=True,
                       help='Don\'t create new ontology (use existing)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('LABELBOX_API_KEY')
    if not api_key:
        print("❌ Error: Labelbox API key not provided")
        print("   Set via --api-key argument or LABELBOX_API_KEY environment variable")
        print("   Get your key from: https://app.labelbox.com/account/api-keys")
        sys.exit(1)
    
    # Upload dataset
    upload_dataset_to_labelbox(
        dataset_root=args.directory,
        api_key=api_key,
        project_name=args.project,
        splits=args.splits,
        dataset_name=args.dataset_name,
        create_ontology=args.create_ontology
    )


if __name__ == "__main__":
    main()
