#!/usr/bin/env python3
"""
Analyze YOLO dataset label balance across train/val/test splits.
Shows class distribution, instance counts, and visualizes the balance.
"""

import os
import yaml
from glob import glob
from collections import defaultdict
import sys


def load_dataset_yaml(yaml_path="dataset.yaml"):
    """Load class names from dataset.yaml"""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {})


def analyze_labels(label_dir, class_names):
    """
    Analyze label files and count instances per class.
    
    Returns:
        dict: {class_id: count} for polygon instances
        int: total number of label files
        int: total number of polygon instances
    """
    class_counts = defaultdict(int)
    total_files = 0
    total_instances = 0
    
    label_files = glob(os.path.join(label_dir, "*.txt"))
    
    for label_path in label_files:
        total_files += 1
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:  # Need at least class + one coordinate pair
                    continue
                
                cls_id = parts[0]
                class_counts[cls_id] += 1
                total_instances += 1
    
    return dict(class_counts), total_files, total_instances


def print_bar(label, count, max_count, width=50):
    """Print a horizontal bar chart"""
    if max_count == 0:
        bar_length = 0
    else:
        bar_length = int((count / max_count) * width)
    
    bar = "█" * bar_length
    percentage = (count / max_count * 100) if max_count > 0 else 0
    print(f"  {label:15s} │ {bar:<{width}s} │ {count:6d} ({percentage:5.1f}%)")


def analyze_dataset(dataset_root=".", yaml_path="dataset.yaml"):
    """Main analysis function"""
    
    print("=" * 80)
    print("YOLO Dataset Balance Analysis")
    print("=" * 80)
    
    # Load class names
    class_names = load_dataset_yaml(yaml_path)
    print(f"\n📋 Dataset: {os.path.abspath(dataset_root)}")
    print(f"📝 Classes defined: {len(class_names)}")
    for cls_id, cls_name in sorted(class_names.items()):
        print(f"   - Class {cls_id}: {cls_name}")
    
    splits = ["train", "val", "test"]
    split_data = {}
    
    # Analyze each split
    for split in splits:
        label_dir = os.path.join(dataset_root, "labels", split)
        
        if not os.path.exists(label_dir):
            print(f"\n⚠️  Split '{split}' not found at {label_dir}")
            continue
        
        class_counts, total_files, total_instances = analyze_labels(label_dir, class_names)
        split_data[split] = {
            "class_counts": class_counts,
            "total_files": total_files,
            "total_instances": total_instances
        }
    
    # Print detailed statistics for each split
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS BY SPLIT")
    print("=" * 80)
    
    for split in splits:
        if split not in split_data:
            continue
        
        data = split_data[split]
        class_counts = data["class_counts"]
        total_files = data["total_files"]
        total_instances = data["total_instances"]
        
        print(f"\n{'─' * 80}")
        print(f"📁 {split.upper()} SET")
        print(f"{'─' * 80}")
        print(f"   Label files: {total_files}")
        print(f"   Total instances: {total_instances}")
        
        if total_instances == 0:
            print("   ⚠️  No instances found!")
            continue
        
        print(f"\n   Instance distribution:")
        print(f"   {'Class':15s} │ {'Distribution':^50s} │ {'Count':^6s}")
        print(f"   {'-'*15:15s}─┼─{'-'*50:50s}─┼─{'-'*13:13s}")
        
        # Get max count for scaling the bars
        max_count = max(class_counts.values()) if class_counts else 0
        
        # Print bars for each class
        for cls_id in sorted(class_names.keys()):
            cls_name = class_names[cls_id]
            count = class_counts.get(str(cls_id), 0)
            label = f"{cls_id}: {cls_name}"
            print_bar(label, count, max_count)
        
        # Calculate balance metrics
        if len(class_counts) > 1:
            counts = list(class_counts.values())
            min_count = min(counts)
            max_count_val = max(counts)
            imbalance_ratio = max_count_val / min_count if min_count > 0 else float('inf')
            print(f"\n   ⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
            
            if imbalance_ratio > 3:
                print(f"   ⚠️  Warning: Significant class imbalance detected!")
            elif imbalance_ratio > 1.5:
                print(f"   ℹ️  Moderate class imbalance")
            else:
                print(f"   ✓  Classes are well balanced")
    
    # Summary comparison across splits
    if len(split_data) > 1:
        print("\n" + "=" * 80)
        print("SPLIT COMPARISON")
        print("=" * 80)
        
        print(f"\n{'Split':10s} │ {'Files':>8s} │ {'Instances':>10s} │ {'Avg/File':>10s}")
        print(f"{'-'*10:10s}─┼─{'-'*8:8s}─┼─{'-'*10:10s}─┼─{'-'*10:10s}")
        
        for split in splits:
            if split not in split_data:
                continue
            
            data = split_data[split]
            avg = data["total_instances"] / data["total_files"] if data["total_files"] > 0 else 0
            print(f"{split:10s} │ {data['total_files']:8d} │ {data['total_instances']:10d} │ {avg:10.2f}")
        
        # Class distribution across splits
        print(f"\n{'Class':15s} │ {'Train':>8s} │ {'Val':>8s} │ {'Test':>8s} │ {'Total':>8s}")
        print(f"{'-'*15:15s}─┼─{'-'*8:8s}─┼─{'-'*8:8s}─┼─{'-'*8:8s}─┼─{'-'*8:8s}")
        
        for cls_id in sorted(class_names.keys()):
            cls_name = class_names[cls_id]
            label = f"{cls_id}: {cls_name}"
            
            train_count = split_data.get("train", {}).get("class_counts", {}).get(str(cls_id), 0)
            val_count = split_data.get("val", {}).get("class_counts", {}).get(str(cls_id), 0)
            test_count = split_data.get("test", {}).get("class_counts", {}).get(str(cls_id), 0)
            total = train_count + val_count + test_count
            
            print(f"{label:15s} │ {train_count:8d} │ {val_count:8d} │ {test_count:8d} │ {total:8d}")
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Check if custom path provided
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    else:
        dataset_root = "."
    
    yaml_path = os.path.join(dataset_root, "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: dataset.yaml not found at {yaml_path}")
        sys.exit(1)
    
    analyze_dataset(dataset_root, yaml_path)
