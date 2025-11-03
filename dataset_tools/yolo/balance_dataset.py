#!/usr/bin/env python3
"""
Balance YOLO dataset by removing files or redistributing across splits.

This tool intelligently balances class distribution by removing files with
dominant overrepresented classes. It proposes changes with detailed statistics
and requires user confirmation before applying modifications.

Author: Alessio Lovato
"""

import os
import shutil
import yaml
from glob import glob
from collections import defaultdict
import sys
import random


def load_dataset_yaml(yaml_path="dataset.yaml"):
    """Load class names from dataset.yaml"""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {})


def analyze_file_labels(label_path):
    """
    Analyze a single label file and return class distribution.
    Returns: dict with class_id counts
    """
    class_counts = defaultdict(int)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls_id = parts[0]
            class_counts[cls_id] += 1
    return dict(class_counts)


def get_split_data(dataset_root, class_names):
    """
    Get detailed information about each file in each split.
    Returns: dict with split -> list of file info
    """
    splits = ["train", "val", "test"]
    split_data = {}
    
    for split in splits:
        label_dir = os.path.join(dataset_root, "labels", split)
        if not os.path.exists(label_dir):
            continue
        
        files = []
        label_files = glob(os.path.join(label_dir, "*.txt"))
        
        for label_path in label_files:
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            class_counts = analyze_file_labels(label_path)
            
            # Calculate dominance (which class is most prevalent)
            if class_counts:
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            else:
                dominant_class = None
            
            files.append({
                "base_name": base_name,
                "label_path": label_path,
                "class_counts": class_counts,
                "dominant_class": dominant_class,
                "total_instances": sum(class_counts.values())
            })
        
        split_data[split] = files
    
    return split_data


def calculate_current_stats(split_data):
    """Calculate current class distribution across splits"""
    stats = {}
    for split, files in split_data.items():
        class_totals = defaultdict(int)
        for file_info in files:
            for cls_id, count in file_info["class_counts"].items():
                class_totals[cls_id] += count
        stats[split] = {
            "files": len(files),
            "class_totals": dict(class_totals)
        }
    return stats


def propose_balancing_strategy(split_data, class_names, target_ratio=2.0):
    """
    Propose a balancing strategy by removing files with dominant overrepresented classes.
    
    Strategy:
    1. Identify the minority class (least instances)
    2. Calculate target max instances based on target_ratio
    3. Remove files dominated by overrepresented classes
    """
    
    print("\n" + "=" * 80)
    print("BALANCING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Calculate total instances per class across all splits
    total_instances = defaultdict(int)
    for split, files in split_data.items():
        for file_info in files:
            for cls_id, count in file_info["class_counts"].items():
                total_instances[cls_id] += count
    
    if not total_instances:
        print("❌ No instances found!")
        return None
    
    # Find minority and majority classes
    min_class = min(total_instances.items(), key=lambda x: x[1])
    max_class = max(total_instances.items(), key=lambda x: x[1])
    
    min_cls_id, min_count = min_class
    max_cls_id, max_count = max_class
    current_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n📊 Current Distribution:")
    for cls_id in sorted(class_names.keys()):
        cls_name = class_names[cls_id]
        count = total_instances.get(str(cls_id), 0)
        percentage = (count / sum(total_instances.values()) * 100) if sum(total_instances.values()) > 0 else 0
        print(f"   Class {cls_id} ({cls_name:10s}): {count:5d} instances ({percentage:5.1f}%)")
    
    print(f"\n⚖️  Current imbalance ratio: {current_ratio:.2f}:1")
    print(f"🎯 Target imbalance ratio: {target_ratio:.2f}:1")
    
    if current_ratio <= target_ratio:
        print(f"\n✓ Dataset is already well balanced (ratio {current_ratio:.2f}:1 <= {target_ratio:.2f}:1)")
        return None
    
    # Calculate target maximum instances
    target_max = int(min_count * target_ratio)
    
    print(f"\n📈 Balancing Plan:")
    print(f"   Minority class: {min_cls_id} ({class_names[int(min_cls_id)]}) with {min_count} instances")
    print(f"   Target max instances per class: {target_max}")
    
    # Identify files to remove for each split
    removal_plan = {}
    
    for split, files in split_data.items():
        # Calculate class totals for this split
        class_totals = defaultdict(int)
        for file_info in files:
            for cls_id, count in file_info["class_counts"].items():
                class_totals[cls_id] += count
        
        # Identify overrepresented classes in this split
        to_remove = []
        removal_targets = {}
        
        for cls_id, total in class_totals.items():
            if total > target_max:
                removal_targets[cls_id] = total - target_max
        
        if removal_targets:
            # Sort files by dominant class and number of instances
            # Prioritize removing files with many instances of overrepresented classes
            candidates = []
            for file_info in files:
                dominant_cls = file_info["dominant_class"]
                if dominant_cls in removal_targets:
                    # Score based on how many overrepresented instances
                    score = file_info["class_counts"].get(dominant_cls, 0)
                    candidates.append((score, file_info))
            
            # Sort by score (descending) - remove files with most overrepresented instances
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Greedily remove files until we reach target
            remaining_targets = removal_targets.copy()
            for score, file_info in candidates:
                if all(v <= 0 for v in remaining_targets.values()):
                    break
                
                # Check if removing this file helps
                helps = False
                for cls_id in remaining_targets:
                    if file_info["class_counts"].get(cls_id, 0) > 0:
                        helps = True
                        break
                
                if helps:
                    to_remove.append(file_info)
                    # Update remaining targets
                    for cls_id, count in file_info["class_counts"].items():
                        if cls_id in remaining_targets:
                            remaining_targets[cls_id] -= count
        
        if to_remove:
            removal_plan[split] = to_remove
    
    return removal_plan


def display_removal_plan(removal_plan, split_data, class_names):
    """Display the proposed removal plan with statistics"""
    
    print("\n" + "=" * 80)
    print("PROPOSED CHANGES")
    print("=" * 80)
    
    total_files_removed = sum(len(files) for files in removal_plan.values())
    
    if total_files_removed == 0:
        print("\n✓ No files need to be removed!")
        return
    
    print(f"\n📝 Total files to remove: {total_files_removed}")
    
    for split, files_to_remove in removal_plan.items():
        original_count = len(split_data[split])
        new_count = original_count - len(files_to_remove)
        
        print(f"\n{'─' * 80}")
        print(f"📁 {split.upper()} SET")
        print(f"{'─' * 80}")
        print(f"   Current files: {original_count}")
        print(f"   Files to remove: {len(files_to_remove)}")
        print(f"   Remaining files: {new_count}")
        
        # Calculate class distribution before and after
        before = defaultdict(int)
        after = defaultdict(int)
        
        for file_info in split_data[split]:
            for cls_id, count in file_info["class_counts"].items():
                before[cls_id] += count
                if file_info not in files_to_remove:
                    after[cls_id] += count
        
        print(f"\n   Class distribution changes:")
        print(f"   {'Class':15s} │ {'Before':>8s} │ {'After':>8s} │ {'Change':>8s}")
        print(f"   {'-'*15:15s}─┼─{'-'*8:8s}─┼─{'-'*8:8s}─┼─{'-'*8:8s}")
        
        for cls_id in sorted(class_names.keys()):
            cls_name = class_names[cls_id]
            label = f"{cls_id}: {cls_name}"
            before_count = before.get(str(cls_id), 0)
            after_count = after.get(str(cls_id), 0)
            change = after_count - before_count
            change_str = f"{change:+d}" if change != 0 else "0"
            print(f"   {label:15s} │ {before_count:8d} │ {after_count:8d} │ {change_str:>8s}")


def calculate_new_stats(split_data, removal_plan, class_names):
    """Calculate statistics after proposed removals"""
    
    new_totals = defaultdict(int)
    
    for split, files in split_data.items():
        files_to_remove = removal_plan.get(split, [])
        for file_info in files:
            if file_info not in files_to_remove:
                for cls_id, count in file_info["class_counts"].items():
                    new_totals[cls_id] += count
    
    if not new_totals:
        return None
    
    min_count = min(new_totals.values())
    max_count = max(new_totals.values())
    new_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("\n" + "=" * 80)
    print("NEW BALANCED DISTRIBUTION")
    print("=" * 80)
    
    print(f"\n📊 After Balancing:")
    total_instances = sum(new_totals.values())
    for cls_id in sorted(class_names.keys()):
        cls_name = class_names[cls_id]
        count = new_totals.get(str(cls_id), 0)
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"   Class {cls_id} ({cls_name:10s}): {count:5d} instances ({percentage:5.1f}%)")
    
    print(f"\n⚖️  New imbalance ratio: {new_ratio:.2f}:1")
    
    if new_ratio <= 2.0:
        print(f"✓ Excellent balance achieved!")
    elif new_ratio <= 3.0:
        print(f"✓ Good balance achieved!")
    else:
        print(f"ℹ️  Moderate balance achieved")
    
    return new_ratio


def apply_removal_plan(dataset_root, removal_plan, backup=True):
    """
    Apply the removal plan by moving files to a backup directory or deleting them.
    """
    
    backup_dir = os.path.join(dataset_root, "removed_files_backup")
    
    if backup:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\n📦 Creating backup in: {backup_dir}")
    
    for split, files_to_remove in removal_plan.items():
        for file_info in files_to_remove:
            base_name = file_info["base_name"]
            
            # Remove label file
            label_src = file_info["label_path"]
            
            # Find corresponding image file
            img_dir = os.path.join(dataset_root, "images", split)
            img_src = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = os.path.join(img_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_src = candidate
                    break
            
            if backup:
                # Move to backup
                backup_split_dir = os.path.join(backup_dir, split)
                os.makedirs(backup_split_dir, exist_ok=True)
                
                # Move label
                shutil.move(label_src, os.path.join(backup_split_dir, os.path.basename(label_src)))
                
                # Move image if found
                if img_src:
                    shutil.move(img_src, os.path.join(backup_split_dir, os.path.basename(img_src)))
            else:
                # Delete permanently
                os.remove(label_src)
                if img_src:
                    os.remove(img_src)
    
    print(f"\n✓ Removal complete!")


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Balance YOLO dataset by removing overrepresented class instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python balance_dataset.py                    # Use default ratio 2.0
  python balance_dataset.py --ratio 1.5        # Target ratio 1.5:1
  python balance_dataset.py --ratio 3.0 /path  # Custom ratio and path
  
Note: Lower ratio = more balanced but more files removed
      Higher ratio = less balanced but fewer files removed
        """
    )
    parser.add_argument('dataset_root', nargs='?', default='.',
                        help='Path to dataset root (default: current directory)')
    parser.add_argument('--ratio', type=float, default=2.0,
                        help='Target imbalance ratio (default: 2.0)')
    
    args = parser.parse_args()
    dataset_root = args.dataset_root
    target_ratio = args.ratio
    
    yaml_path = os.path.join(dataset_root, "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: dataset.yaml not found at {yaml_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("YOLO Dataset Balancing Tool")
    print("=" * 80)
    print(f"\n📋 Dataset: {os.path.abspath(dataset_root)}")
    print(f"🎯 Target ratio: {target_ratio}:1")
    print(f"\n💡 Note: Images and labels are removed from:")
    print(f"   - images/train/, images/val/, images/test/")
    print(f"   - labels/train/, labels/val/, labels/test/")
    
    # Load class names
    class_names = load_dataset_yaml(yaml_path)
    print(f"\n📝 Classes: {', '.join([f'{k}: {v}' for k, v in sorted(class_names.items())])}")
    
    # Get split data
    print(f"\n🔍 Analyzing dataset...")
    split_data = get_split_data(dataset_root, class_names)
    
    # Propose balancing strategy with custom ratio
    removal_plan = propose_balancing_strategy(split_data, class_names, target_ratio=target_ratio)
    
    if removal_plan is None or not removal_plan:
        print(f"\n✓ No changes needed!")
        sys.exit(0)
    
    # Display removal plan
    display_removal_plan(removal_plan, split_data, class_names)
    
    # Calculate and display new statistics
    new_ratio = calculate_new_stats(split_data, removal_plan, class_names)
    
    if new_ratio and new_ratio > target_ratio:
        print(f"\n⚠️  Note: Achieved ratio ({new_ratio:.2f}:1) is higher than target ({target_ratio}:1)")
        print(f"   This happens because files contain multiple classes.")
        print(f"   Try a lower target ratio (e.g., --ratio {target_ratio * 0.7:.1f}) for better balance.")
    
    # Ask for confirmation
    print("\n" + "=" * 80)
    print("⚠️  CONFIRMATION REQUIRED")
    print("=" * 80)
    
    total_files = sum(len(files) for files in removal_plan.values())
    print(f"\nThis will remove {total_files} files from the dataset.")
    print(f"Files will be moved to 'removed_files_backup' folder for safety.")
    print(f"\nYou can restore them later if needed.")
    
    response = input("\n❓ Apply these changes? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        apply_removal_plan(dataset_root, removal_plan, backup=True)
        print(f"\n" + "=" * 80)
        print(f"✓ Dataset balanced successfully!")
        print(f"=" * 80)
        print(f"\n💡 Tip: Run 'python analyze_dataset_balance.py' to verify the new balance.")
    else:
        print(f"\n❌ Operation cancelled. No changes made.")
        sys.exit(0)


if __name__ == "__main__":
    main()
