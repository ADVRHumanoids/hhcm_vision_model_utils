#!/usr/bin/env python3
"""
Check for missing JPEG/JSON pairs in a folder.

- Scans a folder for .jpg/.jpeg and .json files
- Reports files missing their pair (e.g., .jpg without .json or vice versa)
"""
import os
import argparse


def check_missing_pairs(folder):
    jpg_files = set()
    json_files = set()
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.add(os.path.splitext(f)[0])
        elif f.lower().endswith('.json'):
            json_files.add(os.path.splitext(f)[0])
    missing_json = sorted(jpg_files - json_files)
    missing_jpg = sorted(json_files - jpg_files)
    print(f"Checked folder: {folder}")
    if missing_json:
        print("JPEG files missing JSON:")
        for name in missing_json:
            print(f"  {name}.jpg")
    else:
        print("No JPEG files missing JSON.")
    if missing_jpg:
        print("JSON files missing JPEG:")
        for name in missing_jpg:
            print(f"  {name}.json")
    else:
        print("No JSON files missing JPEG.")


def main():
    parser = argparse.ArgumentParser(description="Check for missing JPEG/JSON pairs in a folder.")
    parser.add_argument('--folder', type=str, required=True, help='Folder to check')
    args = parser.parse_args()
    check_missing_pairs(args.folder)

if __name__ == "__main__":
    main()
