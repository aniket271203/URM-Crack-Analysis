#!/usr/bin/env python3
"""
Utility script to check and repair corrupted annotation files.
"""

import os
import json
import glob
import argparse
from annotation import validate_coco_json, backup_coco_json

def check_coco_files(directory):
    """Check all COCO JSON files in a directory for corruption."""
    print(f"Checking COCO JSON files in: {directory}")
    
    # Find all annotation.json files
    json_files = glob.glob(os.path.join(directory, "**/annotations.json"), recursive=True)
    
    if not json_files:
        print("No COCO annotation files found.")
        return
    
    for json_file in json_files:
        print(f"\nChecking: {json_file}")
        is_valid, error_msg = validate_coco_json(json_file)
        
        if is_valid:
            print(f"  ✓ Valid - {error_msg}")
            
            # Count entries
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                print(f"  Images: {len(data.get('images', []))}")
                print(f"  Annotations: {len(data.get('annotations', []))}")
            except:
                pass
        else:
            print(f"  ✗ CORRUPTED - {error_msg}")
            
            # Try to backup the corrupted file
            try:
                backup_path = backup_coco_json(json_file)
                if backup_path:
                    print(f"  Backup created: {backup_path}")
            except Exception as e:
                print(f"  Failed to create backup: {e}")

def list_generated_files(directory):
    """List all generated crack images and annotations."""
    print(f"Listing generated files in: {directory}")
    
    crack_types = ['vertical', 'horizontal', 'diagonal', 'step']
    
    for crack_type in crack_types:
        crack_dir = os.path.join(directory, crack_type)
        if os.path.exists(crack_dir):
            images_dir = os.path.join(crack_dir, "images")
            annotations_dir = os.path.join(crack_dir, "annotations")
            
            if os.path.exists(images_dir):
                images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                print(f"\n{crack_type.upper()} Cracks:")
                print(f"  Images: {len(images)}")
                
                if os.path.exists(annotations_dir):
                    # Count annotations
                    if os.path.exists(os.path.join(annotations_dir, "annotations.json")):
                        try:
                            with open(os.path.join(annotations_dir, "annotations.json"), 'r') as f:
                                data = json.load(f)
                            print(f"  COCO Annotations: {len(data.get('annotations', []))}")
                        except:
                            print(f"  COCO Annotations: CORRUPTED")
                    
                    txt_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
                    print(f"  YOLO Annotations: {len(txt_files)}")

def main():
    parser = argparse.ArgumentParser(description="Check and repair annotation files")
    parser.add_argument("--output_dir", default="output", help="Output directory to check")
    parser.add_argument("--check", action="store_true", help="Check COCO JSON files for corruption")
    parser.add_argument("--list", action="store_true", help="List generated files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Directory does not exist: {args.output_dir}")
        return
    
    if args.check:
        check_coco_files(args.output_dir)
    
    if args.list:
        list_generated_files(args.output_dir)
    
    if not args.check and not args.list:
        print("Use --check to validate JSON files or --list to show generated files")

if __name__ == "__main__":
    main()
