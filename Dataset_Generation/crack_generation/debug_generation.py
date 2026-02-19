#!/usr/bin/env python3
"""
Debug script to investigate annotation generation issues.
"""

import os
import sys
import glob
import csv
import json

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def count_files_by_type(output_dir):
    """Count files in each crack type directory."""
    results = {}
    
    for crack_type in ['vertical', 'horizontal', 'diagonal', 'step']:
        crack_dir = os.path.join(output_dir, crack_type)
        results[crack_type] = {
            'images': 0,
            'annotations_yolo': 0,
            'annotations_coco': 0,
            'reasons': 0
        }
        
        if os.path.exists(crack_dir):
            # Count images
            images_dir = os.path.join(crack_dir, "images")
            if os.path.exists(images_dir):
                images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                results[crack_type]['images'] = len(images)
            
            # Count annotations
            annotations_dir = os.path.join(crack_dir, "annotations")
            if os.path.exists(annotations_dir):
                # YOLO annotations
                txt_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
                results[crack_type]['annotations_yolo'] = len(txt_files)
                
                # COCO annotations
                coco_file = os.path.join(annotations_dir, "annotations.json")
                if os.path.exists(coco_file):
                    try:
                        with open(coco_file, 'r') as f:
                            data = json.load(f)
                        results[crack_type]['annotations_coco'] = len(data.get('annotations', []))
                    except:
                        results[crack_type]['annotations_coco'] = -1  # Corrupted
            
            # Count reasons
            reasons_dir = os.path.join(crack_dir, "reasons")
            if os.path.exists(reasons_dir):
                reason_files = [f for f in os.listdir(reasons_dir) if f.endswith('.txt')]
                results[crack_type]['reasons'] = len(reason_files)
    
    return results

def count_csv_entries(output_dir):
    """Count entries in CSV file by crack type."""
    csv_path = os.path.join(output_dir, "crack_analysis_results.csv")
    csv_counts = {'vertical': 0, 'horizontal': 0, 'diagonal': 0, 'step': 0}
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    crack_type = row.get('Crack Type', '').strip()
                    if crack_type in csv_counts:
                        csv_counts[crack_type] += 1
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    return csv_counts

def check_file_annotation_matches(output_dir):
    """Check if image files have corresponding annotation files."""
    mismatches = []
    
    for crack_type in ['vertical', 'horizontal', 'diagonal', 'step']:
        images_dir = os.path.join(output_dir, crack_type, "images")
        annotations_dir = os.path.join(output_dir, crack_type, "annotations")
        
        if os.path.exists(images_dir) and os.path.exists(annotations_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for image_file in images:
                image_base = os.path.splitext(image_file)[0]
                
                # Check for YOLO annotation
                yolo_annotation = os.path.join(annotations_dir, f"{image_base}.txt")
                if not os.path.exists(yolo_annotation):
                    mismatches.append(f"{crack_type}: {image_file} missing YOLO annotation")
    
    return mismatches

def main():
    output_dir = config.OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
    
    print("Crack Generation Debug Report")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Default annotation format: {config.DEFAULT_ANNOTATION_FORMAT}")
    print()
    
    # Count files by type
    file_counts = count_files_by_type(output_dir)
    csv_counts = count_csv_entries(output_dir)
    
    print("File Counts by Crack Type:")
    print("-" * 30)
    for crack_type in ['vertical', 'horizontal', 'diagonal', 'step']:
        counts = file_counts[crack_type]
        csv_count = csv_counts[crack_type]
        
        print(f"{crack_type.upper()}:")
        print(f"  Images:           {counts['images']}")
        print(f"  YOLO Annotations: {counts['annotations_yolo']}")
        print(f"  COCO Annotations: {counts['annotations_coco']}")
        print(f"  Reasons:          {counts['reasons']}")
        print(f"  CSV Entries:      {csv_count}")
        
        # Check for discrepancies
        if counts['images'] != csv_count:
            print(f"  ⚠️  MISMATCH: {counts['images']} images vs {csv_count} CSV entries")
        
        if config.DEFAULT_ANNOTATION_FORMAT.lower() == 'yolo':
            if counts['images'] != counts['annotations_yolo']:
                print(f"  ⚠️  MISMATCH: {counts['images']} images vs {counts['annotations_yolo']} YOLO annotations")
        elif config.DEFAULT_ANNOTATION_FORMAT.lower() == 'coco':
            if counts['annotations_coco'] == -1:
                print(f"  ⚠️  CORRUPTED: COCO annotation file is corrupted")
            elif counts['images'] > 0 and counts['annotations_coco'] == 0:
                print(f"  ⚠️  MISSING: No COCO annotations found")
        
        print()
    
    # Check for missing annotation files
    mismatches = check_file_annotation_matches(output_dir)
    if mismatches:
        print("Missing Annotation Files:")
        print("-" * 30)
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  {mismatch}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        print()
    
    # Summary
    total_images = sum(file_counts[ct]['images'] for ct in file_counts)
    total_csv = sum(csv_counts.values())
    
    print("Summary:")
    print("-" * 30)
    print(f"Total images: {total_images}")
    print(f"Total CSV entries: {total_csv}")
    
    if total_images != total_csv:
        print("❌ Images and CSV entries don't match!")
        print("This suggests the CSV is being written even when image/annotation creation fails.")
    else:
        print("✅ Images and CSV entries match")

if __name__ == "__main__":
    main()
