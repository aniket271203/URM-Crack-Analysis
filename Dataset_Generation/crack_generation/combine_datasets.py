#!/usr/bin/env python3
"""
Dataset Combiner for Crack Detection

This script combines two crack detection datasets (output and output_1) into a single
unified dataset while maintaining the directory structure and avoiding filename conflicts.

Author: Crack Detection Team
Date: October 2025
"""

import os
import shutil
import glob
from pathlib import Path
import time

def create_combined_dataset(source_dir1, source_dir2, output_dir, dry_run=False):
    """
    Combine two crack detection datasets into one unified dataset.
    
    Args:
        source_dir1 (str): Path to first dataset directory (e.g., 'output')
        source_dir2 (str): Path to second dataset directory (e.g., 'output_1')
        output_dir (str): Path to combined output directory
        dry_run (bool): If True, only show what would be done without actually copying
    
    Returns:
        dict: Statistics about the combination process
    """
    
    # Expected crack types
    crack_types = ['vertical', 'horizontal', 'diagonal', 'step']
    
    # Statistics tracking
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'files_by_type': {},
        'conflicts_resolved': 0,
        'errors': []
    }
    
    print("="*80)
    print("CRACK DETECTION DATASET COMBINER")
    print("="*80)
    print(f"Source 1: {source_dir1}")
    print(f"Source 2: {source_dir2}")
    print(f"Output:   {output_dir}")
    print(f"Dry run:  {dry_run}")
    print("="*80)
    
    # Validate source directories
    if not os.path.exists(source_dir1):
        raise ValueError(f"Source directory 1 does not exist: {source_dir1}")
    if not os.path.exists(source_dir2):
        raise ValueError(f"Source directory 2 does not exist: {source_dir2}")
    
    # Create output directory structure
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        for crack_type in crack_types:
            os.makedirs(os.path.join(output_dir, crack_type, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, crack_type, 'annotations'), exist_ok=True)
            if os.path.exists(os.path.join(source_dir1, crack_type, 'reasons')) or \
               os.path.exists(os.path.join(source_dir2, crack_type, 'reasons')):
                os.makedirs(os.path.join(output_dir, crack_type, 'reasons'), exist_ok=True)
    
    # Track existing filenames to avoid conflicts
    existing_files = {}
    
    def get_unique_filename(base_name, extension, crack_type, file_type):
        """Generate a unique filename to avoid conflicts."""
        original_name = f"{base_name}{extension}"
        counter = 1
        
        # Check if filename already exists
        key = f"{crack_type}_{file_type}"
        if key not in existing_files:
            existing_files[key] = set()
        
        unique_name = original_name
        while unique_name in existing_files[key]:
            unique_name = f"{base_name}_dup{counter:03d}{extension}"
            counter += 1
            if counter > 999:  # Safety check
                raise ValueError(f"Too many duplicates for {original_name}")
        
        existing_files[key].add(unique_name)
        return unique_name
    
    def copy_files_from_source(source_dir, source_name):
        """Copy files from a source directory to the combined dataset."""
        print(f"\nProcessing {source_name}: {source_dir}")
        source_stats = {'images': 0, 'annotations': 0, 'reasons': 0}
        
        for crack_type in crack_types:
            source_crack_dir = os.path.join(source_dir, crack_type)
            
            if not os.path.exists(source_crack_dir):
                print(f"  ⚠️  {crack_type} directory not found in {source_name}")
                continue
            
            print(f"\n  Processing {crack_type} cracks...")
            
            # Initialize stats for this crack type
            if crack_type not in stats['files_by_type']:
                stats['files_by_type'][crack_type] = {'images': 0, 'annotations': 0, 'reasons': 0}
            
            # Process images
            images_dir = os.path.join(source_crack_dir, 'images')
            if os.path.exists(images_dir):
                image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                             glob.glob(os.path.join(images_dir, '*.png')) + \
                             glob.glob(os.path.join(images_dir, '*.jpeg'))
                
                print(f"    Found {len(image_files)} images")
                
                for img_path in image_files:
                    try:
                        img_name = os.path.basename(img_path)
                        base_name, extension = os.path.splitext(img_name)
                        
                        # Generate unique filename
                        unique_name = get_unique_filename(base_name, extension, crack_type, 'images')
                        
                        if unique_name != img_name:
                            stats['conflicts_resolved'] += 1
                            print(f"      Renamed: {img_name} → {unique_name}")
                        
                        # Copy image
                        dst_path = os.path.join(output_dir, crack_type, 'images', unique_name)
                        if not dry_run:
                            shutil.copy2(img_path, dst_path)
                        
                        source_stats['images'] += 1
                        stats['files_by_type'][crack_type]['images'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error copying image {img_path}: {e}"
                        print(f"      ❌ {error_msg}")
                        stats['errors'].append(error_msg)
            
            # Process annotations
            annotations_dir = os.path.join(source_crack_dir, 'annotations')
            if os.path.exists(annotations_dir):
                annotation_files = glob.glob(os.path.join(annotations_dir, '*.txt')) + \
                                 glob.glob(os.path.join(annotations_dir, '*.json'))
                
                print(f"    Found {len(annotation_files)} annotations")
                
                for ann_path in annotation_files:
                    try:
                        ann_name = os.path.basename(ann_path)
                        base_name, extension = os.path.splitext(ann_name)
                        
                        # For annotations, try to match with corresponding image name
                        # If image was renamed, rename annotation accordingly
                        img_key = f"{crack_type}_images"
                        if img_key in existing_files:
                            # Find corresponding image name
                            matching_img = None
                            for existing_img in existing_files[img_key]:
                                existing_base = os.path.splitext(existing_img)[0]
                                if existing_base.startswith(base_name) or base_name.startswith(existing_base):
                                    matching_img = existing_img
                                    break
                            
                            if matching_img:
                                # Use the same base name as the matching image
                                img_base = os.path.splitext(matching_img)[0]
                                unique_name = f"{img_base}{extension}"
                            else:
                                unique_name = get_unique_filename(base_name, extension, crack_type, 'annotations')
                        else:
                            unique_name = get_unique_filename(base_name, extension, crack_type, 'annotations')
                        
                        # Copy annotation
                        dst_path = os.path.join(output_dir, crack_type, 'annotations', unique_name)
                        if not dry_run:
                            shutil.copy2(ann_path, dst_path)
                        
                        source_stats['annotations'] += 1
                        stats['files_by_type'][crack_type]['annotations'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error copying annotation {ann_path}: {e}"
                        print(f"      ❌ {error_msg}")
                        stats['errors'].append(error_msg)
            
            # Process reasons (if they exist)
            reasons_dir = os.path.join(source_crack_dir, 'reasons')
            if os.path.exists(reasons_dir):
                reason_files = glob.glob(os.path.join(reasons_dir, '*.txt'))
                
                print(f"    Found {len(reason_files)} reason files")
                
                for reason_path in reason_files:
                    try:
                        reason_name = os.path.basename(reason_path)
                        base_name, extension = os.path.splitext(reason_name)
                        
                        # Generate unique filename
                        unique_name = get_unique_filename(base_name, extension, crack_type, 'reasons')
                        
                        # Copy reason file
                        dst_path = os.path.join(output_dir, crack_type, 'reasons', unique_name)
                        if not dry_run:
                            shutil.copy2(reason_path, dst_path)
                        
                        source_stats['reasons'] += 1
                        stats['files_by_type'][crack_type]['reasons'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error copying reason {reason_path}: {e}"
                        print(f"      ❌ {error_msg}")
                        stats['errors'].append(error_msg)
        
        print(f"\n  {source_name} Summary:")
        print(f"    Images:      {source_stats['images']}")
        print(f"    Annotations: {source_stats['annotations']}")
        print(f"    Reasons:     {source_stats['reasons']}")
        
        return source_stats
    
    # Copy files from both sources
    stats1 = copy_files_from_source(source_dir1, "Dataset 1")
    stats2 = copy_files_from_source(source_dir2, "Dataset 2")
    
    # Calculate total stats
    stats['total_images'] = stats1['images'] + stats2['images']
    stats['total_annotations'] = stats1['annotations'] + stats2['annotations']
    
    # Copy CSV files if they exist
    print(f"\nCopying CSV files...")
    csv_files_copied = 0
    for source_dir, source_name in [(source_dir1, "dataset1"), (source_dir2, "dataset2")]:
        csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
        for csv_path in csv_files:
            csv_name = os.path.basename(csv_path)
            base_name, extension = os.path.splitext(csv_name)
            
            # Add source prefix to avoid conflicts
            new_name = f"{source_name}_{base_name}{extension}"
            dst_path = os.path.join(output_dir, new_name)
            
            if not dry_run:
                shutil.copy2(csv_path, dst_path)
            
            print(f"  Copied: {csv_name} → {new_name}")
            csv_files_copied += 1
    
    return stats, csv_files_copied

def print_final_summary(stats, csv_files_copied, output_dir, dry_run):
    """Print final summary of the combination process."""
    
    print("\n" + "="*80)
    print("COMBINATION COMPLETE!")
    print("="*80)
    
    if dry_run:
        print("🔍 DRY RUN SUMMARY (no files were actually copied)")
    else:
        print("✅ FILES SUCCESSFULLY COMBINED")
    
    print(f"\nTotal Statistics:")
    print(f"  📁 Output Directory: {output_dir}")
    print(f"  🖼️  Total Images:      {stats['total_images']}")
    print(f"  📝 Total Annotations: {stats['total_annotations']}")
    print(f"  📊 CSV Files:         {csv_files_copied}")
    print(f"  🔧 Conflicts Resolved: {stats['conflicts_resolved']}")
    print(f"  ❌ Errors:            {len(stats['errors'])}")
    
    print(f"\nFiles by Crack Type:")
    for crack_type, counts in stats['files_by_type'].items():
        print(f"  {crack_type.upper():12}")
        print(f"    Images:      {counts['images']:4}")
        print(f"    Annotations: {counts['annotations']:4}")
        if counts['reasons'] > 0:
            print(f"    Reasons:     {counts['reasons']:4}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors encountered:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    
    print("="*80)

def main():
    """Main function to combine datasets."""
    
    # Configuration
    SOURCE_DIR_1 = "output"      # First dataset directory
    SOURCE_DIR_2 = "output_1"    # Second dataset directory
    OUTPUT_DIR = "combined_output"  # Combined dataset directory
    DRY_RUN = False              # Set to True to preview without copying
    
    # Allow command line arguments for convenience
    import sys
    if len(sys.argv) >= 4:
        SOURCE_DIR_1 = sys.argv[1]
        SOURCE_DIR_2 = sys.argv[2]
        OUTPUT_DIR = sys.argv[3]
        if len(sys.argv) >= 5:
            DRY_RUN = sys.argv[4].lower() in ['true', '1', 'yes', 'dry']
    
    try:
        start_time = time.time()
        
        # Combine datasets
        stats, csv_files = create_combined_dataset(
            SOURCE_DIR_1, 
            SOURCE_DIR_2, 
            OUTPUT_DIR, 
            dry_run=DRY_RUN
        )
        
        # Print summary
        print_final_summary(stats, csv_files, OUTPUT_DIR, DRY_RUN)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")
        
        if not DRY_RUN:
            print(f"\n🎉 Combined dataset is ready at: {os.path.abspath(OUTPUT_DIR)}")
            print("\nNext steps:")
            print("1. Verify the combined dataset structure")
            print("2. Check for any missing image-annotation pairs")
            print("3. Use the combined dataset for YOLO training")
        else:
            print(f"\n💡 To actually combine the datasets, set DRY_RUN = False or run:")
            print(f"   python combine_datasets.py {SOURCE_DIR_1} {SOURCE_DIR_2} {OUTPUT_DIR} false")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
