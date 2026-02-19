#!/usr/bin/env python3
"""
Batch processing script for Darbhanga Fort test images
Runs crack analysis pipeline on all images with systematic output organization
"""

import os
import sys
import json
import csv
import glob
from datetime import datetime
import traceback
from pathlib import Path
import shutil
from pipeline_orchestrator import CrackAnalysisPipeline

def create_output_structure(base_output_dir):
    """Create systematic output directory structure"""
    output_dirs = {
        'base': base_output_dir,
        'summaries': os.path.join(base_output_dir, 'summaries'),
        'classifications': os.path.join(base_output_dir, 'classifications'),
        'bounding_boxes': os.path.join(base_output_dir, 'bounding_boxes'),
        'detection_results': os.path.join(base_output_dir, 'detection_results'),
        'intermediate': os.path.join(base_output_dir, 'intermediate'),
        'failed': os.path.join(base_output_dir, 'failed'),
        'logs': os.path.join(base_output_dir, 'logs')
    }
    
    # Create all directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create classification subdirectories
    classification_types = ['vertical', 'horizontal', 'diagonal', 'step', 'no_crack']
    for class_type in classification_types:
        os.makedirs(os.path.join(output_dirs['classifications'], class_type), exist_ok=True)
    
    return output_dirs

def process_single_image(pipeline, image_path, output_dirs, log_file):
    """Process a single image and organize results"""
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    print(f"Processing: {image_name}")
    log_file.write(f"\n=== Processing {image_name} at {datetime.now()} ===\n")
    
    try:
        # Create temporary output directory for this image
        temp_output = os.path.join(output_dirs['intermediate'], base_name)
        os.makedirs(temp_output, exist_ok=True)
        
        # Process image
        results = pipeline.process_image(
            image_path,
            use_rag=False,  # No RAG as requested
            save_intermediate=True,
            output_dir=temp_output
        )
        
        # Get summary
        summary = pipeline.get_summary(results)
        
        # Save summary
        summary_path = os.path.join(output_dirs['summaries'], f"{base_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Processed at: {datetime.now()}\n")
            f.write("="*60 + "\n")
            f.write(summary)
        
        # Extract classification result
        classification = "no_crack"  # default
        crack_detected = False
        
        if results and 'detection' in results:
            detection_results = results['detection']
            if detection_results and len(detection_results.get('boxes', [])) > 0:
                crack_detected = True
                
                # Copy bounding box image if available
                bbox_files = glob.glob(os.path.join(temp_output, "*_detections.jpg"))
                if bbox_files:
                    bbox_source = bbox_files[0]
                    bbox_dest = os.path.join(output_dirs['bounding_boxes'], f"{base_name}_bbox.jpg")
                    shutil.copy2(bbox_source, bbox_dest)
                
                # Get classification if crack was detected
                if 'classification' in results and results['classification']:
                    class_results = results['classification']
                    if 'predicted_class' in class_results:
                        classification = class_results['predicted_class'].lower()
        
        # Copy original image to appropriate classification folder
        dest_path = os.path.join(output_dirs['classifications'], classification, image_name)
        shutil.copy2(image_path, dest_path)
        
        # Save detailed results as JSON
        results_path = os.path.join(output_dirs['detection_results'], f"{base_name}_results.json")
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as json_error:
            log_file.write(f"Warning: Could not save JSON results for {image_name}: {json_error}\n")
        
        log_file.write(f"SUCCESS: Classification = {classification}, Crack detected = {crack_detected}\n")
        log_file.flush()
        
        return {
            'image_name': image_name,
            'status': 'success',
            'crack_detected': crack_detected,
            'classification': classification,
            'summary_path': summary_path,
            'bbox_path': os.path.join(output_dirs['bounding_boxes'], f"{base_name}_bbox.jpg") if crack_detected else None
        }
        
    except Exception as e:
        error_msg = f"ERROR processing {image_name}: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")
        log_file.write(f"Traceback:\n{traceback.format_exc()}\n")
        log_file.flush()
        
        # Move image to failed folder
        failed_path = os.path.join(output_dirs['failed'], image_name)
        try:
            shutil.copy2(image_path, failed_path)
        except:
            pass
        
        return {
            'image_name': image_name,
            'status': 'failed',
            'error': str(e),
            'crack_detected': False,
            'classification': 'error'
        }

def main():
    # Configuration
    INPUT_DIR = "/home/samarth/SEM7/BTP/Crack Data-Darbhanga Fort/test"
    OUTPUT_DIR = "/home/samarth/SEM7/BTP/Final_Combined_Model/darbhanga_results"
    
    # Model paths (using defaults from run_pipeline.py)
    YOLO_MODEL = "Crack_Detection_YOLO/crack_yolo_train/weights/best.pt"
    SEG_MODEL = "Masking_and_Classification_model/pretrained_net_G.pth"
    CLASS_MODEL = "Masking_and_Classification_model/crack_orientation_classifier.h5"
    
    print("="*80)
    print("DARBHANGA FORT CRACK ANALYSIS - BATCH PROCESSING")
    print("="*80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"RAG: Disabled (as requested)")
    
    # Get all JPG images
    image_patterns = [
        os.path.join(INPUT_DIR, "*.jpg"),
        os.path.join(INPUT_DIR, "*.JPG"),
        os.path.join(INPUT_DIR, "*.jpeg"),
        os.path.join(INPUT_DIR, "*.JPEG")
    ]
    
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(pattern))
    
    all_images.sort()  # Sort for consistent processing order
    
    print(f"Found {len(all_images)} images to process")
    
    if len(all_images) == 0:
        print("No images found! Check the input directory path.")
        return 1
    
    # Create output structure
    output_dirs = create_output_structure(OUTPUT_DIR)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    try:
        pipeline = CrackAnalysisPipeline(
            yolo_model_path=YOLO_MODEL,
            segmentation_model_path=SEG_MODEL,
            classification_model_path=CLASS_MODEL,
            rag_data_dir=None,  # No RAG
            device="cuda"
        )
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        traceback.print_exc()
        return 1
    
    # Open log file
    log_path = os.path.join(output_dirs['logs'], f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Process images
    results_summary = []
    
    with open(log_path, 'w') as log_file:
        log_file.write(f"Batch Processing Log - Started at {datetime.now()}\n")
        log_file.write(f"Total images: {len(all_images)}\n")
        log_file.write("="*80 + "\n")
        
        for i, image_path in enumerate(all_images, 1):
            print(f"\n[{i}/{len(all_images)}] Processing {os.path.basename(image_path)}")
            
            result = process_single_image(pipeline, image_path, output_dirs, log_file)
            results_summary.append(result)
            
            # Print progress
            if i % 10 == 0:
                successful = sum(1 for r in results_summary if r['status'] == 'success')
                print(f"Progress: {i}/{len(all_images)} completed ({successful} successful)")
    
    # Create final summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "processing_summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'status', 'crack_detected', 'classification', 'summary_path', 'bbox_path', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_summary:
            # Fill missing fields
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    # Final statistics
    total_images = len(results_summary)
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    failed = total_images - successful
    
    crack_detected = sum(1 for r in results_summary if r.get('crack_detected', False))
    no_crack = successful - crack_detected
    
    # Classification counts
    classification_counts = {}
    for result in results_summary:
        if result['status'] == 'success':
            class_type = result['classification']
            classification_counts[class_type] = classification_counts.get(class_type, 0) + 1
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"Total Images: {total_images}")
    print(f"Successfully Processed: {successful}")
    print(f"Failed: {failed}")
    print(f"\nCrack Detection:")
    print(f"  Crack Detected: {crack_detected}")
    print(f"  No Crack: {no_crack}")
    print(f"\nClassification Distribution:")
    for class_type, count in sorted(classification_counts.items()):
        print(f"  {class_type.upper()}: {count}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Summary CSV: {csv_path}")
    print(f"Log file: {log_path}")
    
    print("\nOutput Structure:")
    print(f"  Classifications: {output_dirs['classifications']}/")
    print(f"  Bounding Boxes: {output_dirs['bounding_boxes']}/")
    print(f"  Summaries: {output_dirs['summaries']}/")
    print(f"  Detection Results: {output_dirs['detection_results']}/")
    print(f"  Failed Images: {output_dirs['failed']}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
