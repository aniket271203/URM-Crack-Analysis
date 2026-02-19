"""
Create final test dataset from existing generated crack data
Consolidates crack data from output/ folder into proper benchmark format
"""

import os
import sys
import json
import shutil
import pandas as pd
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List


def load_yolo_annotation(annotation_path: str):
    """Load YOLO format annotation"""
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
            
        # Parse first line (assuming single object per image)
        parts = lines[0].strip().split()
        if len(parts) >= 5:
            class_id, center_x, center_y, width, height = map(float, parts[:5])
            return {
                'class_id': int(class_id),
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            }
    except Exception as e:
        print(f"Error loading annotation {annotation_path}: {e}")
    
    return None


def yolo_to_bbox(yolo_data: Dict, img_width: int, img_height: int):
    """Convert YOLO format to bounding box coordinates"""
    center_x = yolo_data['center_x'] * img_width
    center_y = yolo_data['center_y'] * img_height
    width = yolo_data['width'] * img_width
    height = yolo_data['height'] * img_height
    
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return [x1, y1, x2, y2]


def consolidate_crack_data(source_dir: Path, target_dir: Path, max_per_type: int = 40):
    """Consolidate crack data from type-specific folders"""
    
    crack_types = ['vertical', 'horizontal', 'diagonal', 'step']
    consolidated_data = []
    
    for crack_type in crack_types:
        crack_type_dir = source_dir / crack_type
        if not crack_type_dir.exists():
            print(f"⚠️ Directory not found: {crack_type_dir}")
            continue
            
        images_dir = crack_type_dir / "images"
        annotations_dir = crack_type_dir / "annotations"
        
        if not images_dir.exists() or not annotations_dir.exists():
            print(f"⚠️ Missing subdirectories for {crack_type}")
            continue
        
        # Get list of images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"📂 Found {len(image_files)} {crack_type} images")
        
        # Randomly select up to max_per_type images
        if len(image_files) > max_per_type:
            image_files = random.sample(image_files, max_per_type)
            print(f"   Selected {max_per_type} images for balance")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Processing {crack_type}"):
            # Find corresponding annotation
            annotation_path = annotations_dir / (img_path.stem + ".txt")
            
            if not annotation_path.exists():
                print(f"⚠️ Missing annotation: {annotation_path}")
                continue
            
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Cannot load image: {img_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Load annotation
            yolo_data = load_yolo_annotation(str(annotation_path))
            if yolo_data is None:
                print(f"⚠️ Cannot load annotation: {annotation_path}")
                continue
            
            # Convert to bbox
            bbox = yolo_to_bbox(yolo_data, img_width, img_height)
            
            # Determine severity from filename (if encoded)
            filename = img_path.name
            if 'light' in filename.lower():
                severity = 'light'
            elif 'severe' in filename.lower():
                severity = 'severe'
            else:
                severity = 'medium'
            
            # Create new filename
            new_filename = f"{crack_type}_{severity}_{len(consolidated_data):03d}.jpg"
            
            # Copy image
            target_img_path = target_dir / "images" / new_filename
            shutil.copy2(img_path, target_img_path)
            
            # Create annotation in YOLO format
            target_ann_path = target_dir / "annotations" / (new_filename.replace('.jpg', '.txt'))
            with open(target_ann_path, 'w') as f:
                f.write(f"0 {yolo_data['center_x']:.6f} {yolo_data['center_y']:.6f} "
                       f"{yolo_data['width']:.6f} {yolo_data['height']:.6f}\n")
            
            # Add to metadata
            consolidated_data.append({
                'filename': new_filename,
                'has_crack': True,
                'crack_type': crack_type,
                'severity': severity,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'source_image': img_path.name
            })
    
    return consolidated_data


def generate_no_crack_images(brick_images_dir: Path, target_dir: Path, num_images: int = 50):
    """Generate no-crack images from brick images"""
    
    # Find brick images
    brick_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        brick_images.extend(list(brick_images_dir.glob(ext)))
        brick_images.extend(list(brick_images_dir.glob(ext.upper())))
    
    if not brick_images:
        print(f"⚠️ No brick images found in {brick_images_dir}")
        return []
    
    print(f"📂 Found {len(brick_images)} brick images")
    
    no_crack_data = []
    
    for i in tqdm(range(num_images), desc="Generating no-crack images"):
        # Select random brick image
        brick_img_path = random.choice(brick_images)
        
        # Load image
        img = cv2.imread(str(brick_img_path))
        if img is None:
            continue
        
        # Apply simple augmentation (brightness/contrast)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        
        img_float = img.astype(np.float32)
        img_float = img_float * contrast + (brightness - 1) * 128
        img_float = np.clip(img_float, 0, 255)
        augmented_img = img_float.astype(np.uint8)
        
        # Create filename
        filename = f"no_crack_{i:03d}.jpg"
        
        # Save image
        target_img_path = target_dir / "images" / filename
        cv2.imwrite(str(target_img_path), augmented_img)
        
        # Create empty annotation
        target_ann_path = target_dir / "annotations" / filename.replace('.jpg', '.txt')
        with open(target_ann_path, 'w') as f:
            pass  # Empty file
        
        # Add to metadata
        no_crack_data.append({
            'filename': filename,
            'has_crack': False,
            'crack_type': 'none',
            'severity': 'none',
            'bbox_x1': None,
            'bbox_y1': None,
            'bbox_x2': None,
            'bbox_y2': None,
            'source_image': brick_img_path.name
        })
    
    return no_crack_data


def create_final_test_dataset():
    """Create the final consolidated test dataset"""
    
    # Paths
    benchmark_dir = Path(__file__).parent
    source_dir = benchmark_dir / "output"
    target_dir = benchmark_dir / "test_data"
    brick_images_dir = benchmark_dir.parent / "Dataset_Generation" / "DATA_bricks"
    
    # Create target directories
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (target_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    print("🏗️ Creating Final Test Dataset")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Consolidate crack data
    print("\n📊 Step 1: Consolidating crack data...")
    crack_data = consolidate_crack_data(source_dir, target_dir, max_per_type=40)
    
    # 2. Generate no-crack images
    print("\n📊 Step 2: Generating no-crack images...")
    if brick_images_dir.exists():
        no_crack_data = generate_no_crack_images(brick_images_dir, target_dir, num_images=50)
    else:
        print(f"⚠️ Brick images directory not found: {brick_images_dir}")
        print("   Skipping no-crack image generation")
        no_crack_data = []
    
    # 3. Combine all data
    all_data = crack_data + no_crack_data
    
    # 4. Create ground truth CSV
    print("\n📊 Step 3: Creating ground truth file...")
    df = pd.DataFrame(all_data)
    ground_truth_path = target_dir / "metadata" / "ground_truth.csv"
    df.to_csv(ground_truth_path, index=False)
    
    # 5. Create dataset summary
    summary = {
        'total_images': len(all_data),
        'crack_images': len(crack_data),
        'no_crack_images': len(no_crack_data),
        'crack_distribution': {},
        'severity_distribution': {},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Calculate distributions
    for item in all_data:
        crack_type = item['crack_type']
        severity = item['severity']
        
        summary['crack_distribution'][crack_type] = \
            summary['crack_distribution'].get(crack_type, 0) + 1
        summary['severity_distribution'][severity] = \
            summary['severity_distribution'].get(severity, 0) + 1
    
    # Save summary
    summary_path = target_dir / "metadata" / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 6. Create dataset info file for benchmark scripts
    dataset_info = {
        'dataset_name': 'Crack Analysis Benchmark Dataset',
        'version': '1.0',
        'description': 'Consolidated dataset for crack detection, segmentation, and classification benchmarking',
        'total_images': len(all_data),
        'classes': ['crack'],
        'crack_types': ['vertical', 'horizontal', 'diagonal', 'step', 'none'],
        'annotation_format': 'yolo',
        'image_format': 'jpg',
        'created': pd.Timestamp.now().isoformat()
    }
    
    info_path = target_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n✅ Final Test Dataset Created Successfully!")
    print("=" * 50)
    print(f"📁 Location: {target_dir}")
    print(f"📊 Total Images: {len(all_data)}")
    print(f"📊 Crack Images: {len(crack_data)}")
    print(f"📊 No-crack Images: {len(no_crack_data)}")
    print(f"📄 Ground Truth: {ground_truth_path}")
    print(f"📄 Summary: {summary_path}")
    print("\nCrack Distribution:")
    for crack_type, count in summary['crack_distribution'].items():
        print(f"  {crack_type}: {count}")
    print("\nSeverity Distribution:")
    for severity, count in summary['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    print(f"\n🚀 Dataset ready for:")
    print(f"   - benchmark_pipeline.py --test-dir {target_dir}")
    print(f"   - pipeline_evaluation.py --test-dir {target_dir}")
    
    return target_dir


if __name__ == "__main__":
    final_dataset_path = create_final_test_dataset()
