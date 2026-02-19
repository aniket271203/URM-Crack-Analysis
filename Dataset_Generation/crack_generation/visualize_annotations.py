"""
Script to visualize images with their bounding box annotations.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random

def read_yolo_annotation(annotation_path, image_width, image_height):
    """
    Read a YOLO format annotation file and convert to absolute coordinates.
    
    Args:
        annotation_path (str): Path to the annotation file
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        list: List of bounding boxes in format [class_id, x_min, y_min, x_max, y_max]
    """
    boxes = []
    
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found: {annotation_path}")
        return boxes
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = line.strip().split()
        if len(data) == 5:
            class_id = int(data[0])
            # YOLO format: center_x, center_y, width, height (normalized)
            center_x = float(data[1]) * image_width
            center_y = float(data[2]) * image_height
            width = float(data[3]) * image_width
            height = float(data[4]) * image_height
            
            # Convert to absolute coordinates [x_min, y_min, x_max, y_max]
            x_min = int(center_x - width / 2)
            y_min = int(center_y - height / 2)
            x_max = int(center_x + width / 2)
            y_max = int(center_y + height / 2)
            
            boxes.append([class_id, x_min, y_min, x_max, y_max])
    
    return boxes

def visualize_image_with_bbox(image, boxes, class_names=None, save_path=None, show=True):
    """
    Visualize an image with bounding boxes.
    
    Args:
        image (numpy.ndarray): Image to visualize
        boxes (list): List of bounding boxes in format [class_id, x_min, y_min, x_max, y_max]
        class_names (list): List of class names
        save_path (str): Path to save the visualization
        show (bool): Whether to display the image
    """
    if class_names is None:
        class_names = ["Crack"]
    
    # Make a copy of the image to draw on
    vis_image = image.copy()
    
    # Generate random colors for each class
    np.random.seed(42)  # for reproducibility
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8).tolist()
    
    # Draw each box
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        
        # Ensure class_id is valid
        if class_id >= len(colors):
            class_id = 0
        
        color = tuple([int(c) for c in colors[class_id]])
        
        # Draw the rectangle
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add class label
        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        cv2.putText(vis_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_image)
    
    if show:
        # Convert from BGR to RGB for matplotlib
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return vis_image

def visualize_dataset(images_dir, annotations_dir, output_dir=None, num_samples=10, random_samples=True, class_names=None):
    """
    Visualize a dataset with annotations.
    
    Args:
        images_dir (str): Directory containing images
        annotations_dir (str): Directory containing annotations
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
        random_samples (bool): Whether to select random samples
        class_names (list): List of class names
    """
    # Get all image files
    image_files = glob(os.path.join(images_dir, '*.jpg')) + \
                  glob(os.path.join(images_dir, '*.jpeg')) + \
                  glob(os.path.join(images_dir, '*.png'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Select samples
    if random_samples:
        if num_samples > len(image_files):
            num_samples = len(image_files)
            print(f"Only {num_samples} images available, visualizing all")
        
        selected_files = random.sample(image_files, num_samples)
    else:
        selected_files = image_files[:num_samples]
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each selected file
    for image_path in selected_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        image_height, image_width = image.shape[:2]
        
        # Find corresponding annotation file
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        annotation_path = os.path.join(annotations_dir, f"{image_name}.txt")
        
        # Read annotations
        boxes = read_yolo_annotation(annotation_path, image_width, image_height)
        
        if not boxes:
            print(f"No annotations found for {image_filename}")
            continue
        
        # Visualize
        if output_dir:
            save_path = os.path.join(output_dir, f"vis_{image_filename}")
        else:
            save_path = None
        
        visualize_image_with_bbox(image, boxes, class_names, save_path, show=(output_dir is None))
        
        if output_dir:
            print(f"Saved visualization for {image_filename} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset with bounding box annotations")
    parser.add_argument("--images", default="combined_output/vertical/images", help="Directory containing images")
    parser.add_argument("--annotations", default="combined_output/vertical/annotations", help="Directory containing annotations")
    parser.add_argument("--output", default="output/visualizations", help="Directory to save visualizations")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--random", action="store_true", help="Select random samples")
    parser.add_argument("--no_save", action="store_true", help="Don't save visualizations, just display them")
    parser.add_argument("--class_names", default="Crack", help="Comma-separated list of class names")
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = args.class_names.split(',')
    
    output_dir = None if args.no_save else args.output
    
    visualize_dataset(
        args.images,
        args.annotations,
        output_dir,
        args.samples,
        args.random,
        class_names
    )

if __name__ == "__main__":
    main()
