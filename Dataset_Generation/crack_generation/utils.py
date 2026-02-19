"""
Utility functions for crack generation.
"""

import os
import cv2
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from disk.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        numpy.ndarray: Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

def save_image(image, output_path):
    """
    Save an image to disk.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the image
        
    Returns:
        str: Path to the saved image
    """
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    
    cv2.imwrite(output_path, image)
    return output_path

def visualize_image_with_bbox(image, bbox, title="Image with Bounding Box"):
    """
    Visualize an image with a bounding box.
    
    Args:
        image (numpy.ndarray): Image to visualize
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        title (str): Title for the plot
        
    Returns:
        None
    """
    # Convert from BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(rgb_image)
    
    # Extract coordinates
    x_min, y_min, x_max, y_max = bbox
    
    # Create rectangle patch
    width = x_max - x_min
    height = y_max - y_min
    rect = plt.Rectangle((x_min, y_min), width, height, 
                         linewidth=2, edgecolor='r', facecolor='none')
    
    # Add the patch to the axis
    ax.add_patch(rect)
    
    # Set title and show
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_dataset_splits(images_dir, annotations_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        images_dir (str): Directory containing images
        annotations_dir (str): Directory containing annotations
        output_dir (str): Directory to save the splits
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        
    Returns:
        tuple: (train_dir, val_dir, test_dir) paths
    """
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    test_images_dir = os.path.join(output_dir, 'test', 'images')
    
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    test_labels_dir = os.path.join(output_dir, 'test', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    # Get list of all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    
    # Calculate split indices
    num_images = len(image_files)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)
    
    # Split the data
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Copy files to their respective directories
    for files, img_dir, lbl_dir in [
        (train_files, train_images_dir, train_labels_dir),
        (val_files, val_images_dir, val_labels_dir),
        (test_files, test_images_dir, test_labels_dir)
    ]:
        for img_file in files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(img_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding annotation
            img_name = os.path.splitext(img_file)[0]
            ann_file = f"{img_name}.txt"  # Assuming YOLO format
            src_ann = os.path.join(annotations_dir, ann_file)
            if os.path.exists(src_ann):
                dst_ann = os.path.join(lbl_dir, ann_file)
                shutil.copy2(src_ann, dst_ann)
    
    return (
        os.path.join(output_dir, 'train'),
        os.path.join(output_dir, 'val'),
        os.path.join(output_dir, 'test')
    )

def create_data_yaml(output_dir, class_names=["crack"]):
    """
    Create a data.yaml file for YOLOv5 training.
    
    Args:
        output_dir (str): Directory to save the YAML file
        class_names (list): List of class names
        
    Returns:
        str: Path to the created YAML file
    """
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    train_path = os.path.join(output_dir, 'train')
    val_path = os.path.join(output_dir, 'val')
    test_path = os.path.join(output_dir, 'test')
    
    yaml_content = f"""
# Train/val/test paths
train: {train_path}/images
val: {val_path}/images
test: {test_path}/images

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def enhance_crack_visibility(image, bbox, enhancement_factor=1.5):
    """
    Enhance the visibility of cracks within a bounding box.
    
    Args:
        image (numpy.ndarray): Original image
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        enhancement_factor (float): Factor to enhance contrast
        
    Returns:
        numpy.ndarray: Image with enhanced crack visibility
    """
    result = image.copy()
    
    # Extract bbox coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
    
    # Extract the region of interest (ROI)
    roi = result[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale for processing
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_roi)
    
    # Convert back to color
    enhanced_roi = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    # Blend with original using enhancement factor
    blended_roi = cv2.addWeighted(roi, enhancement_factor, enhanced_roi, 1 - enhancement_factor, 0)
    
    # Put the enhanced ROI back into the original image
    result[y_min:y_max, x_min:x_max] = blended_roi
    
    return result

def adjust_bbox_size(bbox, image_shape, margin=0.1):
    """
    Adjust bounding box size with a margin.
    
    Args:
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        image_shape (tuple): Image shape (height, width)
        margin (float): Margin to add around the box (proportion of box size)
        
    Returns:
        list: Adjusted bounding box coordinates
    """
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate current dimensions
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Calculate margins
    x_margin = int(box_width * margin)
    y_margin = int(box_height * margin)
    
    # Adjust coordinates with margins
    new_x_min = max(0, x_min - x_margin)
    new_y_min = max(0, y_min - y_margin)
    new_x_max = min(width, x_max + x_margin)
    new_y_max = min(height, y_max + y_margin)
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def ensure_min_bbox_size(bbox, min_size=20):
    """
    Ensure the bounding box has at least a minimum size.
    
    Args:
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        min_size (int): Minimum size for both width and height
        
    Returns:
        list: Adjusted bounding box coordinates
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate current dimensions
    width = x_max - x_min
    height = y_max - y_min
    
    # Adjust if needed
    if width < min_size:
        # Expand equally on both sides
        expand = (min_size - width) // 2
        x_min -= expand
        x_max += expand
        # Add an extra pixel if needed
        if x_max - x_min < min_size:
            x_max += 1
    
    if height < min_size:
        # Expand equally on both sides
        expand = (min_size - height) // 2
        y_min -= expand
        y_max += expand
        # Add an extra pixel if needed
        if y_max - y_min < min_size:
            y_max += 1
    
    return [x_min, y_min, x_max, y_max]

def save_visualization(image, bbox, output_path, class_id=0, class_names=None):
    """
    Save an image with visualized bounding box.
    
    Args:
        image (numpy.ndarray): Image to visualize
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        output_path (str): Path to save the visualization
        class_id (int): Class ID for the annotation
        class_names (list): List of class names
        
    Returns:
        str: Path to the saved visualization
    """
    if class_names is None:
        class_names = ["crack"]
    
    # Make a copy of the image
    vis_image = image.copy()
    
    # Extract coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
    
    # Generate a color based on class_id
    np.random.seed(class_id)  # Consistent color for the same class
    color = tuple([int(c) for c in np.random.randint(0, 255, size=3)])
    
    # Draw the rectangle
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Add class label
    label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
    cv2.putText(vis_image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, vis_image)
    
    return output_path
