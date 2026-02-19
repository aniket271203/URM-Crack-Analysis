"""
Bounding box annotation generation and storage utilities.
"""

import os
import json
import numpy as np
import time
import shutil

def convert_to_yolo_format(bbox, image_width, image_height):
    """
    Convert a bounding box from [x_min, y_min, x_max, y_max] to YOLO format.
    
    Args:
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        image_width (int): Width of the image
        image_height (int): Height of the image
        
    Returns:
        tuple: (center_x, center_y, width, height) normalized to 0-1
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center point
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to 0-1
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height
    
    return center_x, center_y, width, height

def save_yolo_annotation(bbox, image_path, output_dir, class_id=0, annotation_filename=None):
    """
    Save a bounding box annotation in YOLO format.
    
    Args:
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        image_path (str): Path to the image
        output_dir (str): Directory to save the annotation
        class_id (int): Class ID for the annotation
        annotation_filename (str): Optional explicit filename (without extension)
        
    Returns:
        str: Path to the saved annotation file
    """
    # Get image dimensions
    image_width = None
    image_height = None
    
    # Try to get dimensions from the image if needed
    if image_width is None or image_height is None:
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            image_height, image_width = image.shape[:2]
        else:
            raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to YOLO format
    center_x, center_y, width, height = convert_to_yolo_format(bbox, image_width, image_height)
    
    # Create annotation file path
    if annotation_filename:
        # Use provided filename
        annotation_filename_with_ext = f"{annotation_filename}.txt"
    else:
        # Use image filename
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        annotation_filename_with_ext = f"{image_name}.txt"
    
    annotation_path = os.path.join(output_dir, annotation_filename_with_ext)
    
    # Write to file with error handling
    try:
        with open(annotation_path, 'w') as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Verify the file was written
        if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:
            return annotation_path
        else:
            raise IOError(f"Annotation file was not written properly: {annotation_path}")
            
    except Exception as e:
        print(f"Error writing YOLO annotation: {e}")
        raise

def save_coco_annotation(image_id, image_path, bbox, output_file, category_id=1, category_name="crack"):
    """
    Add an annotation to a COCO format JSON file with atomic write protection.
    
    Args:
        image_id (int): Image ID
        image_path (str): Path to the image
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        output_file (str): Path to the output JSON file
        category_id (int): Category ID
        category_name (str): Category name
        
    Returns:
        None
    """
    import cv2
    import tempfile
    import shutil
    import fcntl
    import time
    
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_height, image_width = image.shape[:2]
    
    # Calculate COCO format bbox [x, y, width, height]
    x_min, y_min, x_max, y_max = bbox
    coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    
    # Calculate area
    area = (x_max - x_min) * (y_max - y_min)
    
    # Create a unique annotation ID using timestamp and image_id
    annotation_id = int(time.time() * 1000000) + image_id
    
    # Create annotation entry
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": coco_bbox,
        "area": area,
        "iscrowd": 0
    }
    
    # Use file locking to prevent race conditions
    lock_file = output_file + ".lock"
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Try to acquire lock
            with open(lock_file, 'w') as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Create backup if file exists and is valid
                if os.path.exists(output_file):
                    is_valid, error_msg = validate_coco_json(output_file)
                    if is_valid:
                        backup_coco_json(output_file)
                    else:
                        print(f"Warning: Existing COCO file is corrupted ({error_msg}), creating new one")
                
                # Load existing COCO JSON if it exists, or create new structure
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            coco_data = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read existing COCO file, creating new one: {e}")
                        coco_data = create_empty_coco_structure(category_id, category_name)
                else:
                    coco_data = create_empty_coco_structure(category_id, category_name)
                
                # Check if image is already in the dataset
                image_exists = False
                for img in coco_data["images"]:
                    if img["id"] == image_id:
                        image_exists = True
                        break
                
                if not image_exists:
                    # Add image information
                    image_info = {
                        "id": image_id,
                        "width": image_width,
                        "height": image_height,
                        "file_name": os.path.basename(image_path),
                        "license": 1
                    }
                    coco_data["images"].append(image_info)
                
                # Add annotation
                coco_data["annotations"].append(annotation)
                
                # Use atomic write: write to temporary file first, then move
                temp_file = output_file + ".tmp"
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(coco_data, f, indent=2)
                    
                    # Validate the written file before moving
                    is_valid, error_msg = validate_coco_json(temp_file)
                    if not is_valid:
                        raise ValueError(f"Generated JSON file is invalid: {error_msg}")
                    
                    # Atomically replace the original file
                    shutil.move(temp_file, output_file)
                    return  # Success!
                    
                except Exception as e:
                    # Clean up temp file if something went wrong
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    raise e
                    
        except (IOError, OSError) as e:
            # Lock acquisition failed or other IO error
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise IOError(f"Could not acquire lock for COCO file after {max_retries} attempts: {e}")
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except:
                pass

def create_empty_coco_structure(category_id=1, category_name="crack"):
    """Create an empty COCO data structure."""
    return {
        "info": {
            "description": "Crack Detection Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "AI Generated"
        },
        "licenses": [{
            "id": 1,
            "name": "N/A",
            "url": "N/A"
        }],
        "categories": [{
            "id": category_id,
            "name": category_name,
            "supercategory": "defect"
        }],
        "images": [],
        "annotations": []
    }

def save_pascal_voc_annotation(image_path, bbox, output_dir, class_name="crack", annotation_filename=None):
    """
    Save an annotation in Pascal VOC XML format.
    
    Args:
        image_path (str): Path to the image
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        output_dir (str): Directory to save the annotation
        class_name (str): Class name
        annotation_filename (str): Optional explicit filename (without extension)
        
    Returns:
        str: Path to the saved annotation file
    """
    import cv2
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_height, image_width = image.shape[:2]
    
    # Determine annotation filename
    if annotation_filename:
        annotation_filename_with_ext = f"{annotation_filename}.xml"
        image_filename = f"{annotation_filename}.jpg"  # Assume jpg for XML reference
    else:
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        annotation_filename_with_ext = f"{image_name}.xml"
    
    # Create XML structure
    annotation = Element('annotation')
    
    folder = SubElement(annotation, 'folder')
    folder.text = os.path.basename(os.path.dirname(image_path))
    
    filename = SubElement(annotation, 'filename')
    filename.text = image_filename
    
    path = SubElement(annotation, 'path')
    path.text = image_path
    
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(image_width)
    height = SubElement(size, 'height')
    height.text = str(image_height)
    depth = SubElement(size, 'depth')
    depth.text = str(3)  # Assuming BGR image
    
    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Add object
    obj = SubElement(annotation, 'object')
    name = SubElement(obj, 'name')
    name.text = class_name
    pose = SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = SubElement(obj, 'truncated')
    truncated.text = '0'
    difficult = SubElement(obj, 'difficult')
    difficult.text = '0'
    
    bndbox = SubElement(obj, 'bndbox')
    xmin = SubElement(bndbox, 'xmin')
    xmin.text = str(int(bbox[0]))
    ymin = SubElement(bndbox, 'ymin')
    ymin.text = str(int(bbox[1]))
    xmax = SubElement(bndbox, 'xmax')
    xmax.text = str(int(bbox[2]))
    ymax = SubElement(bndbox, 'ymax')
    ymax.text = str(int(bbox[3]))
    
    # Convert to pretty XML
    rough_string = tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Save to file
    annotation_path = os.path.join(output_dir, annotation_filename_with_ext)
    
    try:
        with open(annotation_path, 'w') as f:
            f.write(pretty_xml)
        
        # Verify the file was written
        if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:
            return annotation_path
        else:
            raise IOError(f"Annotation file was not written properly: {annotation_path}")
            
    except Exception as e:
        print(f"Error writing Pascal VOC annotation: {e}")
        raise

def save_annotation(image_path, bbox, output_dir, annotation_format="yolo", class_id=0, class_name="crack", annotation_filename=None):
    """
    Save annotation in the specified format.
    
    Args:
        image_path (str): Path to the image
        bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max]
        output_dir (str): Directory to save the annotation
        annotation_format (str): Format of the annotation ("yolo", "coco", or "pascal_voc")
        class_id (int): Class ID
        class_name (str): Class name
        annotation_filename (str): Optional explicit filename (without extension)
        
    Returns:
        str: Path to the saved annotation file
    """
    import time
    import hashlib
    
    os.makedirs(output_dir, exist_ok=True)
    
    if annotation_format.lower() == "yolo":
        return save_yolo_annotation(bbox, image_path, output_dir, class_id, annotation_filename)
    
    elif annotation_format.lower() == "coco":
        # Create a more robust image ID using file path hash and timestamp
        if annotation_filename:
            # Use provided filename to generate consistent ID
            path_hash = hashlib.md5(annotation_filename.encode()).hexdigest()[:8]
        else:
            image_basename = os.path.basename(image_path)
            path_hash = hashlib.md5(image_basename.encode()).hexdigest()[:8]
        
        timestamp = int(time.time() * 1000) % 10000000  # Last 7 digits of timestamp
        image_id = int(path_hash, 16) % 1000000 + timestamp  # Combine hash and timestamp
        
        output_file = os.path.join(output_dir, "annotations.json")
        save_coco_annotation(image_id, image_path, bbox, output_file, class_id, class_name)
        return output_file
    
    elif annotation_format.lower() == "pascal_voc":
        return save_pascal_voc_annotation(image_path, bbox, output_dir, class_name, annotation_filename)
    
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}")

def compute_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1 (list): First bounding box coordinates [x_min, y_min, x_max, y_max]
        bbox2 (list): Second bounding box coordinates [x_min, y_min, x_max, y_max]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Extract coordinates
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    
    # Calculate the area of both boxes
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # Calculate the coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    # Check if the boxes intersect
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0
    
    # Calculate the area of intersection
    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    # Calculate the area of union
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def validate_coco_json(json_path):
    """
    Validate a COCO JSON file to ensure it's not corrupted.
    
    Args:
        json_path (str): Path to the COCO JSON file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if not os.path.exists(json_path):
            return False, "File does not exist"
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['info', 'licenses', 'categories', 'images', 'annotations']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check if lists are actually lists
        for field in ['categories', 'images', 'annotations']:
            if not isinstance(data[field], list):
                return False, f"Field {field} is not a list"
        
        # Basic integrity checks
        image_ids = set(img['id'] for img in data['images'])
        annotation_image_ids = set(ann['image_id'] for ann in data['annotations'])
        
        # Check if all annotation image_ids exist in images
        orphaned_annotations = annotation_image_ids - image_ids
        if orphaned_annotations:
            return False, f"Found annotations with non-existent image IDs: {orphaned_annotations}"
        
        return True, "Valid"
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def backup_coco_json(json_path):
    """
    Create a backup of the COCO JSON file before modification.
    
    Args:
        json_path (str): Path to the COCO JSON file
        
    Returns:
        str: Path to the backup file
    """
    if os.path.exists(json_path):
        backup_path = json_path + f".backup_{int(time.time())}"
        shutil.copy2(json_path, backup_path)
        return backup_path
    return None

def validate_generation(output_image_path, annotation_path, annotation_format):
    """
    Validate that both image and annotation were created successfully.
    
    Args:
        output_image_path (str): Path to the generated image
        annotation_path (str): Path to the generated annotation
        annotation_format (str): Format of the annotation
        
    Returns:
        bool: True if both files exist and are valid
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check if image exists and is valid
    if not os.path.exists(output_image_path):
        logger.error(f"Generated image does not exist: {output_image_path}")
        return False
    
    if os.path.getsize(output_image_path) == 0:
        logger.error(f"Generated image is empty: {output_image_path}")
        return False
    
    # Check if annotation exists and is valid
    if not annotation_path or not os.path.exists(annotation_path):
        logger.error(f"Annotation does not exist: {annotation_path}")
        return False
    
    if os.path.getsize(annotation_path) == 0:
        logger.error(f"Annotation file is empty: {annotation_path}")
        return False
    
    # Format-specific validation
    if annotation_format.lower() == "yolo":
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    logger.error(f"YOLO annotation file is empty: {annotation_path}")
                    return False
                
                # Check if first line has correct format
                parts = lines[0].strip().split()
                if len(parts) != 5:
                    logger.error(f"YOLO annotation has incorrect format: {annotation_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error reading YOLO annotation {annotation_path}: {e}")
            return False
    
    elif annotation_format.lower() == "pascal_voc":
        try:
            import xml.etree.ElementTree as ET
            ET.parse(annotation_path)
        except Exception as e:
            logger.error(f"Error parsing Pascal VOC annotation {annotation_path}: {e}")
            return False
    
    elif annotation_format.lower() == "coco":
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                if not data.get('annotations'):
                    logger.error(f"COCO annotation has no annotations: {annotation_path}")
                    return False
        except Exception as e:
            logger.error(f"Error reading COCO annotation {annotation_path}: {e}")
            return False
    
    return True
