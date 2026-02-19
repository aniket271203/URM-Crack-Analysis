"""
Image augmentation utilities for crack generation.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import config

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image (numpy.ndarray): Original image
        brightness (float): Brightness adjustment factor (>1 for brighter)
        contrast (float): Contrast adjustment factor (>1 for more contrast)
        
    Returns:
        numpy.ndarray: Adjusted image
    """
    # Convert to PIL image for easier adjustment
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    
    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return result

def add_noise(image, scale=10):
    """
    Add random noise to an image.
    
    Args:
        image (numpy.ndarray): Original image
        scale (int): Scale of the noise (higher = more noise)
        
    Returns:
        numpy.ndarray: Image with added noise
    """
    noise = np.random.normal(0, scale, image.shape).astype(np.int16)
    result = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result

def apply_blur(image, kernel_size=3):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (numpy.ndarray): Original image
        kernel_size (int): Size of the blur kernel
        
    Returns:
        numpy.ndarray: Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_motion_blur(image, kernel_size=15, angle=45):
    """
    Apply motion blur to an image.
    
    Args:
        image (numpy.ndarray): Original image
        kernel_size (int): Size of the blur kernel (must be odd)
        angle (float): Angle of the motion blur in degrees
        
    Returns:
        numpy.ndarray: Motion blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd
        
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1
    kernel = cv2.warpAffine(
        kernel, 
        cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0),
        (kernel_size, kernel_size)
    )
    kernel = kernel / np.sum(kernel)
    
    # Apply the kernel
    result = cv2.filter2D(image, -1, kernel)
    
    return result

def apply_perspective_transform(image, bbox=None, strength=0.1):
    """
    Apply a perspective transform to the image.
    
    Args:
        image (numpy.ndarray): Original image
        bbox (list, optional): Bounding box [x_min, y_min, x_max, y_max] to transform as well
        strength (float): Strength of the perspective distortion
        
    Returns:
        tuple: (Transformed image, Transformed bounding box)
    """
    height, width = image.shape[:2]
    
    # Define the 4 source points (corners of the image)
    src_pts = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ])
    
    # Define the 4 destination points with random offsets
    max_offset = int(min(width, height) * strength)
    dst_pts = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [width - 1 - random.randint(0, max_offset), random.randint(0, max_offset)],
        [random.randint(0, max_offset), height - 1 - random.randint(0, max_offset)],
        [width - 1 - random.randint(0, max_offset), height - 1 - random.randint(0, max_offset)]
    ])
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transform
    result = cv2.warpPerspective(image, M, (width, height))
    
    # Transform the bounding box if provided
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        
        # Convert the bounding box to corner points
        bbox_corners = np.array([
            [x_min, y_min, 1],
            [x_max, y_min, 1],
            [x_min, y_max, 1],
            [x_max, y_max, 1]
        ])
        
        # Transform each corner point
        transformed_corners = []
        for corner in bbox_corners:
            point = np.dot(M, corner)
            # Normalize by the third coordinate
            point = point / point[2]
            transformed_corners.append(point[:2])
        
        transformed_corners = np.array(transformed_corners)
        
        # Get the min/max coordinates of the transformed corners
        new_x_min = max(0, int(np.min(transformed_corners[:, 0])))
        new_y_min = max(0, int(np.min(transformed_corners[:, 1])))
        new_x_max = min(width - 1, int(np.max(transformed_corners[:, 0])))
        new_y_max = min(height - 1, int(np.max(transformed_corners[:, 1])))
        
        new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
        return result, new_bbox
    
    return result, None

def apply_random_lighting(image):
    """
    Apply random lighting conditions to an image.
    
    Args:
        image (numpy.ndarray): Original image
        
    Returns:
        numpy.ndarray: Image with adjusted lighting
    """
    brightness = random.uniform(config.BRIGHTNESS_RANGE[0], config.BRIGHTNESS_RANGE[1])
    contrast = random.uniform(config.CONTRAST_RANGE[0], config.CONTRAST_RANGE[1])
    
    return adjust_brightness_contrast(image, brightness, contrast)

def add_shadow(image):
    """
    Add a random shadow to an image.
    
    Args:
        image (numpy.ndarray): Original image
        
    Returns:
        numpy.ndarray: Image with added shadow
    """
    height, width = image.shape[:2]
    
    # Create a blank shadow mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Generate random shadow shape
    shadow_type = random.choice(['linear', 'ellipse', 'polygon'])
    
    if shadow_type == 'linear':
        # Linear gradient shadow
        direction = random.choice(['left', 'right', 'top', 'bottom'])
        
        if direction in ['left', 'right']:
            for x in range(width):
                opacity = x / width if direction == 'left' else (width - x) / width
                mask[:, x] = int(100 * opacity)
        else:  # top or bottom
            for y in range(height):
                opacity = y / height if direction == 'top' else (height - y) / height
                mask[y, :] = int(100 * opacity)
                
    elif shadow_type == 'ellipse':
        # Elliptical shadow
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        axes_length = (random.randint(width // 4, width // 2), random.randint(height // 4, height // 2))
        angle = random.randint(0, 180)
        
        cv2.ellipse(mask, (center_x, center_y), axes_length, angle, 0, 360, 100, -1)
        
    else:  # polygon
        # Polygonal shadow
        num_points = random.randint(3, 6)
        points = []
        for _ in range(num_points):
            points.append([random.randint(0, width), random.randint(0, height)])
        
        points = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, points, 100)
    
    # Apply the shadow with random darkness
    darkness = random.uniform(0.6, 0.9)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Convert to 3 channel
    shadow = np.stack([mask, mask, mask], axis=-1) / 255.0
    
    # Apply shadow to image
    result = np.clip(image * (1 - (1 - darkness) * shadow), 0, 255).astype(np.uint8)
    
    return result

def apply_random_augmentation(image, bbox=None):
    """
    Apply a random combination of augmentations to an image.
    
    Args:
        image (numpy.ndarray): Original image
        bbox (list, optional): Bounding box [x_min, y_min, x_max, y_max] to transform as well
        
    Returns:
        tuple: (Augmented image, Transformed bounding box)
    """
    result = image.copy()
    transformed_bbox = bbox
    
    # Apply lighting adjustment
    if random.random() < 0.7:
        result = apply_random_lighting(result)
    
    # Add shadow
    if random.random() < 0.4:
        result = add_shadow(result)
    
    # Apply blur
    if random.random() < config.BLUR_PROBABILITY:
        kernel_size = random.choice([3, 5, 7])
        if random.random() < 0.3:
            # Motion blur
            result = apply_motion_blur(result, kernel_size, random.randint(0, 180))
        else:
            # Gaussian blur
            result = apply_blur(result, kernel_size)
    
    # Add noise
    if random.random() < config.NOISE_PROBABILITY:
        noise_scale = random.randint(config.NOISE_RANGE[0], config.NOISE_RANGE[1])
        result = add_noise(result, noise_scale)
    
    # Apply rotation
    if random.random() < 0.3 and bbox is not None:
        angle = random.randint(config.ROTATION_RANGE[0], config.ROTATION_RANGE[1])
        height, width = result.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (width, height))
        
        # Transform the bounding box
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            
            # Get the four corners of the bounding box
            corners = np.array([
                [x_min, y_min, 1],
                [x_max, y_min, 1],
                [x_min, y_max, 1],
                [x_max, y_max, 1]
            ])
            
            # Transform each corner
            transformed_corners = []
            for corner in corners:
                transformed_corner = np.dot(M, corner)
                transformed_corners.append(transformed_corner[:2])
            
            transformed_corners = np.array(transformed_corners)
            
            # Find the new bounding box coordinates
            new_x_min = max(0, int(np.min(transformed_corners[:, 0])))
            new_y_min = max(0, int(np.min(transformed_corners[:, 1])))
            new_x_max = min(width - 1, int(np.max(transformed_corners[:, 0])))
            new_y_max = min(height - 1, int(np.max(transformed_corners[:, 1])))
            
            transformed_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
    
    # Apply perspective transform
    if random.random() < config.PERSPECTIVE_PROBABILITY and bbox is not None:
        result, transformed_bbox = apply_perspective_transform(result, transformed_bbox, 0.1)
    
    return result, transformed_bbox
