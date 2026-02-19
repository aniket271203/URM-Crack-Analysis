"""
Texture overlay based crack generation techniques.
"""

import cv2
import numpy as np
import random
import os
from PIL import Image, ImageEnhance, ImageOps
import config

def create_crack_texture(width, height, crack_type='random'):
    """
    Create a crack texture of specified size.
    
    Args:
        width (int): Width of the texture
        height (int): Height of the texture
        crack_type (str): Type of crack texture to generate
        
    Returns:
        numpy.ndarray: Grayscale crack texture
    """
    # Create a black image
    texture = np.zeros((height, width), dtype=np.uint8)
    
    if crack_type == 'random':
        crack_type = random.choice(['vertical', 'horizontal', 'diagonal', 'branching'])
    
    if crack_type == 'vertical':
        # Draw a vertical crack
        x = random.randint(width // 3, 2 * width // 3)
        thickness = random.randint(2, 5)
        cv2.line(texture, (x, 0), (x, height), 255, thickness)
        
        # Add some jitter
        for i in range(0, height, 10):
            offset = random.randint(-10, 10)
            length = random.randint(5, 20)
            cv2.line(texture, (x + offset, i), (x + offset, i + length), 255, random.randint(1, 3))
            
    elif crack_type == 'horizontal':
        # Draw a horizontal crack
        y = random.randint(height // 3, 2 * height // 3)
        thickness = random.randint(2, 5)
        cv2.line(texture, (0, y), (width, y), 255, thickness)
        
        # Add some jitter
        for i in range(0, width, 10):
            offset = random.randint(-10, 10)
            length = random.randint(5, 20)
            cv2.line(texture, (i, y + offset), (i + length, y + offset), 255, random.randint(1, 3))
            
    elif crack_type == 'diagonal':
        # Draw a diagonal crack
        thickness = random.randint(2, 5)
        if random.choice([True, False]):
            cv2.line(texture, (0, 0), (width, height), 255, thickness)
        else:
            cv2.line(texture, (width, 0), (0, height), 255, thickness)
            
        # Add some jitter
        for _ in range(5):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-20, 20)
            y2 = y1 + random.randint(-20, 20)
            cv2.line(texture, (x1, y1), (x2, y2), 255, random.randint(1, 2))
            
    elif crack_type == 'branching':
        # Create branching cracks
        center_x = width // 2
        center_y = height // 2
        
        for _ in range(random.randint(3, 7)):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(height // 4, height // 2)
            end_x = int(center_x + length * np.cos(angle))
            end_y = int(center_y + length * np.sin(angle))
            cv2.line(texture, (center_x, center_y), (end_x, end_y), 255, random.randint(2, 4))
            
            # Add branches
            branch_count = random.randint(1, 3)
            for _ in range(branch_count):
                branch_start_t = random.uniform(0.3, 0.8)
                branch_x = int(center_x + branch_start_t * (end_x - center_x))
                branch_y = int(center_y + branch_start_t * (end_y - center_y))
                
                branch_angle = angle + random.uniform(-np.pi/4, np.pi/4)
                branch_length = random.randint(10, 30)
                branch_end_x = int(branch_x + branch_length * np.cos(branch_angle))
                branch_end_y = int(branch_y + branch_length * np.sin(branch_angle))
                
                cv2.line(texture, (branch_x, branch_y), (branch_end_x, branch_end_y), 255, random.randint(1, 3))
    
    # Apply Gaussian blur to make it more natural
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    
    return texture

def apply_texture_overlay(image, crack_type='random', alpha=0.7, intensity=0.9, rotate=False):
    """
    Apply a crack texture overlay to an image.
    
    Args:
        image (numpy.ndarray): Original image
        crack_type (str): Type of crack texture to apply
        alpha (float): Blending factor (0-1)
        intensity (float): Intensity of the crack (0-1)
        rotate (bool): Whether to rotate the texture
        
    Returns:
        tuple: (Modified image, Bounding box of the crack [x_min, y_min, x_max, y_max])
    """
    height, width = image.shape[:2]
    
    # Create or load crack texture
    texture = create_crack_texture(width, height, crack_type)
    
    # Optionally rotate the texture
    if rotate:
        angle = random.randint(-90, 90)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        texture = cv2.warpAffine(texture, M, (width, height))
    
    # Adjust intensity
    texture = cv2.multiply(texture, intensity)
    
    # Find bounding box of the crack
    contours, _ = cv2.findContours(texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find combined bounding box of all contours
        x_min, y_min = width, height
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
    else:
        # If no contours found, use the whole image (shouldn't happen)
        x_min, y_min = 0, 0
        x_max, y_max = width, height
    
    # Invert the texture so cracks are dark (black)
    texture = 255 - texture
    
    # Convert texture to 3-channel
    texture_3channel = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    
    # Blend the texture with the original image
    result = cv2.addWeighted(image, 1.0, texture_3channel, alpha, 0)
    
    return result, [x_min, y_min, x_max, y_max]

def load_texture_from_file(file_path, width=None, height=None):
    """
    Load a crack texture from a file and resize if needed.
    
    Args:
        file_path (str): Path to the texture file
        width (int, optional): Width to resize to
        height (int, optional): Height to resize to
        
    Returns:
        numpy.ndarray: Loaded texture
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Texture file not found: {file_path}")
    
    # Load the texture using PIL
    texture = Image.open(file_path).convert('L')  # Convert to grayscale
    
    # Resize if dimensions provided
    if width is not None and height is not None:
        texture = texture.resize((width, height), Image.LANCZOS)
    
    # Convert to numpy array
    texture = np.array(texture)
    
    return texture

def create_texture_directory():
    """
    Create a directory with procedurally generated crack textures.
    """
    os.makedirs(config.CRACK_TEXTURES_DIR, exist_ok=True)
    
    # Generate different types of crack textures
    texture_types = ['vertical', 'horizontal', 'diagonal', 'branching']
    texture_counts = {t: 5 for t in texture_types}
    
    for texture_type in texture_types:
        for i in range(texture_counts[texture_type]):
            texture = create_crack_texture(512, 512, texture_type)
            file_path = os.path.join(config.CRACK_TEXTURES_DIR, f"{texture_type}_crack_{i}.png")
            cv2.imwrite(file_path, texture)
            
    print(f"Generated {sum(texture_counts.values())} crack textures in {config.CRACK_TEXTURES_DIR}")

def apply_random_texture_overlay(image):
    """
    Apply a randomly generated texture overlay to create a crack.
    
    Args:
        image (numpy.ndarray): Original image
        
    Returns:
        tuple: (Modified image, Crack type, Bounding box)
    """
    # Select random parameters
    crack_type = random.choice(['vertical', 'horizontal', 'diagonal', 'branching'])
    alpha = random.uniform(config.ALPHA_MIN, config.ALPHA_MAX)
    intensity = random.uniform(0.7, 1.0)
    rotate = random.choice([True, False])
    
    # Apply texture overlay
    result_image, bbox = apply_texture_overlay(image, crack_type, alpha, intensity, rotate)
    
    return result_image, crack_type, bbox
