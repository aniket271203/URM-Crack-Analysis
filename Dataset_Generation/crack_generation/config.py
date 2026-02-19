"""
Configuration settings for crack generation and augmentation.
"""

import os

# Directory settings
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DATA_bricks'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
OUTPUT_ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, 'annotations')

# Crack parameters
CRACK_TYPES = ['vertical', 'horizontal', 'diagonal', 'step']
CRACK_COUNTS = {
    'vertical': 20,
    'horizontal': 20,
    'diagonal': 20,
    'step': 20,
    # 'x': 15,
    # 'radial': 15
}

# Procedural crack parameters
CRACK_THICKNESS_MIN = 2
CRACK_THICKNESS_MAX = 8
CRACK_JAGGEDNESS_MIN = 1
CRACK_JAGGEDNESS_MAX = 4
BRANCH_PROBABILITY = 0.3
BRANCH_MIN_LENGTH = 10
BRANCH_MAX_LENGTH = 30

# Texture overlay parameters
CRACK_TEXTURES_DIR = os.path.join(os.path.dirname(__file__), 'textures')
ALPHA_MIN = 0.6
ALPHA_MAX = 0.9

# Augmentation parameters
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
BLUR_PROBABILITY = 0.5
BLUR_RANGE = (1, 3)
NOISE_PROBABILITY = 0.7
NOISE_RANGE = (5, 15)
ROTATION_RANGE = (-15, 15)
PERSPECTIVE_PROBABILITY = 0.3

# Annotation settings
ANNOTATION_FORMATS = ['yolo', 'coco']  # 'yolo', 'coco', 'pascal_voc'
DEFAULT_ANNOTATION_FORMAT = 'yolo'  # Changed to YOLO to avoid JSON corruption issues

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
