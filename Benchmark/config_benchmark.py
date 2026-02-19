"""
Configuration for crack analysis benchmarking system
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
TEST_DATA_DIR = BASE_DIR / "test_data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset generation settings
BRICK_IMAGES_DIR = ROOT_DIR / "Dataset_Generation" / "DATA_bricks"
CRACK_GENERATION_DIR = ROOT_DIR / "Dataset_Generation" / "crack_generation"

# Test dataset composition (updated to match final consolidated dataset)
TEST_DATASET_CONFIG = {
    'vertical': 40,      # Number of vertical crack images
    'horizontal': 40,    # Number of horizontal crack images  
    'diagonal': 40,      # Number of diagonal crack images
    'step': 40,         # Number of step crack images
    'no_crack': 50,     # Number of images without cracks
}

TOTAL_IMAGES = sum(TEST_DATASET_CONFIG.values())

# Crack generation parameters for benchmarking
CRACK_PARAMS = {
    'thickness_range': (2, 10),      # Pixel thickness range
    'jaggedness_range': (1, 5),      # Crack irregularity
    'length_range': (50, 200),       # Crack length in pixels
    'branch_probability': 0.3,       # Probability of crack branching
    'noise_levels': [0, 5, 10, 15], # Different noise levels for testing
    'blur_levels': [0, 1, 2, 3],    # Different blur levels for testing
}

# Model paths
MODEL_PATHS = {
    'yolo': ROOT_DIR / "Crack_Detection_YOLO" / "crack_yolo_train" / "weights" / "best.pt",
    'segmentation': ROOT_DIR / "Masking_and_Classification_model" / "pretrained_net_G.pth",
    'classification': ROOT_DIR / "Masking_and_Classification_model" / "crack_orientation_classifier.h5",
    'rag_data': ROOT_DIR / "Rag_and_Reasoning" / "crack_analysis_rag" / "data",
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    'confidence_threshold': 0.25,    # YOLO confidence threshold
    'iou_threshold': 0.45,           # YOLO IoU threshold for NMS
    'segmentation_threshold': 0.5,   # Segmentation mask threshold
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'confidence_thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'classification_classes': ['vertical', 'horizontal', 'diagonal', 'step'],
}

# Hardware settings
HARDWARE_CONFIG = {
    'device': 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu',
    'batch_size': 8,
    'num_workers': 4,
}

# Benchmark settings
BENCHMARK_CONFIG = {
    'save_intermediate_results': True,
    'generate_visualizations': True,
    'measure_inference_time': True,
    'measure_memory_usage': True,
    'save_failure_cases': True,
    'detailed_error_analysis': True,
}

# Output settings
OUTPUT_CONFIG = {
    'image_format': 'png',
    'plot_dpi': 300,
    'plot_style': 'seaborn-v0_8',
    'figure_size': (12, 8),
}

# Create necessary directories
for directory in [TEST_DATA_DIR, RESULTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Subdirectories for organized results
RESULT_SUBDIRS = {
    'yolo_evaluation': RESULTS_DIR / "yolo_evaluation",
    'segmentation_evaluation': RESULTS_DIR / "segmentation_evaluation", 
    'classification_evaluation': RESULTS_DIR / "classification_evaluation",
    'pipeline_evaluation': RESULTS_DIR / "pipeline_evaluation",
    'timing_analysis': RESULTS_DIR / "timing_analysis",
    'memory_analysis': RESULTS_DIR / "memory_analysis",
}

for subdir in RESULT_SUBDIRS.values():
    subdir.mkdir(parents=True, exist_ok=True)

# Test data subdirectories
TEST_DATA_SUBDIRS = {
    'images': TEST_DATA_DIR / "images",
    'annotations': TEST_DATA_DIR / "annotations", 
    'metadata': TEST_DATA_DIR / "metadata",
}

for subdir in TEST_DATA_SUBDIRS.values():
    subdir.mkdir(parents=True, exist_ok=True)
