#!/usr/bin/env python3
"""
Crack Detection Inference Script
Uses trained YOLOv8 model to detect cracks in images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

class CrackDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the crack detector
        
        Args:
            model_path (str): Path to the trained YOLO model (.pt file)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Model info
        print(f"Model loaded successfully!")
        print(f"Model classes: {self.model.names}")
    
    def detect_cracks(self, image_path, save_results=True, output_dir="results"):
        """
        Detect cracks in a single image
        
        Args:
            image_path (str): Path to the input image
            save_results (bool): Whether to save the annotated results
            output_dir (str): Directory to save results
            
        Returns:
            dict: Detection results with bounding boxes, confidence scores, etc.
        """
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        # Extract detection information
        detections = []
        result = results[0]  # Get first (and only) result
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class indices
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box
                class_name = self.model.names[int(cls)]
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': int(cls)
                }
                detections.append(detection)
        
        # Save results if requested
        if save_results:
            self._save_annotated_image(image_path, detections, output_dir)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_cracks': len(detections),
            'image_shape': image.shape
        }
    
    def _save_annotated_image(self, image_path, detections, output_dir):
        """Save image with bounding boxes drawn"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            ax.text(
                x1, y1 - 10, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                color='white', fontsize=10, weight='bold'
            )
        
        ax.set_title(f"Crack Detection Results - Found {len(detections)} crack(s)")
        ax.axis('off')
        
        # Save annotated image
        input_filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{input_filename}_detected.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Annotated image saved to: {output_path}")
    
    def process_directory(self, input_dir, output_dir="results"):
        """
        Process all images in a directory
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save results
            
        Returns:
            list: List of detection results for all images
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all images in directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        all_results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing image {i}/{len(image_files)}: {image_path.name}")
            try:
                result = self.detect_cracks(str(image_path), save_results=True, output_dir=output_dir)
                all_results.append(result)
                print(f"Found {result['num_cracks']} crack(s)")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return all_results
    
    def print_summary(self, results):
        """Print summary of detection results"""
        if not results:
            print("No results to summarize")
            return
        
        total_images = len(results)
        total_cracks = sum(r['num_cracks'] for r in results)
        images_with_cracks = sum(1 for r in results if r['num_cracks'] > 0)
        
        print("\n" + "="*50)
        print("CRACK DETECTION SUMMARY")
        print("="*50)
        print(f"Total images processed: {total_images}")
        print(f"Images with cracks detected: {images_with_cracks}")
        print(f"Images without cracks: {total_images - images_with_cracks}")
        print(f"Total cracks detected: {total_cracks}")
        print(f"Average cracks per image: {total_cracks/total_images:.2f}")
        
        if images_with_cracks > 0:
            avg_cracks_in_positive = total_cracks / images_with_cracks
            print(f"Average cracks per positive image: {avg_cracks_in_positive:.2f}")
        
        print("\nDetailed Results:")
        print("-" * 30)
        for result in results:
            filename = Path(result['image_path']).name
            print(f"{filename}: {result['num_cracks']} crack(s)")


def main():
    parser = argparse.ArgumentParser(description="Crack Detection using trained YOLOv8 model")
    parser.add_argument("--model", type=str, default="weights/best.pt", 
                       help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory containing images")
    parser.add_argument("--output", type=str, default="inference_results",
                       help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS (0-1)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save annotated images")
    
    args = parser.parse_args()
    
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        return
    
    # Initialize detector
    try:
        detector = CrackDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process input
    save_results = not args.no_save
    
    if os.path.isfile(args.input):
        # Single image
        print(f"\nProcessing single image: {args.input}")
        try:
            result = detector.detect_cracks(args.input, save_results=save_results, output_dir=args.output)
            detector.print_summary([result])
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"\nProcessing directory: {args.input}")
        try:
            results = detector.process_directory(args.input, output_dir=args.output)
            detector.print_summary(results)
        except Exception as e:
            print(f"Error processing directory: {e}")
    
    else:
        print(f"Error: Input must be a file or directory: {args.input}")


if __name__ == "__main__":
    # Example usage when run directly
    print("Crack Detection Inference Script")
    print("Usage examples:")
    print("  Single image: python crack_detection_inference.py --input /path/to/image.jpg")
    print("  Directory:    python crack_detection_inference.py --input /path/to/images/")
    print("  Custom model: python crack_detection_inference.py --model custom_model.pt --input image.jpg")
    print("  Adjust conf:  python crack_detection_inference.py --input image.jpg --conf 0.5")
    print()
    
    main()
