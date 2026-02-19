#!/usr/bin/env python3
"""
Simple test script for crack detection inference
This script provides an easy way to test the crack detection model
"""

import os
import sys
from crack_detection_inference import CrackDetector

def test_crack_detection():
    """Simple test function to demonstrate crack detection"""
    
    # Model path (using the best trained weights)
    model_path = "weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure you have trained the model and the weights file exists.")
        return False
    
    try:
        # Initialize the detector
        print("Initializing crack detector...")
        detector = CrackDetector(
            model_path=model_path,
            conf_threshold=0.25,  # Confidence threshold
            iou_threshold=0.45    # IoU threshold for Non-Maximum Suppression
        )
        
        # Test with validation images from training
        test_images = [
            "val_batch0_pred.jpg",
            "val_batch1_pred.jpg", 
            "val_batch2_pred.jpg"
        ]
        
        print("\nTesting with validation images...")
        results = []
        
        for img_file in test_images:
            if os.path.exists(img_file):
                print(f"\nProcessing: {img_file}")
                try:
                    result = detector.detect_cracks(
                        img_file, 
                        save_results=True, 
                        output_dir="test_results"
                    )
                    results.append(result)
                    print(f"Found {result['num_cracks']} crack(s) in {img_file}")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            else:
                print(f"Test image not found: {img_file}")
        
        if results:
            detector.print_summary(results)
        else:
            print("No images were successfully processed.")
            
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def interactive_test():
    """Interactive test mode where user can input image paths"""
    
    model_path = "weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        detector = CrackDetector(model_path=model_path)
        
        print("\n=== Interactive Crack Detection Test ===")
        print("Enter image paths to test (or 'quit' to exit)")
        
        while True:
            image_path = input("\nEnter image path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
                
            if not image_path:
                continue
                
            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                continue
            
            try:
                result = detector.detect_cracks(
                    image_path, 
                    save_results=True, 
                    output_dir="interactive_results"
                )
                
                print(f"\nResults for {image_path}:")
                print(f"Number of cracks detected: {result['num_cracks']}")
                
                if result['detections']:
                    print("Detections:")
                    for i, detection in enumerate(result['detections'], 1):
                        print(f"  {i}. Class: {detection['class']}, "
                              f"Confidence: {detection['confidence']:.3f}, "
                              f"Bbox: {detection['bbox']}")
                else:
                    print("No cracks detected.")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
    
    except Exception as e:
        print(f"Error initializing detector: {e}")

if __name__ == "__main__":
    print("Crack Detection Test Script")
    print("=" * 40)
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print("✓ Ultralytics YOLO is available")
    except ImportError:
        print("✗ Ultralytics not found. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_test()
        else:
            print("Usage:")
            print("  python test_crack_detection.py          # Run basic test")
            print("  python test_crack_detection.py -i       # Interactive mode")
    else:
        # Run basic test
        success = test_crack_detection()
        if success:
            print("\n✓ Test completed successfully!")
        else:
            print("\n✗ Test failed!")
