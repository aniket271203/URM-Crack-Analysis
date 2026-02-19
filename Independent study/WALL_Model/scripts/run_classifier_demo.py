import cv2
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.component_classifier import WallClassifier

def main():
    parser = argparse.ArgumentParser(description="Run Wall Component Classifier on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument("--output", default="output_classified.jpg", help="Path to save the output image.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)
        
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image '{args.image}'.")
        sys.exit(1)
        
    print(f"Processing image: {args.image}")
    
    classifier = WallClassifier()
    
    # 1. Detect Openings & Boundaries (returns dicts now)
    detections = classifier.detect_openings(image)
    print(f"Detected {len(detections)} objects.")
    for d in detections:
        print(f" - Found {d['label']} ({d['score']:.2f}) at {d['bbox']}")

    # 2. Infer Structure
    components = classifier.infer_structure(image.shape, detections)
    print(f"Inferred {len(components)} components.")
    
    for c in components:
        print(f" - {c.id} ({c.type}): {c.description} at {c.bbox}")
        
    # 3. Visualize
    vis_image = classifier.visualize(image, components)
    
    cv2.imwrite(args.output, vis_image)
    print(f"Saved visualization to '{args.output}'")

if __name__ == "__main__":
    main()
