import os
import sys
import cv2
import numpy as np
import torch
from pipeline_orchestrator import SegmentationModel

def main():
    # Configuration
    model_path = "Masking_and_Classification_model/pretrained_net_G.pth"
    image_path = "1.jpg"  # Default test image
    output_path = "segmentation_test_result.jpg"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        # Try to find any jpg/jpeg/png image in the current directory
        import glob
        images = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
        if images:
            image_path = images[0]
            print(f"Using alternative image: {image_path}")
        else:
            print("No images found to test.")
            return

    print(f"Initializing SegmentationModel from {model_path}...")
    try:
        # device='cuda' if available, otherwise 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seg_model = SegmentationModel(model_path, device=device)
        
        print(f"Processing image: {image_path}")
        mask = seg_model.segment(image_path)
        
        # Check output
        print(f"Segmentation complete. Mask shape: {mask.shape}, Range: [{mask.min()}, {mask.max()}]")
        
        # Normalize mask for better visibility if range is small
        if mask.max() < 50:
            print("Low confidence detected. Applying contrast normalization...")
            mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
            # Apply gamma correction to boost mid-tones
            gamma = 0.5
            mask_normalized = np.power(mask_normalized / 255.0, gamma) * 255.0
            mask = mask_normalized.astype(np.uint8)
            print(f"Normalized range: [{mask.min()}, {mask.max()}]")
        
        # Save result
        # Save side-by-side: Original | Mask
        original = cv2.imread(image_path)
        if original is not None:
            # Resize mask to match original if needed (though segment method usually returns mask of same size or similar, 
            # let's stick to simple save of mask for now or overlay)
            
            # Ensure mask is 0-255 uint8
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            
            # Color map the mask for better visibility
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # Resize if dimensions differ (unlikely with this pipeline but good safety)
            if original.shape[:2] != mask_colored.shape[:2]:
                 mask_colored = cv2.resize(mask_colored, (original.shape[1], original.shape[0]))
            
            # Create comparison image
            combined = np.hstack((original, mask_colored))
            cv2.imwrite(output_path, combined)
            print(f"Saved comparison result to {output_path}")
            
            # Also save raw mask
            raw_mask_path = "segmentation_mask_only.jpg"
            cv2.imwrite(raw_mask_path, mask)
            print(f"Saved raw mask to {raw_mask_path}")
            
        else:
            print("Could not load original image for comparison.")
            cv2.imwrite(output_path, mask)
            print(f"Saved mask to {output_path}")

    except Exception as e:
        print(f"Error running segmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
