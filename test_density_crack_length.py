import os
import sys
import cv2
import numpy as np
import argparse
from skimage.morphology import skeletonize

def get_longest_connected_component(binary_image):
    """
    Finds the largest connected component in a binary image (excluding background).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_image)
    
    # Exclude background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255

def analyze_crack_density_length(image_path, mask_path=None, seg_model_path=None, join_threshold=20, output_path="density_length_result.jpg"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
        
    # Get mask
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    elif seg_model_path and os.path.exists(seg_model_path):
        import torch
        from pipeline_orchestrator import SegmentationModel
        print("Running segmentation model...")
        seg_model = SegmentationModel(seg_model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        mask = seg_model.segment(image_path)
    else:
        print("Error: Must provide either a pre-computed mask or a segmentation model path.")
        return
        
    # Apply density logic
    print("Computing density map to fuse crack segments...")
    blur_sigma = 15.0
    kernel_size = int(blur_sigma * 6) | 1
    
    # 1. Blur to create heat/density
    heatmap = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), blur_sigma)
    
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
        
    # Save the original un-thresholded heatmap (colored)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_out_path = "orginal_density_heatmap.jpg"
    
    # Overlay the heatmap on the original image for context
    alpha = 0.5
    heatmap_display_mask = (heatmap > 10).astype(np.float32)
    heatmap_display_mask = np.stack([heatmap_display_mask] * 3, axis=-1)
    overlay_heatmap = (image.astype(np.float32) * (1 - alpha * heatmap_display_mask) + 
                       heatmap_colored.astype(np.float32) * (alpha * heatmap_display_mask))
    overlay_heatmap = np.clip(overlay_heatmap, 0, 255).astype(np.uint8)
    cv2.imwrite(heatmap_out_path, overlay_heatmap)
    print(f"=> 🖼️ Original heatmap overlay saved to {heatmap_out_path}")
        
    # 2. Threshold to get a solid blob for the main crack paths
    threshold = 10
    heatmap_mask = (heatmap > threshold).astype(np.uint8) * 255
    
    # 3. Join nearby disconnected components using Morphological Closing
    if join_threshold > 0:
        print(f"Joining components separated by up to {join_threshold} pixels...")
        # An ellipse kernel connects features in all directions smoothly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join_threshold, join_threshold))
        heatmap_mask = cv2.morphologyEx(heatmap_mask, cv2.MORPH_CLOSE, kernel)
        
    print("Extracting all valid connected components...")
    all_blobs = heatmap_mask
    
    # 4. Skeletonize all components
    print("Skeletonizing to find the center paths...")
    blob_bool = all_blobs > 0
    skeleton = skeletonize(blob_bool)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # 5. Measure length in pixels
    pixel_length = np.count_nonzero(skeleton_uint8)
    print(f"\n=> 📏 Measured total crack length (all components): {pixel_length} pixels")
    
    # Visualization: Original Image + Density Overlay + Skeleton
    overlay = image.copy()
    
    # Density blobs as transparent blue region
    blue_mask = np.zeros_like(overlay)
    blue_mask[all_blobs > 0] = [255, 0, 0] # BGR Blue
    cv2.addWeighted(overlay, 1.0, blue_mask, 0.4, 0, overlay)
    
    # Skeleton as solid red line
    # Thicken skeleton slightly for visibility in overlay
    skel_thick = cv2.dilate(skeleton_uint8, np.ones((3,3), np.uint8), iterations=1)
    overlay[skel_thick > 0] = [0, 0, 255] # Red
    
    cv2.imwrite(output_path, overlay)
    print(f"=> 🖼️ Skeleton visualization (All) saved to {output_path}")

    # 6. Extract longest connected component alone
    print("Extracting longest connected component for final output...")
    longest_blob = get_longest_connected_component(all_blobs)
    longest_skel = skeletonize(longest_blob > 0)
    longest_skel_uint8 = (longest_skel * 255).astype(np.uint8)
    longest_length = np.count_nonzero(longest_skel_uint8)
    print(f"\n=> 📏 Measured longest crack length: {longest_length} pixels")
    
    longest_overlay = image.copy()
    longest_blue = np.zeros_like(longest_overlay)
    longest_blue[longest_blob > 0] = [255, 0, 0]
    cv2.addWeighted(longest_overlay, 1.0, longest_blue, 0.4, 0, longest_overlay)
    
    longest_skel_thick = cv2.dilate(longest_skel_uint8, np.ones((3,3), np.uint8), iterations=1)
    longest_overlay[longest_skel_thick > 0] = [0, 0, 255]
    
    longest_out_path = "longest_crack_overlay.jpg"
    cv2.imwrite(longest_out_path, longest_overlay)
    print(f"=> 🖼️ Longest component overlay saved to {longest_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script to calculate crack length in pixels using density maps.")
    parser.add_argument("image", help="Path to the original image")
    parser.add_argument("--mask", help="Path to the pre-computed mask (optional)", default=None)
    parser.add_argument("--seg-model", help="Path to segmentation model", 
                        default="Masking_and_Classification_model/pretrained_net_G.pth")
    parser.add_argument("--join-dist", type=int, default=10,
                        help="Maximum pixel distance to bridge/join disconnected crack segments. (0 to disable)")
    parser.add_argument("--output", help="Output path for the visualization image", default="density_length_result.jpg")
    
    args = parser.parse_args()
    analyze_crack_density_length(args.image, args.mask, args.seg_model, args.join_dist, args.output)
