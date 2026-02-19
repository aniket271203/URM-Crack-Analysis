#!/usr/bin/env python3
"""
Brick-Calibrated Crack Measurement System

This script measures crack dimensions using brick pattern recognition for scale calibration.
Instead of using ArUco markers, it detects bricks in the wall image and uses their
known physical dimensions to establish the pixel-to-mm scale factor.

Usage:
    python brick_calibrated_measurement.py <image_path> [options]
    
Examples:
    # Auto-detect bricks with Indian modular brick dimensions
    python brick_calibrated_measurement.py Test/test_98.jpg --brick-type india_modular
    
    # Use custom brick dimensions
    python brick_calibrated_measurement.py Test/test_98.jpg --brick-length 215 --brick-height 65
    
    # Interactive mode - manually select a brick
    python brick_calibrated_measurement.py Test/test_98.jpg --interactive
"""

import argparse
import sys
import os
import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import segmentation model from pipeline
from pipeline_orchestrator import SegmentationModel


# ============================================================================
# STANDARD BRICK DIMENSIONS (in mm)
# ============================================================================
BRICK_STANDARDS = {
    'india_modular': {
        'name': 'Indian Modular Brick',
        'length_mm': 190,
        'height_mm': 90,
        'mortar_mm': 10,
        'description': 'Standard modular brick used in India (190x90x90mm)'
    },
    'india_traditional': {
        'name': 'Indian Traditional Brick', 
        'length_mm': 230,
        'height_mm': 75,
        'mortar_mm': 12,
        'description': 'Traditional brick size in India (230x115x75mm)'
    },
    'uk_standard': {
        'name': 'UK Standard Brick',
        'length_mm': 215,
        'height_mm': 65,
        'mortar_mm': 10,
        'description': 'Standard UK brick (215x102.5x65mm)'
    },
    'us_standard': {
        'name': 'US Standard Brick',
        'length_mm': 194,
        'height_mm': 57,
        'mortar_mm': 10,
        'description': 'Standard US modular brick (194x92x57mm)'
    },
    'europe_nf': {
        'name': 'European NF Brick',
        'length_mm': 240,
        'height_mm': 71,
        'mortar_mm': 10,
        'description': 'European Normalformat brick (240x115x71mm)'
    }
}


@dataclass
class BrickDetectionResult:
    """Results from brick detection"""
    success: bool
    brick_length_px: float = 0.0
    brick_height_px: float = 0.0
    scale_x_mm_per_px: float = 0.0
    scale_y_mm_per_px: float = 0.0
    avg_scale_mm_per_px: float = 0.0
    num_bricks_detected: int = 0
    confidence: float = 0.0
    method: str = ""
    error: str = ""
    debug_info: Dict = None


class BrickDetector:
    """
    Detects bricks in masonry wall images and calculates scale factors.
    
    Uses multiple methods:
    1. Mortar line detection (Hough lines)
    2. Frequency analysis (FFT for periodic brick pattern)
    3. Edge-based segmentation
    """
    
    def __init__(self, brick_length_mm: float = 190, brick_height_mm: float = 90, 
                 mortar_width_mm: float = 10):
        """
        Initialize brick detector with known brick dimensions.
        
        Args:
            brick_length_mm: Brick length in mm (default: 190 for Indian modular)
            brick_height_mm: Brick height in mm (default: 90 for Indian modular)
            mortar_width_mm: Mortar joint width in mm (default: 10)
        """
        self.brick_length_mm = brick_length_mm
        self.brick_height_mm = brick_height_mm
        self.mortar_width_mm = mortar_width_mm
        
        # With mortar, the repeating unit is:
        self.unit_length_mm = brick_length_mm + mortar_width_mm  # horizontal repeat
        self.unit_height_mm = brick_height_mm + mortar_width_mm  # vertical repeat
    
    def detect(self, image: np.ndarray) -> BrickDetectionResult:
        """
        Main detection method - tries multiple approaches.
        Focuses on detecting BRICK HEIGHT (horizontal mortar lines) as it's more consistent.
        
        Args:
            image: Input BGR image
            
        Returns:
            BrickDetectionResult with scale factors
        """
        results = []
        
        # Method 1: Mortar line detection (focuses on horizontal lines for height)
        result1 = self._detect_via_mortar_lines(image)
        if result1.success:
            results.append(result1)
        
        # Method 2: Frequency analysis (focuses on vertical frequency for height)
        result2 = self._detect_via_frequency(image)
        if result2.success:
            results.append(result2)
        
        # Method 3: Edge analysis (backup)
        result3 = self._detect_via_edges(image)
        if result3.success:
            results.append(result3)
        
        # Choose best result based on confidence
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result
        
        return BrickDetectionResult(
            success=False,
            error="Could not detect brick pattern using any method"
        )
    
    def _detect_via_mortar_lines(self, image: np.ndarray) -> BrickDetectionResult:
        """
        Detect bricks by finding HORIZONTAL mortar lines (for brick HEIGHT).
        Brick height is more consistent than length, so we focus on that.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Estimate minimum brick height in pixels (at least 1/20th of image height)
            min_brick_height = h // 20
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
            
            # Focus on HORIZONTAL lines (for brick height measurement)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            
            # Detect horizontal lines
            horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
            horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, kernel_h)
            
            # Find horizontal line positions using projection
            h_projection = np.sum(horizontal, axis=1).astype(float)
            # Smooth the projection
            h_projection = ndimage.gaussian_filter1d(h_projection, sigma=3)
            
            # Find peaks (horizontal mortar lines)
            h_peaks, h_props = find_peaks(h_projection, 
                                          distance=min_brick_height,
                                          prominence=np.max(h_projection) * 0.1)
            
            # Calculate height spacings
            if len(h_peaks) >= 2:
                h_spacings = np.diff(h_peaks)
                # Filter out outliers
                h_spacings = h_spacings[(h_spacings > min_brick_height) & 
                                        (h_spacings < h // 3)]
                if len(h_spacings) > 0:
                    brick_height_px = np.median(h_spacings)
                else:
                    return BrickDetectionResult(success=False, method="mortar_lines",
                                               error="Could not determine consistent brick height")
            else:
                return BrickDetectionResult(success=False, method="mortar_lines",
                                           error=f"Only {len(h_peaks)} horizontal lines detected (need 2+)")
            
            if brick_height_px > min_brick_height:
                # Calculate scale from HEIGHT only (more reliable)
                scale_y = self.brick_height_mm / brick_height_px
                
                # Estimate length based on known aspect ratio
                expected_aspect = self.brick_length_mm / self.brick_height_mm
                brick_length_px = brick_height_px * expected_aspect
                scale_x = scale_y  # Same scale
                
                avg_scale = scale_y  # Use height-based scale
                
                # Confidence based on number of lines detected and consistency
                spacing_std = np.std(h_spacings) if len(h_spacings) > 1 else 0
                spacing_cv = spacing_std / brick_height_px if brick_height_px > 0 else 1
                confidence = max(0, 1 - spacing_cv) * min(len(h_peaks) / 5, 1.0)
                confidence = min(confidence, 1.0)
                
                return BrickDetectionResult(
                    success=True,
                    brick_length_px=brick_length_px,
                    brick_height_px=brick_height_px,
                    scale_x_mm_per_px=scale_x,
                    scale_y_mm_per_px=scale_y,
                    avg_scale_mm_per_px=avg_scale,
                    num_bricks_detected=len(h_peaks) - 1,
                    confidence=confidence,
                    method="mortar_lines_height",
                    debug_info={
                        'h_peaks': h_peaks.tolist(),
                        'h_spacings': h_spacings.tolist(),
                        'spacing_std': float(spacing_std)
                    }
                )
            
            return BrickDetectionResult(success=False, method="mortar_lines",
                                       error="Detected brick height too small")
            
        except Exception as e:
            return BrickDetectionResult(success=False, method="mortar_lines", error=str(e))
    
    def _detect_via_frequency(self, image: np.ndarray) -> BrickDetectionResult:
        """
        Detect brick HEIGHT using FFT frequency analysis.
        Looks for periodic horizontal pattern (brick courses).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Estimate expected brick height range in pixels
            min_brick_height = h // 20
            max_brick_height = h // 3
            
            # Apply FFT
            f = np.fft.fft2(gray.astype(float))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Analyze VERTICAL frequencies (for brick HEIGHT - horizontal pattern)
            center_col = magnitude[:, w//2]
            
            # Convert expected brick sizes to frequency domain
            min_freq_v = max(2, h // max_brick_height)
            max_freq_v = h // min_brick_height
            
            # Find peaks in vertical frequency (for height)
            v_search_start = h//2 + min_freq_v
            v_search_end = min(h//2 + max_freq_v, len(center_col))
            
            v_region = center_col[v_search_start:v_search_end]
            
            v_peaks, _ = find_peaks(v_region, distance=3, prominence=np.max(v_region) * 0.1)
            
            if len(v_peaks) > 0:
                # Get the dominant frequency (highest peak)
                v_peak_idx = v_peaks[np.argmax(v_region[v_peaks])]
                
                # Convert frequency to spatial period (pixels)
                freq_v = v_peak_idx + min_freq_v
                brick_height_px = h / freq_v if freq_v > 0 else 0
                
                if min_brick_height < brick_height_px < max_brick_height:
                    # Calculate scale from HEIGHT only
                    scale_y = self.brick_height_mm / brick_height_px
                    
                    # Estimate length based on aspect ratio
                    expected_aspect = self.brick_length_mm / self.brick_height_mm
                    brick_length_px = brick_height_px * expected_aspect
                    scale_x = scale_y
                    
                    avg_scale = scale_y
                    
                    # Confidence based on peak strength
                    peak_strength = v_region[v_peak_idx] / np.mean(v_region)
                    confidence = min(0.7 * (peak_strength / 10), 0.7)
                    
                    return BrickDetectionResult(
                        success=True,
                        brick_length_px=brick_length_px,
                        brick_height_px=brick_height_px,
                        scale_x_mm_per_px=scale_x,
                        scale_y_mm_per_px=scale_y,
                        avg_scale_mm_per_px=avg_scale,
                        num_bricks_detected=int(h / brick_height_px),
                        confidence=confidence,
                        method="frequency_height"
                    )
            
            return BrickDetectionResult(success=False, method="frequency_analysis",
                                       error="No clear horizontal periodic pattern detected")
            
        except Exception as e:
            return BrickDetectionResult(success=False, method="frequency_analysis", error=str(e))
    
    def _detect_via_edges(self, image: np.ndarray) -> BrickDetectionResult:
        """
        Detect brick HEIGHT using edge detection.
        Looks for horizontal edges (mortar lines between courses).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Estimate expected brick height range
            min_brick_height = h // 20
            max_brick_height = h // 3
            
            # Sobel edge detection for horizontal edges
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_y = np.abs(sobel_y)
            
            # Project to find horizontal lines
            h_projection = np.sum(sobel_y, axis=1)
            h_projection = ndimage.gaussian_filter1d(h_projection, sigma=5)
            
            # Find peaks
            h_peaks, _ = find_peaks(h_projection, 
                                   distance=min_brick_height,
                                   prominence=np.max(h_projection) * 0.1)
            
            if len(h_peaks) >= 2:
                h_spacings = np.diff(h_peaks)
                # Filter outliers
                h_spacings = h_spacings[(h_spacings > min_brick_height) & 
                                       (h_spacings < max_brick_height)]
                
                if len(h_spacings) > 0:
                    brick_height_px = np.median(h_spacings)
                    
                    # Calculate scale from HEIGHT
                    scale_y = self.brick_height_mm / brick_height_px
                    
                    # Estimate length from aspect ratio
                    expected_aspect = self.brick_length_mm / self.brick_height_mm
                    brick_length_px = brick_height_px * expected_aspect
                    scale_x = scale_y
                    
                    avg_scale = scale_y
                    
                    # Confidence
                    spacing_std = np.std(h_spacings)
                    spacing_cv = spacing_std / brick_height_px
                    confidence = max(0, 0.6 * (1 - spacing_cv))
                    
                    return BrickDetectionResult(
                        success=True,
                        brick_length_px=brick_length_px,
                        brick_height_px=brick_height_px,
                        scale_x_mm_per_px=scale_x,
                        scale_y_mm_per_px=scale_y,
                        avg_scale_mm_per_px=avg_scale,
                        num_bricks_detected=len(h_peaks) - 1,
                        confidence=confidence,
                        method="edge_height"
                    )
            
            return BrickDetectionResult(success=False, method="edge_analysis",
                                       error=f"Only {len(h_peaks)} horizontal edges found (need 2+)")
            
        except Exception as e:
            return BrickDetectionResult(success=False, method="edge_analysis", error=str(e))
    
    def interactive_select(self, image: np.ndarray, output_dir: str = None) -> BrickDetectionResult:
        """
        Allow user to measure brick HEIGHT for calibration.
        Since brick heights are more consistent than lengths, we only need the height.
        Uses matplotlib for GUI which works without GTK.
        """
        try:
            return self._interactive_select_matplotlib(image, output_dir)
        except Exception as e:
            print(f"GUI selection failed: {e}")
            print("Falling back to terminal input...")
            return self._interactive_select_terminal(image, output_dir)
    
    def _interactive_select_matplotlib(self, image: np.ndarray, output_dir: str = None) -> BrickDetectionResult:
        """
        Interactive brick HEIGHT selection using matplotlib.
        User draws a vertical line across ONE brick's height.
        """
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend which is more widely available
        import matplotlib.pyplot as plt
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("\n" + "="*60)
        print("INTERACTIVE BRICK HEIGHT CALIBRATION")
        print("="*60)
        print("Instructions:")
        print("  1. Click on the TOP edge of a brick")
        print("  2. Click on the BOTTOM edge of the SAME brick")
        print("  3. This measures the brick HEIGHT (most consistent dimension)")
        print("  4. Close the window when done")
        print()
        print(f"Known brick height: {self.brick_height_mm} mm")
        print("="*60 + "\n")
        
        # Store clicks
        clicks = {'points': []}
        
        def onclick(event):
            """Callback for mouse clicks"""
            if event.xdata is None or event.ydata is None:
                return
            
            x, y = int(event.xdata), int(event.ydata)
            clicks['points'].append((x, y))
            
            # Draw point
            ax.plot(x, y, 'go', markersize=10)
            
            if len(clicks['points']) == 1:
                print(f"  Point 1 (top): ({x}, {y})")
                ax.set_title("Now click on the BOTTOM edge of the same brick")
            elif len(clicks['points']) == 2:
                x1, y1 = clicks['points'][0]
                x2, y2 = clicks['points'][1]
                
                # Draw line between points
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
                
                # Calculate height in pixels
                height_px = abs(y2 - y1)
                print(f"  Point 2 (bottom): ({x}, {y})")
                print(f"  Brick height: {height_px} pixels")
                
                ax.set_title(f"Brick height: {height_px} px = {self.brick_height_mm} mm\nClose window to continue")
            
            fig.canvas.draw()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title("Click on TOP edge of a brick, then BOTTOM edge")
        
        # Connect click event
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.tight_layout()
        plt.show()
        
        # Process selection
        if len(clicks['points']) < 2:
            return BrickDetectionResult(success=False, error="Need 2 points to measure brick height")
        
        y1 = clicks['points'][0][1]
        y2 = clicks['points'][1][1]
        brick_height_px = abs(y2 - y1)
        
        if brick_height_px < 10:
            return BrickDetectionResult(success=False, error="Selection too small")
        
        # Calculate scale factor from height only
        scale_y = self.brick_height_mm / brick_height_px
        
        # Estimate brick length in pixels based on known aspect ratio
        expected_aspect = self.brick_length_mm / self.brick_height_mm
        brick_length_px = brick_height_px * expected_aspect
        scale_x = self.brick_length_mm / brick_length_px  # This equals scale_y
        
        avg_scale = scale_y  # Use height-based scale as the primary
        
        print(f"\n  ✓ Brick height: {brick_height_px} pixels = {self.brick_height_mm} mm")
        print(f"  ✓ Scale factor: {avg_scale:.4f} mm/pixel")
        
        return BrickDetectionResult(
            success=True,
            brick_length_px=brick_length_px,
            brick_height_px=brick_height_px,
            scale_x_mm_per_px=scale_x,
            scale_y_mm_per_px=scale_y,
            avg_scale_mm_per_px=avg_scale,
            num_bricks_detected=1,
            confidence=0.95,  # Manual selection is reliable
            method="interactive_height"
        )
    
    def _interactive_select_terminal(self, image: np.ndarray, output_dir: str = None) -> BrickDetectionResult:
        """
        Fallback: Allow user to specify brick HEIGHT via terminal input.
        """
        h, w = image.shape[:2]
        
        print("\n" + "="*60)
        print("BRICK HEIGHT CALIBRATION (Terminal Mode)")
        print("="*60)
        print(f"Image size: {w} x {h} pixels")
        print()
        print("To calibrate, measure the HEIGHT of ONE brick in pixels.")
        print(f"Known brick height: {self.brick_height_mm} mm")
        print()
        print("TIP: Brick height is more consistent than length.")
        print("     Measure from top edge to bottom edge of one brick.")
        print("="*60)
        
        # Save a reference image with horizontal lines for height reference
        if output_dir:
            ref_path = os.path.join(output_dir, 'reference_grid.jpg')
            ref_img = image.copy()
            # Draw horizontal lines only (more useful for height measurement)
            for y in range(0, h, 50):
                cv2.line(ref_img, (0, y), (w, y), (0, 255, 255), 1)
                cv2.putText(ref_img, str(y), (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # Add vertical reference at edges
            for x in range(0, w, 200):
                cv2.line(ref_img, (x, 0), (x, h), (0, 255, 255), 1)
                cv2.putText(ref_img, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(ref_path, ref_img)
            print(f"\n📷 Reference image saved to: {ref_path}")
            print("   Open this image to measure brick height.\n")
        
        while True:
            print("\nChoose input method:")
            print("  [1] Enter brick HEIGHT in pixels directly")
            print("  [2] Enter Y coordinates of top and bottom edges")
            print("  [q] Quit/Cancel")
            
            choice = input("\nYour choice (1/2/q): ").strip().lower()
            
            if choice == 'q':
                return BrickDetectionResult(success=False, error="User cancelled")
            
            elif choice == '1':
                try:
                    brick_height_px = float(input(f"Enter brick HEIGHT in pixels (represents {self.brick_height_mm} mm): "))
                    
                    if brick_height_px < 10:
                        print("Error: Height too small. Try again.")
                        continue
                    
                    break
                    
                except ValueError:
                    print("Error: Invalid number. Try again.")
                    continue
            
            elif choice == '2':
                try:
                    print("\nEnter Y coordinates:")
                    y1 = int(input("  Y of TOP edge of brick: "))
                    y2 = int(input("  Y of BOTTOM edge of brick: "))
                    
                    brick_height_px = abs(y2 - y1)
                    
                    if brick_height_px < 10:
                        print("Error: Height too small. Try again.")
                        continue
                    
                    print(f"\nCalculated brick height: {brick_height_px} pixels")
                    break
                    
                except ValueError:
                    print("Error: Invalid coordinates. Try again.")
                    continue
            
            else:
                print("Invalid choice. Enter 1, 2, or q.")
        
        # Calculate scale factor from height only
        scale_y = self.brick_height_mm / brick_height_px
        
        # Estimate brick length based on aspect ratio
        expected_aspect = self.brick_length_mm / self.brick_height_mm
        brick_length_px = brick_height_px * expected_aspect
        scale_x = scale_y  # Same scale in both directions
        
        avg_scale = scale_y
        
        print(f"\n  ✓ Brick height: {brick_height_px} pixels = {self.brick_height_mm} mm")
        print(f"  ✓ Scale factor: {avg_scale:.4f} mm/pixel")
        
        return BrickDetectionResult(
            success=True,
            brick_length_px=brick_length_px,
            brick_height_px=brick_height_px,
            scale_x_mm_per_px=scale_x,
            scale_y_mm_per_px=scale_y,
            avg_scale_mm_per_px=avg_scale,
            num_bricks_detected=1,
            confidence=0.95,  # Manual selection is reliable
            method="interactive_height_terminal"
        )


class CrackMeasurerWithBrickCalibration:
    """
    Measures crack dimensions using brick-calibrated scale factor.
    """
    
    def __init__(self, seg_model_path: str = None, device: str = 'cuda'):
        """Initialize with segmentation model."""
        if seg_model_path and os.path.exists(seg_model_path):
            self.seg_model = SegmentationModel(seg_model_path, device=device)
        else:
            self.seg_model = None
            print("Warning: No segmentation model provided. Must supply pre-computed mask.")
    
    def measure(self, image: np.ndarray, mask: np.ndarray, 
                scale_mm_per_px: float) -> Dict:
        """
        Measure crack dimensions using calibrated scale.
        
        Args:
            image: Original image (for reference)
            mask: Binary crack mask (255 = crack, 0 = background)
            scale_mm_per_px: Calibrated scale factor
            
        Returns:
            Dictionary with measurements (including skeleton array)
        """
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt
        
        # Ensure binary mask
        mask_binary = (mask > 80).astype(np.uint8)
        
        # Calculate area
        crack_pixels = np.sum(mask_binary > 0)
        area_mm2 = crack_pixels * (scale_mm_per_px ** 2)
        
        # Skeletonize for length
        skeleton = skeletonize(mask_binary > 0)
        length_pixels = np.sum(skeleton)
        length_mm = length_pixels * scale_mm_per_px
        
        # Width via distance transform
        dist_transform = distance_transform_edt(mask_binary)
        skeleton_coords = np.where(skeleton)
        
        if len(skeleton_coords[0]) > 0:
            widths_px = dist_transform[skeleton_coords] * 2  # diameter = 2 * radius
            widths_mm = widths_px * scale_mm_per_px
            
            width_stats = {
                'max_mm': float(np.max(widths_mm)),
                'min_mm': float(np.min(widths_mm)),
                'mean_mm': float(np.mean(widths_mm)),
                'median_mm': float(np.median(widths_mm)),
                'std_mm': float(np.std(widths_mm)),
                'p95_mm': float(np.percentile(widths_mm, 95))
            }
        else:
            width_stats = {
                'max_mm': 0, 'min_mm': 0, 'mean_mm': 0,
                'median_mm': 0, 'std_mm': 0, 'p95_mm': 0
            }
        
        return {
            'success': True,
            'length_mm': float(length_mm),
            'area_mm2': float(area_mm2),
            'width_stats': width_stats,
            'scale_mm_per_px': scale_mm_per_px,
            'crack_pixels': int(crack_pixels),
            'skeleton_pixels': int(length_pixels),
            'skeleton': skeleton,  # Return skeleton array for visualization
            'binary_mask': mask_binary  # Return binary mask too
        }


def visualize_skeleton(image: np.ndarray, skeleton: np.ndarray, binary_mask: np.ndarray,
                       output_dir: str, threshold: int = 100) -> None:
    """
    Create multiple skeleton visualizations to help calibrate threshold.
    
    Saves:
    1. skeleton_only.png - White skeleton on black background
    2. skeleton_overlay.png - Skeleton (red) overlaid on original image
    3. skeleton_vs_mask.png - Side-by-side comparison showing mask, skeleton, and overlay
    4. skeleton_analysis.png - Detailed analysis with crack coverage stats
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    h, w = image.shape[:2]
    
    # 1. Skeleton only (white on black)
    skeleton_img = (skeleton.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, 'skeleton_only.png'), skeleton_img)
    
    # 2. Skeleton overlay on original image
    overlay = image.copy()
    # Make skeleton thicker for visibility (dilate)
    kernel = np.ones((3, 3), np.uint8)
    skeleton_thick = cv2.dilate(skeleton_img, kernel, iterations=1)
    # Color skeleton pixels red
    overlay[skeleton_thick > 0] = [0, 0, 255]  # BGR = Red
    cv2.imwrite(os.path.join(output_dir, 'skeleton_overlay.png'), overlay)
    
    # 3. Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Skeleton Analysis (Threshold = {threshold})', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title(f'Binary Mask (threshold={threshold})\n{np.sum(binary_mask > 0):,} pixels')
    axes[0, 1].axis('off')
    
    # Skeleton only
    axes[0, 2].imshow(skeleton, cmap='hot')
    axes[0, 2].set_title(f'Skeleton (length calculation)\n{np.sum(skeleton):,} pixels')
    axes[0, 2].axis('off')
    
    # Overlay - skeleton on mask
    mask_with_skeleton = np.zeros((h, w, 3), dtype=np.uint8)
    mask_with_skeleton[:, :, 0] = binary_mask  # Blue channel = mask
    mask_with_skeleton[:, :, 1] = binary_mask  # Green channel = mask  
    mask_with_skeleton[:, :, 2] = skeleton.astype(np.uint8) * 255  # Red channel = skeleton
    axes[1, 0].imshow(mask_with_skeleton)
    axes[1, 0].set_title('Skeleton (red) on Mask (cyan)')
    axes[1, 0].axis('off')
    
    # Overlay on original
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title('Skeleton (red) on Original')
    axes[1, 1].axis('off')
    
    # Statistics text
    skeleton_pixels = np.sum(skeleton)
    mask_pixels = np.sum(binary_mask > 0)
    coverage_ratio = skeleton_pixels / mask_pixels if mask_pixels > 0 else 0
    
    stats_text = f"""
SKELETON ANALYSIS
═══════════════════════════════════

Threshold Used: {threshold}

PIXEL COUNTS:
  Binary mask pixels:  {mask_pixels:,}
  Skeleton pixels:     {skeleton_pixels:,}
  Coverage ratio:      {coverage_ratio:.2%}

INTERPRETATION:
  • Skeleton pixels = crack length in pixels
  • Each skeleton pixel = 1 unit of length
  • Diagonal connections add √2 ≈ 1.41 per pixel

THRESHOLD TIPS:
  • If crack appears broken → LOWER threshold
  • If too much noise/extra regions → RAISE threshold
  • Ideal: Continuous skeleton covering visible crack

CURRENT STATUS:
  {'✓ Skeleton looks continuous' if skeleton_pixels > 50 else '⚠ Very few skeleton pixels - check threshold'}
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Analysis Summary')
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, 'skeleton_analysis.png')
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Skeleton visualizations saved:")
    print(f"      - skeleton_only.png (white skeleton on black)")
    print(f"      - skeleton_overlay.png (skeleton on original)")
    print(f"      - skeleton_analysis.png (comprehensive analysis)")


def visualize_detection(image: np.ndarray, detection: BrickDetectionResult,
                       output_path: str = None) -> np.ndarray:
    """Create visualization of brick detection results."""
    vis = image.copy()
    h, w = vis.shape[:2]
    
    if detection.success:
        # Draw grid based on detected brick size
        brick_w = int(detection.brick_length_px)
        brick_h = int(detection.brick_height_px)
        
        # Draw vertical lines
        for x in range(0, w, brick_w):
            cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        
        # Draw horizontal lines
        for y in range(0, h, brick_h):
            cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        
        # Add info text
        info_text = [
            f"Method: {detection.method}",
            f"Brick: {brick_w}x{brick_h} px",
            f"Scale: {detection.avg_scale_mm_per_px:.4f} mm/px",
            f"Confidence: {detection.confidence:.2f}"
        ]
        
        y_pos = 30
        for text in info_text:
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            y_pos += 30
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Measure crack dimensions using brick-based scale calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Standard Brick Types:
  india_modular     - Indian Modular (190x90mm)
  india_traditional - Indian Traditional (230x75mm)  
  uk_standard       - UK Standard (215x65mm)
  us_standard       - US Standard (194x57mm)
  europe_nf         - European NF (240x71mm)

Examples:
  # Auto-detect with Indian modular bricks
  python brick_calibrated_measurement.py image.jpg --brick-type india_modular
  
  # Custom brick dimensions
  python brick_calibrated_measurement.py image.jpg --brick-length 200 --brick-height 80
  
  # Interactive mode (manually select a brick)
  python brick_calibrated_measurement.py image.jpg --interactive
        """
    )
    
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--brick-type', choices=list(BRICK_STANDARDS.keys()),
                       default='india_modular',
                       help='Standard brick type (default: india_modular)')
    parser.add_argument('--brick-length', type=float, default=None,
                       help='Custom brick length in mm (overrides --brick-type)')
    parser.add_argument('--brick-height', type=float, default=None,
                       help='Custom brick height in mm (overrides --brick-type)')
    parser.add_argument('--mortar-width', type=float, default=10,
                       help='Mortar joint width in mm (default: 10)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactively select a brick for calibration')
    parser.add_argument('--seg-model', default='Masking_and_Classification_model/pretrained_net_G.pth',
                       help='Path to segmentation model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device for inference')
    parser.add_argument('--threshold', type=int, default=100,
                       help='Segmentation threshold (0-255)')
    parser.add_argument('--output-dir', '-o', default='./brick_calibrated_results',
                       help='Output directory')
    parser.add_argument('--show-detection', action='store_true',
                       help='Show brick detection visualization')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)
    
    # Determine brick dimensions
    if args.brick_length and args.brick_height:
        brick_length = args.brick_length
        brick_height = args.brick_height
        brick_name = "Custom"
    else:
        brick_info = BRICK_STANDARDS[args.brick_type]
        brick_length = brick_info['length_mm']
        brick_height = brick_info['height_mm']
        brick_name = brick_info['name']
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           BRICK-CALIBRATED CRACK MEASUREMENT                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Image: {os.path.basename(args.image):<54} ║
║  Brick Type: {brick_name:<49} ║
║  Brick Dimensions: {brick_length:.0f} x {brick_height:.0f} mm{' ':<36} ║
║  Mortar Width: {args.mortar_width:.0f} mm{' ':<44} ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize brick detector
    detector = BrickDetector(
        brick_length_mm=brick_length,
        brick_height_mm=brick_height,
        mortar_width_mm=args.mortar_width
    )
    
    # Detect bricks
    print("\n[1/3] Detecting brick pattern...")
    
    if args.interactive:
        detection = detector.interactive_select(image, output_dir=args.output_dir)
    else:
        detection = detector.detect(image)
    
    if not detection.success:
        print(f"\n⚠️  Automatic brick detection failed: {detection.error}")
        print("    Falling back to interactive mode...")
        detection = detector.interactive_select(image, output_dir=args.output_dir)
    
    if not detection.success:
        print(f"\n❌ Error: {detection.error}")
        sys.exit(1)
    
    print(f"""
    ✓ Brick detection successful!
      Method: {detection.method}
      Detected brick size: {detection.brick_length_px:.1f} x {detection.brick_height_px:.1f} pixels
      Scale factor (X): {detection.scale_x_mm_per_px:.4f} mm/pixel
      Scale factor (Y): {detection.scale_y_mm_per_px:.4f} mm/pixel
      Average scale: {detection.avg_scale_mm_per_px:.4f} mm/pixel
      Confidence: {detection.confidence:.2%}
      Bricks detected: ~{detection.num_bricks_detected}
    """)
    
    # Save detection visualization
    vis_path = os.path.join(args.output_dir, 'brick_detection.jpg')
    vis = visualize_detection(image, detection, vis_path)
    print(f"    Detection visualization saved: {vis_path}")
    
    if args.show_detection:
        # Try to display, but don't fail if GUI not available
        try:
            cv2.imshow("Brick Detection", cv2.resize(vis, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("    (Cannot display image - OpenCV GUI not available)")
    
    # Segment crack
    print("\n[2/3] Segmenting crack...")
    
    if not os.path.exists(args.seg_model):
        print(f"Error: Segmentation model not found: {args.seg_model}")
        sys.exit(1)
    
    seg_model = SegmentationModel(args.seg_model, device=args.device)
    raw_mask = seg_model.segment(args.image)
    
    # Save raw mask
    raw_mask_path = os.path.join(args.output_dir, 'segmentation_raw.png')
    cv2.imwrite(raw_mask_path, raw_mask)
    print(f"    Raw segmentation mask saved: {raw_mask_path}")
    
    # Binarize
    binary_mask = (raw_mask > args.threshold).astype(np.uint8) * 255
    binary_mask_path = os.path.join(args.output_dir, 'segmentation_binary.png')
    cv2.imwrite(binary_mask_path, binary_mask)
    print(f"    Binary mask saved: {binary_mask_path}")
    
    # Measure crack
    print("\n[3/3] Measuring crack dimensions...")
    
    measurer = CrackMeasurerWithBrickCalibration()
    results = measurer.measure(image, binary_mask, detection.avg_scale_mm_per_px)
    
    # Generate skeleton visualizations
    print("\n[4/4] Generating skeleton visualizations...")
    if 'skeleton' in results and results['skeleton'] is not None:
        visualize_skeleton(
            image=image,
            skeleton=results['skeleton'],
            binary_mask=results['binary_mask'],
            output_dir=args.output_dir,
            threshold=args.threshold
        )
        # Remove numpy arrays from results before saving to JSON
        skeleton_for_save = results.pop('skeleton', None)
        binary_mask_for_save = results.pop('binary_mask', None)
    
    # Add calibration info to results
    results['calibration'] = {
        'method': detection.method,
        'brick_type': brick_name,
        'brick_length_mm': brick_length,
        'brick_height_mm': brick_height,
        'brick_length_px': detection.brick_length_px,
        'brick_height_px': detection.brick_height_px,
        'scale_x_mm_per_px': detection.scale_x_mm_per_px,
        'scale_y_mm_per_px': detection.scale_y_mm_per_px,
        'confidence': detection.confidence,
        'threshold_used': args.threshold
    }
    
    # Display results
    width_stats = results['width_stats']
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    CRACK MEASUREMENTS                             ║
║              (Brick-Calibrated - REAL Dimensions)                 ║
╠══════════════════════════════════════════════════════════════════╣
║  CALIBRATION:                                                    ║
║    Brick type: {brick_name:<47} ║
║    Reference: {brick_length:.0f} x {brick_height:.0f} mm = {detection.brick_length_px:.0f} x {detection.brick_height_px:.0f} px{' ':<20} ║
║    Scale: {detection.avg_scale_mm_per_px:.4f} mm/pixel (confidence: {detection.confidence:.0%}){' ':<15} ║
╠══════════════════════════════════════════════════════════════════╣
║  CRACK DIMENSIONS:                                               ║
║    Length:        {results['length_mm']:>10.2f} mm{' ':<34} ║
║    Max Width:     {width_stats['max_mm']:>10.2f} mm{' ':<34} ║
║    Mean Width:    {width_stats['mean_mm']:>10.2f} mm{' ':<34} ║
║    Median Width:  {width_stats['median_mm']:>10.2f} mm{' ':<34} ║
║    Area:          {results['area_mm2']:>10.2f} mm²{' ':<33} ║
╠══════════════════════════════════════════════════════════════════╣
║  These measurements are calibrated using the brick pattern       ║
║  detected in the image. Accuracy depends on:                     ║
║    1. Correct brick type selection                               ║
║    2. Quality of brick detection                                 ║
║    3. Camera angle (perspective distortion)                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results to file
    import json
    results_path = os.path.join(args.output_dir, 'measurements.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, 'measurements.txt')
    with open(summary_path, 'w') as f:
        f.write("BRICK-CALIBRATED CRACK MEASUREMENTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image: {args.image}\n")
        f.write(f"Brick Type: {brick_name}\n")
        f.write(f"Brick Size: {brick_length} x {brick_height} mm\n")
        f.write(f"Scale Factor: {detection.avg_scale_mm_per_px:.4f} mm/pixel\n")
        f.write(f"Calibration Confidence: {detection.confidence:.2%}\n\n")
        f.write(f"Length: {results['length_mm']:.2f} mm\n")
        f.write(f"Max Width: {width_stats['max_mm']:.2f} mm\n")
        f.write(f"Mean Width: {width_stats['mean_mm']:.2f} mm\n")
        f.write(f"Median Width: {width_stats['median_mm']:.2f} mm\n")
        f.write(f"Area: {results['area_mm2']:.2f} mm²\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
