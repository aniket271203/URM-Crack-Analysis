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
    
    Uses the RAW segmentation probability map (0-255) instead of a hard binary
    threshold. This preserves crack pixels that have moderate confidence but are
    still clearly part of the crack structure.
    
    Approach:
    1. Hysteresis thresholding: high-confidence seeds expanded to medium-confidence
       connected neighbors (like Canny edge detection logic)
    2. Connected component cleanup: keep only the largest crack region
    3. Geodesic skeleton length: proper √2 diagonal distance accounting
    4. Intensity-weighted perpendicular width profiling along the skeleton
    """
    
    def __init__(self, seg_model_path: str = None, device: str = 'cuda'):
        """Initialize with segmentation model."""
        if seg_model_path and os.path.exists(seg_model_path):
            self.seg_model = SegmentationModel(seg_model_path, device=device)
        else:
            self.seg_model = None
            print("Warning: No segmentation model provided. Must supply pre-computed mask.")
    
    @staticmethod
    def hysteresis_threshold(raw_mask: np.ndarray, 
                              low_thresh: int = 30, 
                              high_thresh: int = 80) -> np.ndarray:
        """
        Apply hysteresis thresholding on the raw segmentation probability map.
        
        - Pixels above high_thresh are definite crack (seeds).
        - Pixels between low_thresh and high_thresh are accepted only if they
          are connected (8-connectivity) to a definite crack pixel.
        - Pixels below low_thresh are rejected.
        
        This retains faint but connected crack regions that a single hard
        threshold would discard.
        
        Args:
            raw_mask: Grayscale probability map (uint8, 0-255)
            low_thresh: Lower threshold for candidate pixels
            high_thresh: Upper threshold for definite seed pixels
            
        Returns:
            Binary mask (uint8, 0 or 255)
        """
        from scipy.ndimage import label as ndimage_label
        
        seeds = (raw_mask >= high_thresh).astype(np.uint8)
        candidates = (raw_mask >= low_thresh).astype(np.uint8)
        
        # Label connected components in the candidate region
        labeled, num_features = ndimage_label(candidates)
        
        # Find which labels contain at least one seed pixel
        seed_labels = set(np.unique(labeled[seeds > 0]))
        seed_labels.discard(0)  # remove background label
        
        # Keep only candidate components that touch a seed
        result = np.zeros_like(raw_mask)
        for lbl in seed_labels:
            result[labeled == lbl] = 255
        
        return result
    
    @staticmethod
    def cleanup_connected_components(mask: np.ndarray, 
                                      min_area_fraction: float = 0.01) -> np.ndarray:
        """
        Remove small noise components. Keep components that are at least
        min_area_fraction of the largest component's area.
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            min_area_fraction: Minimum fraction of largest component to keep
            
        Returns:
            Cleaned binary mask (uint8, 0 or 255)
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            return mask
        
        # stats[:, cv2.CC_STAT_AREA] — area of each component (label 0 = background)
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        max_area = np.max(areas)
        min_area = max_area * min_area_fraction
        
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        
        return cleaned
    
    @staticmethod
    def geodesic_skeleton_length(skeleton: np.ndarray) -> float:
        """
        Compute the true geodesic length of a skeleton by accounting for
        diagonal vs cardinal neighbor connections.
        
        Cardinal neighbors (up/down/left/right) contribute 1.0 pixel distance.
        Diagonal neighbors contribute √2 ≈ 1.414 pixel distance.
        
        Args:
            skeleton: Boolean skeleton array
            
        Returns:
            Geodesic length in pixels (float)
        """
        skel = skeleton.astype(np.uint8)
        coords = np.argwhere(skel > 0)
        
        if len(coords) == 0:
            return 0.0
        
        # Build a set for O(1) lookup
        coord_set = set(map(tuple, coords))
        
        length = 0.0
        visited_edges = set()
        
        SQRT2 = np.sqrt(2)
        # 8-connected neighbors: (dy, dx, distance)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),  # cardinal
            (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)  # diagonal
        ]
        
        for (r, c) in coords:
            for dr, dc, dist in neighbors:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coord_set:
                    edge = (min((r, c), (nr, nc)), max((r, c), (nr, nc)))
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        length += dist
        
        # Handle isolated skeleton pixels (single dot)
        if length == 0.0 and len(coords) > 0:
            length = 1.0
        
        return length
    
    @staticmethod
    def perpendicular_width_profile(skeleton: np.ndarray, 
                                     raw_mask: np.ndarray,
                                     sample_step: int = 3,
                                     profile_half_length: int = 50) -> np.ndarray:
        """
        Measure crack width at each sampled skeleton point by casting
        a perpendicular line and counting continuous non-zero pixels
        in the raw segmentation mask.
        
        Uses the raw probability map directly: walks outward from each
        skeleton point in both perpendicular directions until the intensity
        drops below a minimum threshold, counting the span as the width.
        
        Args:
            skeleton: Boolean skeleton array
            raw_mask: Raw grayscale probability map (uint8, 0-255)
            sample_step: Sample every N-th skeleton pixel for speed
            profile_half_length: Maximum pixels to walk in each direction
            
        Returns:
            Array of width measurements in pixels (one per sampled point)
        """
        from scipy.ndimage import label as ndimage_label
        
        skel = skeleton.astype(np.uint8)
        h, w = skel.shape
        
        # Get ordered skeleton points by tracing connected paths
        coords = np.argwhere(skel > 0)
        if len(coords) < 2:
            return np.array([1.0])
        
        # Compute local tangent direction at each skeleton point using neighbors
        # Use a small Gaussian-blurred version of skeleton for smooth gradient
        skel_float = skeleton.astype(np.float64)
        # Sobel gives the gradient direction of the skeleton path
        gy = cv2.Sobel(skel_float, cv2.CV_64F, 0, 1, ksize=3)
        gx = cv2.Sobel(skel_float, cv2.CV_64F, 1, 0, ksize=3)
        
        # Use a low threshold on raw mask for width boundary detection
        # This should be the minimum intensity we still consider "crack"
        min_intensity = 15  # Very low — captures faint edges of crack
        
        widths = []
        sampled_coords = coords[::sample_step]
        
        for (r, c) in sampled_coords:
            tx, ty = gx[r, c], gy[r, c]
            
            # Tangent is along the crack; perpendicular is rotated 90°
            mag = np.sqrt(tx**2 + ty**2)
            if mag < 1e-6:
                # Fallback: use local neighborhood to estimate direction
                # Check which neighbors are also skeleton pixels
                local = skel[max(0,r-2):r+3, max(0,c-2):c+3]
                local_coords = np.argwhere(local > 0)
                if len(local_coords) >= 2:
                    diff = local_coords[-1] - local_coords[0]
                    ty, tx = float(diff[0]), float(diff[1])
                    mag = np.sqrt(tx**2 + ty**2)
                if mag < 1e-6:
                    # Truly isolated, assume horizontal crack
                    px, py = 0.0, 1.0
                else:
                    px, py = -ty/mag, tx/mag
            else:
                # Perpendicular direction
                px, py = -ty/mag, tx/mag
            
            # Walk in +perpendicular direction
            count_pos = 0
            for step in range(1, profile_half_length + 1):
                nr = int(round(r + py * step))
                nc = int(round(c + px * step))
                if 0 <= nr < h and 0 <= nc < w and raw_mask[nr, nc] >= min_intensity:
                    count_pos += 1
                else:
                    break
            
            # Walk in -perpendicular direction
            count_neg = 0
            for step in range(1, profile_half_length + 1):
                nr = int(round(r - py * step))
                nc = int(round(c - px * step))
                if 0 <= nr < h and 0 <= nc < w and raw_mask[nr, nc] >= min_intensity:
                    count_neg += 1
                else:
                    break
            
            total_width = count_pos + count_neg + 1  # +1 for the skeleton pixel itself
            widths.append(float(total_width))
        
        return np.array(widths) if widths else np.array([1.0])
    
    def measure(self, image: np.ndarray, raw_mask: np.ndarray, 
                scale_mm_per_px: float,
                low_thresh: int = 30, high_thresh: int = 80) -> Dict:
        """
        Measure crack dimensions using the RAW segmentation probability map.
        
        Instead of a single hard binary threshold (which loses faint crack pixels),
        this uses:
        1. Hysteresis thresholding to recover connected faint regions
        2. Connected component cleanup to remove noise
        3. Geodesic skeleton length for accurate crack length
        4. Perpendicular intensity profiling on raw mask for accurate width
        
        Args:
            image: Original image (for reference)
            raw_mask: Raw grayscale segmentation map (uint8, 0-255)
            scale_mm_per_px: Calibrated scale factor
            low_thresh: Lower hysteresis threshold (default: 30)
            high_thresh: Upper hysteresis threshold (default: 80)
            
        Returns:
            Dictionary with measurements (including skeleton & mask arrays)
        """
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt
        
        print(f"    Using hysteresis thresholding (low={low_thresh}, high={high_thresh})")
        
        # --- Step 1: Hysteresis thresholding on raw probability map ---
        refined_mask = self.hysteresis_threshold(raw_mask, low_thresh, high_thresh)
        
        # --- Step 2: Clean up small noise components ---
        refined_mask = self.cleanup_connected_components(refined_mask, min_area_fraction=0.01)
        
        # Also create the old-style hard binary mask for comparison
        hard_binary = (raw_mask > high_thresh).astype(np.uint8) * 255
        
        # Stats comparison
        raw_nonzero = int(np.sum(raw_mask > 0))
        hard_nonzero = int(np.sum(hard_binary > 0))
        refined_nonzero = int(np.sum(refined_mask > 0))
        print(f"    Pixel counts — Raw non-zero: {raw_nonzero:,}, "
              f"Hard binary (>{high_thresh}): {hard_nonzero:,}, "
              f"Hysteresis refined: {refined_nonzero:,}")
        recovery_pct = ((refined_nonzero - hard_nonzero) / max(hard_nonzero, 1)) * 100
        print(f"    Recovered {recovery_pct:+.1f}% more crack pixels via hysteresis")
        
        # --- Step 3: Compute mask binary for skeletonization ---
        mask_binary = (refined_mask > 0).astype(np.uint8)
        
        # Calculate area
        crack_pixels = int(np.sum(mask_binary > 0))
        area_mm2 = crack_pixels * (scale_mm_per_px ** 2)
        
        # --- Step 4: Skeletonize the refined mask ---
        skeleton = skeletonize(mask_binary > 0)
        skeleton_pixel_count = int(np.sum(skeleton))
        
        # --- Step 5: Geodesic skeleton length ---
        geodesic_length_px = self.geodesic_skeleton_length(skeleton)
        length_mm = geodesic_length_px * scale_mm_per_px
        
        # Also compute naive length for comparison
        naive_length_px = skeleton_pixel_count
        naive_length_mm = naive_length_px * scale_mm_per_px
        print(f"    Length — Naive (pixel count): {naive_length_mm:.2f} mm, "
              f"Geodesic (diagonal-aware): {length_mm:.2f} mm")
        
        # --- Step 6: Width via perpendicular profiling on raw mask ---
        print(f"    Computing perpendicular width profiles on raw segmentation...")
        widths_px = self.perpendicular_width_profile(
            skeleton, raw_mask, sample_step=3, profile_half_length=50
        )
        widths_mm = widths_px * scale_mm_per_px
        
        # Also compute distance-transform widths for comparison
        dist_transform = distance_transform_edt(mask_binary)
        skeleton_coords = np.where(skeleton)
        if len(skeleton_coords[0]) > 0:
            dt_widths_px = dist_transform[skeleton_coords] * 2
            dt_widths_mm = dt_widths_px * scale_mm_per_px
            dt_width_stats = {
                'max_mm': float(np.max(dt_widths_mm)),
                'mean_mm': float(np.mean(dt_widths_mm)),
                'median_mm': float(np.median(dt_widths_mm)),
            }
        else:
            dt_width_stats = {'max_mm': 0, 'mean_mm': 0, 'median_mm': 0}
        
        if len(widths_mm) > 0:
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
        
        print(f"    Width (perpendicular profile) — Max: {width_stats['max_mm']:.2f} mm, "
              f"Mean: {width_stats['mean_mm']:.2f} mm, Median: {width_stats['median_mm']:.2f} mm")
        print(f"    Width (distance transform)    — Max: {dt_width_stats['max_mm']:.2f} mm, "
              f"Mean: {dt_width_stats['mean_mm']:.2f} mm, Median: {dt_width_stats['median_mm']:.2f} mm")
        
        return {
            'success': True,
            'length_mm': float(length_mm),
            'length_naive_mm': float(naive_length_mm),
            'area_mm2': float(area_mm2),
            'width_stats': width_stats,
            'width_stats_distance_transform': dt_width_stats,
            'scale_mm_per_px': scale_mm_per_px,
            'crack_pixels': crack_pixels,
            'crack_pixels_hard_binary': hard_nonzero,
            'crack_pixels_recovered': refined_nonzero - hard_nonzero,
            'skeleton_pixels': skeleton_pixel_count,
            'geodesic_length_px': float(geodesic_length_px),
            'hysteresis_thresholds': {'low': low_thresh, 'high': high_thresh},
            'skeleton': skeleton,
            'binary_mask': mask_binary,
            'refined_mask': refined_mask,
            'hard_binary_mask': hard_binary
        }


def visualize_skeleton(image: np.ndarray, skeleton: np.ndarray, binary_mask: np.ndarray,
                       output_dir: str, threshold: int = 100,
                       raw_mask: np.ndarray = None, 
                       refined_mask: np.ndarray = None,
                       hard_binary_mask: np.ndarray = None) -> None:
    """
    Create comprehensive skeleton and segmentation comparison visualizations.
    
    Saves:
    1. skeleton_only.png - White skeleton on black background
    2. skeleton_overlay.png - Skeleton (red) overlaid on original image
    3. skeleton_analysis.png - Detailed analysis with raw vs refined comparison
    4. segmentation_comparison.png - Side-by-side of raw, hard binary, and hysteresis masks
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
    kernel = np.ones((3, 3), np.uint8)
    skeleton_thick = cv2.dilate(skeleton_img, kernel, iterations=1)
    overlay[skeleton_thick > 0] = [0, 0, 255]  # BGR = Red
    cv2.imwrite(os.path.join(output_dir, 'skeleton_overlay.png'), overlay)
    
    # 3. Save refined mask
    if refined_mask is not None:
        cv2.imwrite(os.path.join(output_dir, 'segmentation_refined.png'), refined_mask)
    
    # 4. Create comprehensive comparison figure
    has_comparison = raw_mask is not None and hard_binary_mask is not None and refined_mask is not None
    
    if has_comparison:
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'Segmentation & Skeleton Analysis (Hysteresis Thresholding)', 
                     fontsize=16, fontweight='bold')
        
        # Row 1: Segmentation comparison
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(raw_mask, cmap='hot')
        axes[0, 1].set_title(f'Raw Segmentation (probability)\n'
                             f'{np.sum(raw_mask > 0):,} non-zero pixels')
        axes[0, 1].axis('off')
        
        hard_px = np.sum(hard_binary_mask > 0)
        axes[0, 2].imshow(hard_binary_mask, cmap='gray')
        axes[0, 2].set_title(f'Hard Binary (>{threshold})\n{hard_px:,} pixels')
        axes[0, 2].axis('off')
        
        refined_px = np.sum(refined_mask > 0)
        axes[0, 3].imshow(refined_mask, cmap='gray')
        axes[0, 3].set_title(f'Hysteresis Refined\n{refined_px:,} pixels '
                             f'({((refined_px - hard_px) / max(hard_px, 1)) * 100:+.1f}%)')
        axes[0, 3].axis('off')
        
        # Row 2: Skeleton analysis
        # Difference visualization: green = recovered pixels, white = both
        diff_vis = np.zeros((h, w, 3), dtype=np.uint8)
        hard_bool = hard_binary_mask > 0
        refined_bool = refined_mask > 0
        diff_vis[hard_bool] = [255, 255, 255]  # White = in both
        recovered = refined_bool & ~hard_bool
        diff_vis[recovered] = [0, 255, 0]  # Green = recovered by hysteresis
        axes[1, 0].imshow(diff_vis)
        axes[1, 0].set_title(f'Recovered Pixels (green)\n{np.sum(recovered):,} additional pixels')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(skeleton, cmap='hot')
        axes[1, 1].set_title(f'Skeleton\n{np.sum(skeleton):,} pixels')
        axes[1, 1].axis('off')
        
        # Skeleton on refined mask
        mask_with_skeleton = np.zeros((h, w, 3), dtype=np.uint8)
        mask_with_skeleton[:, :, 1] = (refined_mask > 0).astype(np.uint8) * 180
        mask_with_skeleton[:, :, 2] = skeleton.astype(np.uint8) * 255
        axes[1, 2].imshow(mask_with_skeleton)
        axes[1, 2].set_title('Skeleton (red) on Refined Mask (green)')
        axes[1, 2].axis('off')
        
        # Overlay on original
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[1, 3].imshow(overlay_rgb)
        axes[1, 3].set_title('Skeleton (red) on Original')
        axes[1, 3].axis('off')
        
    else:
        # Fallback: original 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Skeleton Analysis (Threshold = {threshold})', fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(binary_mask, cmap='gray')
        axes[0, 1].set_title(f'Binary Mask\n{np.sum(binary_mask > 0):,} pixels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(skeleton, cmap='hot')
        axes[0, 2].set_title(f'Skeleton\n{np.sum(skeleton):,} pixels')
        axes[0, 2].axis('off')
        
        mask_with_skeleton = np.zeros((h, w, 3), dtype=np.uint8)
        mask_with_skeleton[:, :, 0] = binary_mask
        mask_with_skeleton[:, :, 1] = binary_mask
        mask_with_skeleton[:, :, 2] = skeleton.astype(np.uint8) * 255
        axes[1, 0].imshow(mask_with_skeleton)
        axes[1, 0].set_title('Skeleton (red) on Mask (cyan)')
        axes[1, 0].axis('off')
        
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(overlay_rgb)
        axes[1, 1].set_title('Skeleton (red) on Original')
        axes[1, 1].axis('off')
        
        skeleton_pixels = np.sum(skeleton)
        mask_pixels = np.sum(binary_mask > 0)
        coverage_ratio = skeleton_pixels / mask_pixels if mask_pixels > 0 else 0
        stats_text = f"""
SKELETON ANALYSIS
Threshold: {threshold}
Mask pixels:     {mask_pixels:,}
Skeleton pixels: {skeleton_pixels:,}
Coverage ratio:  {coverage_ratio:.2%}
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
    if refined_mask is not None:
        print(f"      - segmentation_refined.png (hysteresis-refined mask)")
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
    parser.add_argument('--threshold', type=int, default=80,
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
    
    # Also save old-style hard binary for reference
    binary_mask = (raw_mask > args.threshold).astype(np.uint8) * 255
    binary_mask_path = os.path.join(args.output_dir, 'segmentation_binary.png')
    cv2.imwrite(binary_mask_path, binary_mask)
    print(f"    Hard binary mask saved (for reference): {binary_mask_path}")
    
    # Measure crack using RAW segmentation with hysteresis thresholding
    print("\n[3/3] Measuring crack dimensions (using raw segmentation)...")
    
    # Compute hysteresis thresholds: use the CLI --threshold as high, and half as low
    high_thresh = args.threshold
    low_thresh = max(10, args.threshold // 3)  # Typically ~27 for default threshold=80
    
    measurer = CrackMeasurerWithBrickCalibration()
    results = measurer.measure(
        image, raw_mask, detection.avg_scale_mm_per_px,
        low_thresh=low_thresh, high_thresh=high_thresh
    )
    
    # Generate skeleton visualizations
    print("\n[4/4] Generating skeleton visualizations...")
    if 'skeleton' in results and results['skeleton'] is not None:
        visualize_skeleton(
            image=image,
            skeleton=results['skeleton'],
            binary_mask=results['binary_mask'],
            output_dir=args.output_dir,
            threshold=args.threshold,
            raw_mask=raw_mask,
            refined_mask=results.get('refined_mask'),
            hard_binary_mask=results.get('hard_binary_mask')
        )
        # Remove numpy arrays from results before saving to JSON
        results.pop('skeleton', None)
        results.pop('binary_mask', None)
        results.pop('refined_mask', None)
        results.pop('hard_binary_mask', None)
    
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
        f.write("(Using Raw Segmentation + Hysteresis Thresholding)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image: {args.image}\n")
        f.write(f"Brick Type: {brick_name}\n")
        f.write(f"Brick Size: {brick_length} x {brick_height} mm\n")
        f.write(f"Scale Factor: {detection.avg_scale_mm_per_px:.4f} mm/pixel\n")
        f.write(f"Calibration Confidence: {detection.confidence:.2%}\n\n")
        f.write(f"Hysteresis Thresholds: low={low_thresh}, high={high_thresh}\n")
        f.write(f"Crack Pixels (hard binary): {results.get('crack_pixels_hard_binary', 'N/A')}\n")
        f.write(f"Crack Pixels (hysteresis):  {results.get('crack_pixels', 'N/A')}\n")
        f.write(f"Pixels Recovered:           {results.get('crack_pixels_recovered', 'N/A')}\n\n")
        f.write(f"Length (geodesic):  {results['length_mm']:.2f} mm\n")
        f.write(f"Length (naive):     {results.get('length_naive_mm', 0):.2f} mm\n")
        f.write(f"Max Width:   {width_stats['max_mm']:.2f} mm\n")
        f.write(f"Mean Width:  {width_stats['mean_mm']:.2f} mm\n")
        f.write(f"Median Width: {width_stats['median_mm']:.2f} mm\n")
        f.write(f"Area: {results['area_mm2']:.2f} mm²\n")
        dt_stats = results.get('width_stats_distance_transform', {})
        if dt_stats:
            f.write(f"\n--- Distance Transform Width (for comparison) ---\n")
            f.write(f"Max Width (DT):    {dt_stats.get('max_mm', 0):.2f} mm\n")
            f.write(f"Mean Width (DT):   {dt_stats.get('mean_mm', 0):.2f} mm\n")
            f.write(f"Median Width (DT): {dt_stats.get('median_mm', 0):.2f} mm\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
