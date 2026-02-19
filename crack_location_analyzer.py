"""
Crack Spatial Location Analyzer
================================
Implements soft spatial grid + heatmap-based crack location analysis.

Given a crack segmentation mask, produces:
- Probabilistic location labels (not hard buckets)
- Handles long/curved/diagonal cracks
- Robust to lighting and texture
- Interpretable & tunable

Core Approach:
- Spatial density estimation via soft grid
- Area-weighted voting from segmentation mask
- Gaussian heatmap smoothing
- Probabilistic semantic region aggregation
"""

import os
import sys
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json

# For visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class LocationAnalysisResult:
    """Result of spatial crack location analysis"""
    success: bool = False
    
    # Raw data
    density_grid: np.ndarray = None  # Raw occupancy grid
    heatmap: np.ndarray = None       # Smoothed heatmap
    
    # Vertical position probabilities
    vertical_position: Dict[str, float] = field(default_factory=dict)
    
    # Horizontal position probabilities  
    horizontal_position: Dict[str, float] = field(default_factory=dict)
    
    # Derived labels
    dominant_location: str = ""      # e.g., "upper-right"
    secondary_location: str = ""     # Second highest concentration
    spread_type: str = ""            # "localized", "multi-region", "widespread"
    confidence: float = 0.0          # How concentrated is the crack
    
    # Propagation analysis
    propagation_direction: str = ""  # e.g., "upper-right → lower-left"
    propagation_vector: Tuple[float, float] = (0.0, 0.0)  # Normalized direction vector
    origin_intensity: float = 0.0    # Intensity at crack origin (higher = deeper)
    terminus_intensity: float = 0.0  # Intensity at crack terminus (lower = shallower)
    
    # Additional metrics
    centroid: Tuple[float, float] = (0.0, 0.0)  # Normalized (x, y)
    crack_endpoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # start/end points
    coverage_ratio: float = 0.0      # Fraction of image covered
    aspect_ratio: float = 0.0        # Width/Height of crack bounding box
    orientation: str = ""            # "horizontal", "vertical", "diagonal"
    
    # Grid configuration used
    grid_size: Tuple[int, int] = (5, 5)
    
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'success': self.success,
            'vertical_position': self.vertical_position,
            'horizontal_position': self.horizontal_position,
            'dominant_location': self.dominant_location,
            'secondary_location': self.secondary_location,
            'propagation_direction': self.propagation_direction,
            'propagation_vector': self.propagation_vector,
            'origin_intensity': self.origin_intensity,
            'terminus_intensity': self.terminus_intensity,
            'spread_type': self.spread_type,
            'confidence': self.confidence,
            'centroid': self.centroid,
            'crack_endpoints': self.crack_endpoints,
            'coverage_ratio': self.coverage_ratio,
            'aspect_ratio': self.aspect_ratio,
            'orientation': self.orientation,
            'grid_size': self.grid_size,
            'error': self.error
        }


class CrackLocationAnalyzer:
    """
    Analyzes spatial location of cracks using soft grid + heatmap approach.
    
    Architecture:
        Input Image
         ├─ Crack Segmentation Mask
         ├─ Soft Spatial Grid
         │    ├─ Area-weighted voting
         │    ├─ Heatmap smoothing
         │    └─ Probabilistic aggregation
         └─ Location Inference (soft labels)
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (5, 5),
                 smoothing_sigma: float = 1.0,
                 threshold: int = 100):
        """
        Initialize the location analyzer.
        
        Args:
            grid_size: Tuple of (rows, cols) for the soft grid. 
                      Recommended: (5, 5) or (6, 6) for good spatial resolution.
            smoothing_sigma: Gaussian smoothing sigma for heatmap.
                            Higher = more blur, smoother transitions.
            threshold: Binary threshold for mask (if not already binary)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.smoothing_sigma = smoothing_sigma
        self.threshold = threshold
        
        # Define semantic region mappings for 5x5 grid
        self._define_semantic_regions()
    
    def _define_semantic_regions(self):
        """
        Define how grid cells map to semantic regions.
        Uses overlapping regions for smooth transitions.
        """
        # For a 5x5 grid:
        # Rows: 0-1 = upper, 2 = middle, 3-4 = lower
        # Cols: 0-1 = left, 2 = center, 3-4 = right
        
        # Vertical regions (row ranges)
        self.vertical_regions = {
            'upper': (0, 2),    # rows 0, 1
            'middle': (1, 4),   # rows 1, 2, 3 (overlapping)
            'lower': (3, 5)     # rows 3, 4
        }
        
        # For non-overlapping version (cleaner interpretation):
        self.vertical_regions_strict = {
            'upper': (0, int(self.grid_rows * 0.4)),
            'middle': (int(self.grid_rows * 0.4), int(self.grid_rows * 0.6)),
            'lower': (int(self.grid_rows * 0.6), self.grid_rows)
        }
        
        # Horizontal regions (column ranges)
        self.horizontal_regions = {
            'left': (0, 2),     # cols 0, 1
            'center': (1, 4),   # cols 1, 2, 3 (overlapping)
            'right': (3, 5)     # cols 3, 4
        }
        
        self.horizontal_regions_strict = {
            'left': (0, int(self.grid_cols * 0.4)),
            'center': (int(self.grid_cols * 0.4), int(self.grid_cols * 0.6)),
            'right': (int(self.grid_cols * 0.6), self.grid_cols)
        }
    
    def analyze(self, 
                mask: np.ndarray,
                original_image: Optional[np.ndarray] = None,
                use_strict_regions: bool = True) -> LocationAnalysisResult:
        """
        Analyze the spatial location of a crack from its segmentation mask.
        
        Args:
            mask: Crack segmentation mask (grayscale or binary, 0-255)
            original_image: Optional original image for visualization
            use_strict_regions: Use non-overlapping regions for cleaner labels
            
        Returns:
            LocationAnalysisResult with probabilistic location labels
        """
        result = LocationAnalysisResult(grid_size=(self.grid_rows, self.grid_cols))
        
        try:
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            H, W = mask.shape
            
            # Binarize mask
            binary_mask = (mask > self.threshold).astype(np.uint8)
            
            # Get crack pixel coordinates
            crack_coords = np.argwhere(binary_mask > 0)  # Returns (y, x) pairs
            
            if len(crack_coords) == 0:
                result.error = "No crack pixels found in mask"
                return result
            
            # Step 1: Build density grid via area-weighted voting
            density_grid = self._build_density_grid(crack_coords, H, W)
            result.density_grid = density_grid
            
            # Step 2: Apply Gaussian smoothing for heatmap
            heatmap = gaussian_filter(density_grid, sigma=self.smoothing_sigma)
            
            # Normalize heatmap to sum to 1
            if heatmap.sum() > 0:
                heatmap = heatmap / heatmap.sum()
            
            result.heatmap = heatmap
            
            # Step 3: Aggregate to semantic regions
            v_regions = self.vertical_regions_strict if use_strict_regions else self.vertical_regions
            h_regions = self.horizontal_regions_strict if use_strict_regions else self.horizontal_regions
            
            result.vertical_position = self._aggregate_vertical(heatmap, v_regions)
            result.horizontal_position = self._aggregate_horizontal(heatmap, h_regions)
            
            # Step 4: Compute crack endpoints (geometric, before intensity reordering)
            raw_endpoints, _ = self._compute_crack_endpoints(crack_coords, H, W)
            
            # Step 5: Reorder endpoints by intensity
            # Origin (dominant) = higher intensity (deeper crack)
            # Terminus (secondary) = lower intensity (shallower crack)
            result.crack_endpoints, result.propagation_vector, intensities = self._reorder_endpoints_by_intensity(
                raw_endpoints, binary_mask, H, W
            )
            result.origin_intensity = intensities['origin_intensity']
            result.terminus_intensity = intensities['terminus_intensity']
            
            # Step 6: Derive dominant (origin) and secondary (terminus) from intensity-ordered endpoints
            # Dominant = where crack STARTS (high stress origin, deeper)
            # Secondary = where crack ENDS (propagation terminus, shallower)
            result.dominant_location, result.secondary_location = self._get_propagation_locations(result.crack_endpoints)
            
            # Step 7: Format propagation direction
            result.propagation_direction = self._format_propagation_direction(
                result.dominant_location, 
                result.secondary_location,
                result.crack_endpoints
            )
            
            # Step 8: Compute spread type and confidence
            result.spread_type, result.confidence = self._compute_spread_metrics(heatmap)
            
            # Step 9: Additional metrics
            result.centroid = self._compute_centroid(crack_coords, H, W)
            result.coverage_ratio = len(crack_coords) / (H * W)
            result.aspect_ratio, result.orientation = self._compute_orientation(crack_coords)
            
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def _build_density_grid(self, 
                           crack_coords: np.ndarray, 
                           H: int, W: int) -> np.ndarray:
        """
        Build spatial density grid via area-weighted voting.
        
        For each crack pixel (y, x):
        1. Normalize coordinates to [0, 1]
        2. Map to grid cell
        3. Increment cell weight
        
        Args:
            crack_coords: Array of (y, x) crack pixel coordinates
            H, W: Image height and width
            
        Returns:
            Normalized density grid
        """
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        for y, x in crack_coords:
            # Normalize coordinates
            x_norm = x / W
            y_norm = y / H
            
            # Map to grid cell (clamp to valid range)
            gx = min(int(x_norm * self.grid_cols), self.grid_cols - 1)
            gy = min(int(y_norm * self.grid_rows), self.grid_rows - 1)
            
            # Increment cell weight
            grid[gy, gx] += 1
        
        # Normalize to probability distribution
        if grid.sum() > 0:
            grid = grid / grid.sum()
        
        return grid
    
    def _aggregate_vertical(self, 
                           heatmap: np.ndarray,
                           regions: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Aggregate heatmap to vertical position probabilities."""
        result = {}
        total = 0.0
        
        for name, (start_row, end_row) in regions.items():
            # Sum probability in this region
            prob = heatmap[start_row:end_row, :].sum()
            result[name] = float(prob)
            total += prob
        
        # Normalize if using overlapping regions
        if total > 1.0:
            for name in result:
                result[name] /= total
        
        return result
    
    def _aggregate_horizontal(self,
                             heatmap: np.ndarray,
                             regions: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Aggregate heatmap to horizontal position probabilities."""
        result = {}
        total = 0.0
        
        for name, (start_col, end_col) in regions.items():
            # Sum probability in this region
            prob = heatmap[:, start_col:end_col].sum()
            result[name] = float(prob)
            total += prob
        
        # Normalize if using overlapping regions
        if total > 1.0:
            for name in result:
                result[name] /= total
        
        return result
    
    def _get_dominant_location(self,
                               vertical: Dict[str, float],
                               horizontal: Dict[str, float]) -> str:
        """
        Get the dominant location label based on density.
        
        Returns labels like: "upper-right", "middle-center", "lower-left"
        """
        # Find dominant vertical position
        v_dominant = max(vertical.keys(), key=lambda k: vertical[k])
        
        # Find dominant horizontal position
        h_dominant = max(horizontal.keys(), key=lambda k: horizontal[k])
        
        return f"{v_dominant}-{h_dominant}"
    
    def _get_location_from_point(self, x: float, y: float) -> str:
        """
        Convert normalized (x, y) coordinates to a location label.
        
        Uses equal thirds for each region (33.3% each):
        - Vertical: upper (0-33%), middle (33-67%), lower (67-100%)
        - Horizontal: left (0-33%), center (33-67%), right (67-100%)
        
        Args:
            x: Normalized x coordinate (0-1, left to right)
            y: Normalized y coordinate (0-1, top to bottom)
            
        Returns:
            Location string like "upper-left", "middle-center", etc.
        """
        # Equal thirds: 0-0.333, 0.333-0.667, 0.667-1.0
        THIRD_1 = 1.0 / 3.0  # ~0.333
        THIRD_2 = 2.0 / 3.0  # ~0.667
        
        # Determine vertical region from y coordinate
        if y < THIRD_1:
            v_region = 'upper'
        elif y > THIRD_2:
            v_region = 'lower'
        else:
            v_region = 'middle'
        
        # Determine horizontal region from x coordinate
        if x < THIRD_1:
            h_region = 'left'
        elif x > THIRD_2:
            h_region = 'right'
        else:
            h_region = 'center'
        
        return f"{v_region}-{h_region}"
    
    def _get_propagation_locations(self,
                                   endpoints: Dict[str, Tuple[float, float]]) -> Tuple[str, str]:
        """
        Get dominant (origin) and secondary (terminus) locations from crack endpoints.
        
        For propagation analysis:
        - Dominant = where crack STARTS (high stress origin)
        - Secondary = where crack ENDS (propagation terminus)
        
        Args:
            endpoints: Dict with 'start' and 'end' normalized (x, y) coordinates
            
        Returns:
            Tuple of (dominant_location, secondary_location)
        """
        start = endpoints.get('start', (0.5, 0.5))
        end = endpoints.get('end', (0.5, 0.5))
        
        dominant = self._get_location_from_point(start[0], start[1])
        secondary = self._get_location_from_point(end[0], end[1])
        
        return dominant, secondary
    
    def _compute_crack_endpoints(self, 
                                 crack_coords: np.ndarray,
                                 H: int, W: int) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
        """
        Compute crack endpoints using PCA to find the main axis.
        
        Note: This returns geometric endpoints only. Use _reorder_endpoints_by_intensity()
        to determine which end is the origin (deeper) vs terminus (shallower).
        
        Returns:
            endpoints: Dict with 'start' and 'end' normalized coordinates
            propagation_vector: Normalized direction vector (dx, dy)
        """
        if len(crack_coords) < 2:
            return {'start': (0.5, 0.5), 'end': (0.5, 0.5)}, (0.0, 0.0)
        
        # Crack coords are (y, x) pairs
        y_coords = crack_coords[:, 0]
        x_coords = crack_coords[:, 1]
        
        # Find centroid
        cy, cx = np.mean(y_coords), np.mean(x_coords)
        
        # Compute covariance matrix for PCA
        coords_centered = np.column_stack([x_coords - cx, y_coords - cy])
        cov_matrix = np.cov(coords_centered.T)
        
        # Get principal eigenvector (main crack direction)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]  # (dx, dy)
        
        # Project all points onto principal axis
        projections = coords_centered @ principal_axis
        
        # Find extreme points along principal axis
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        # Get actual endpoint coordinates (pixel coords)
        point1 = (int(x_coords[min_idx]), int(y_coords[min_idx]))
        point2 = (int(x_coords[max_idx]), int(y_coords[max_idx]))
        
        # Normalize to [0, 1]
        point1_norm = (float(point1[0] / W), float(point1[1] / H))
        point2_norm = (float(point2[0] / W), float(point2[1] / H))
        
        # Store pixel coordinates for intensity lookup
        endpoints = {
            'start': point1_norm,
            'end': point2_norm,
            'start_px': point1,
            'end_px': point2
        }
        
        # Compute normalized propagation vector (will be updated after intensity reordering)
        dx = point2_norm[0] - point1_norm[0]
        dy = point2_norm[1] - point1_norm[1]
        magnitude = (dx**2 + dy**2) ** 0.5
        if magnitude > 0:
            propagation_vector = (dx / magnitude, dy / magnitude)
        else:
            propagation_vector = (0.0, 0.0)
        
        return endpoints, propagation_vector
    
    def _compute_local_intensity(self,
                                 mask: np.ndarray,
                                 point: Tuple[int, int],
                                 radius: int = 20) -> float:
        """
        Compute average crack intensity in a local neighborhood around a point.
        
        Higher intensity = denser/deeper crack region.
        
        Args:
            mask: Binary segmentation mask
            point: (x, y) pixel coordinates
            radius: Radius of neighborhood to sample
            
        Returns:
            Average intensity (0-1) in the neighborhood
        """
        H, W = mask.shape[:2]
        x, y = point
        
        # Define bounding box
        x1 = max(0, x - radius)
        x2 = min(W, x + radius)
        y1 = max(0, y - radius)
        y2 = min(H, y + radius)
        
        # Extract neighborhood
        neighborhood = mask[y1:y2, x1:x2]
        
        if neighborhood.size == 0:
            return 0.0
        
        # Return mean intensity (for binary mask, this is density of crack pixels)
        return float(np.mean(neighborhood) / 255.0) if mask.max() > 1 else float(np.mean(neighborhood))
    
    def _reorder_endpoints_by_intensity(self,
                                        endpoints: Dict,
                                        mask: np.ndarray,
                                        H: int, W: int) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float], Dict[str, float]]:
        """
        Reorder endpoints so that 'start' (ORIGIN/dominant) is the end with HIGHER intensity
        and 'end' (TERMINUS/secondary) is the end with LOWER intensity.
        
        Theory: Crack originates where stress is highest, creating deeper/wider crack.
        The crack propagates outward, becoming shallower at the terminus.
        
        Args:
            endpoints: Dict with 'start', 'end', 'start_px', 'end_px'
            mask: Segmentation mask for intensity lookup
            H, W: Image dimensions
            
        Returns:
            reordered_endpoints: Dict with 'start' and 'end' (normalized)
            propagation_vector: Direction from start to end
            intensities: Dict with 'origin_intensity' and 'terminus_intensity'
        """
        start_px = endpoints.get('start_px', (0, 0))
        end_px = endpoints.get('end_px', (0, 0))
        
        # Compute local intensity at each endpoint
        intensity_start = self._compute_local_intensity(mask, start_px)
        intensity_end = self._compute_local_intensity(mask, end_px)
        
        # Reorder: higher intensity = origin (start), lower intensity = terminus (end)
        if intensity_start >= intensity_end:
            # Keep current order
            origin_norm = endpoints['start']
            terminus_norm = endpoints['end']
            origin_intensity = intensity_start
            terminus_intensity = intensity_end
        else:
            # Swap: end becomes start (origin)
            origin_norm = endpoints['end']
            terminus_norm = endpoints['start']
            origin_intensity = intensity_end
            terminus_intensity = intensity_start
        
        # Compute propagation vector (from origin to terminus)
        dx = terminus_norm[0] - origin_norm[0]
        dy = terminus_norm[1] - origin_norm[1]
        magnitude = (dx**2 + dy**2) ** 0.5
        if magnitude > 0:
            propagation_vector = (dx / magnitude, dy / magnitude)
        else:
            propagation_vector = (0.0, 0.0)
        
        reordered = {
            'start': origin_norm,
            'end': terminus_norm
        }
        
        intensities = {
            'origin_intensity': origin_intensity,
            'terminus_intensity': terminus_intensity
        }
        
        return reordered, propagation_vector, intensities
    
    def _format_propagation_direction(self,
                                      dominant: str,
                                      secondary: str,
                                      endpoints: Dict[str, Tuple[float, float]]) -> str:
        """
        Format the propagation direction as a readable string.
        
        Uses endpoints to determine the actual direction, combined with
        dominant/secondary regions for semantic meaning.
        """
        start = endpoints.get('start', (0.5, 0.5))
        end = endpoints.get('end', (0.5, 0.5))
        
        # Determine start and end regions from coordinates
        start_v = 'upper' if start[1] < 0.4 else ('lower' if start[1] > 0.6 else 'middle')
        start_h = 'left' if start[0] < 0.4 else ('right' if start[0] > 0.6 else 'center')
        
        end_v = 'upper' if end[1] < 0.4 else ('lower' if end[1] > 0.6 else 'middle')
        end_h = 'left' if end[0] < 0.4 else ('right' if end[0] > 0.6 else 'center')
        
        start_region = f"{start_v}-{start_h}"
        end_region = f"{end_v}-{end_h}"
        
        if start_region == end_region:
            return f"localized at {start_region}"
        else:
            return f"{start_region} → {end_region}"

    def _compute_spread_metrics(self, heatmap: np.ndarray) -> Tuple[str, float]:
        """
        Compute how spread out the crack is across the grid.
        
        Returns:
            spread_type: "localized", "multi-region", or "widespread"
            confidence: How concentrated the crack is (higher = more localized)
        """
        # Flatten heatmap
        probs = heatmap.flatten()
        
        # Compute entropy-like metric
        # Lower entropy = more concentrated = higher confidence
        probs_nonzero = probs[probs > 0]
        if len(probs_nonzero) == 0:
            return "unknown", 0.0
        
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero + 1e-10))
        max_entropy = np.log(len(probs))  # Maximum possible entropy (uniform)
        
        # Normalize entropy to [0, 1] and invert for confidence
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        confidence = 1.0 - normalized_entropy
        
        # Count how many cells have significant probability (> 5% each)
        significant_cells = np.sum(probs > 0.05)
        total_cells = self.grid_rows * self.grid_cols
        spread_ratio = significant_cells / total_cells
        
        # Classify spread type
        if spread_ratio < 0.2:
            spread_type = "localized"
        elif spread_ratio < 0.5:
            spread_type = "multi-region"
        else:
            spread_type = "widespread"
        
        return spread_type, float(confidence)
    
    def _compute_centroid(self, 
                         crack_coords: np.ndarray,
                         H: int, W: int) -> Tuple[float, float]:
        """Compute normalized centroid (x, y) of crack pixels."""
        if len(crack_coords) == 0:
            return (0.5, 0.5)
        
        mean_y = np.mean(crack_coords[:, 0])
        mean_x = np.mean(crack_coords[:, 1])
        
        # Normalize to [0, 1]
        return (float(mean_x / W), float(mean_y / H))
    
    def _compute_orientation(self, crack_coords: np.ndarray) -> Tuple[float, str]:
        """
        Compute aspect ratio and orientation of crack.
        
        Returns:
            aspect_ratio: Width / Height of bounding box
            orientation: "horizontal", "vertical", or "diagonal"
        """
        if len(crack_coords) < 2:
            return 1.0, "unknown"
        
        # Get bounding box
        y_min, y_max = crack_coords[:, 0].min(), crack_coords[:, 0].max()
        x_min, x_max = crack_coords[:, 1].min(), crack_coords[:, 1].max()
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Classify orientation
        if aspect_ratio > 2.0:
            orientation = "horizontal"
        elif aspect_ratio < 0.5:
            orientation = "vertical"
        else:
            orientation = "diagonal"
        
        return float(aspect_ratio), orientation
    
    def _draw_region_grid(self, 
                          image: np.ndarray,
                          endpoints: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Draw region boundary grid lines on the image.
        
        Grid divisions using equal thirds (33.3% each):
        - Vertical: upper (0-33%), middle (33-67%), lower (67-100%)
        - Horizontal: left (0-33%), center (33-67%), right (67-100%)
        
        Args:
            image: BGR image to draw on
            endpoints: Optional crack endpoints to mark
            
        Returns:
            Image with grid lines drawn
        """
        img = image.copy()
        H, W = img.shape[:2]
        
        # Grid boundary positions (equal thirds at 33.3% and 66.7%)
        y_33 = int(H / 3)
        y_67 = int(H * 2 / 3)
        x_33 = int(W / 3)
        x_67 = int(W * 2 / 3)
        
        # Colors
        grid_color = (255, 255, 0)  # Cyan for grid lines
        label_color = (255, 255, 255)  # White for labels
        
        # Draw horizontal lines (dividing upper/middle/lower)
        cv2.line(img, (0, y_33), (W, y_33), grid_color, 2)
        cv2.line(img, (0, y_67), (W, y_67), grid_color, 2)
        
        # Draw vertical lines (dividing left/center/right)
        cv2.line(img, (x_33, 0), (x_33, H), grid_color, 2)
        cv2.line(img, (x_67, 0), (x_67, H), grid_color, 2)
        
        # Add region labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Label positions (center of each region)
        labels = [
            ("UPPER-LEFT", (x_33//2, y_33//2)),
            ("UPPER-CENTER", ((x_33 + x_67)//2, y_33//2)),
            ("UPPER-RIGHT", ((x_67 + W)//2, y_33//2)),
            ("MIDDLE-LEFT", (x_33//2, (y_33 + y_67)//2)),
            ("MIDDLE-CENTER", ((x_33 + x_67)//2, (y_33 + y_67)//2)),
            ("MIDDLE-RIGHT", ((x_67 + W)//2, (y_33 + y_67)//2)),
            ("LOWER-LEFT", (x_33//2, (y_67 + H)//2)),
            ("LOWER-CENTER", ((x_33 + x_67)//2, (y_67 + H)//2)),
            ("LOWER-RIGHT", ((x_67 + W)//2, (y_67 + H)//2)),
        ]
        
        for label, (cx, cy) in labels:
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # Draw background rectangle
            cv2.rectangle(img, 
                         (text_x - 3, text_y - text_size[1] - 3),
                         (text_x + text_size[0] + 3, text_y + 3),
                         (0, 0, 0), -1)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, label_color, thickness)
        
        # Draw endpoints if provided
        if endpoints:
            start = endpoints.get('start', None)
            end = endpoints.get('end', None)
            
            if start:
                sx, sy = int(start[0] * W), int(start[1] * H)
                cv2.circle(img, (sx, sy), 10, (0, 255, 0), -1)  # Green = Start/Dominant (deeper)
                cv2.putText(img, "ORIGIN", (sx + 15, sy), font, 0.6, (0, 255, 0), 2)
            
            if end:
                ex, ey = int(end[0] * W), int(end[1] * H)
                cv2.circle(img, (ex, ey), 10, (0, 0, 255), -1)  # Red = End/Secondary (shallower)
                cv2.putText(img, "TERMINUS", (ex + 15, ey), font, 0.6, (0, 0, 255), 2)
            
            # Draw arrow from start to end
            if start and end:
                cv2.arrowedLine(img, (sx, sy), (ex, ey), (255, 0, 255), 2, tipLength=0.03)
        
        return img
    
    def visualize(self,
                 result: LocationAnalysisResult,
                 original_image: np.ndarray,
                 mask: np.ndarray,
                 output_path: Optional[str] = None,
                 alpha: float = 0.5,
                 blur_sigma: float = 15.0) -> np.ndarray:
        """
        Create a single visualization with 3 panels:
        1. Original Image
        2. Segmentation Mask  
        3. Heatmap Overlay on Original
        
        Plus text annotations showing location analysis results.
        
        Args:
            result: LocationAnalysisResult from analyze()
            original_image: Original BGR image
            mask: Segmentation mask
            output_path: Path to save visualization
            alpha: Transparency for heatmap overlay
            blur_sigma: Gaussian blur for heatmap smoothing
            
        Returns:
            Visualization image as numpy array
        """
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # Resize mask to match original if needed
        if mask_gray.shape[:2] != original_image.shape[:2]:
            mask_gray = cv2.resize(mask_gray, (original_image.shape[1], original_image.shape[0]))
        
        # Create image with grid overlay showing regions + endpoints
        original_with_grid = self._draw_region_grid(
            original_image, 
            endpoints=result.crack_endpoints if result.success else None
        )
        
        # Create heatmap overlay
        heatmap_overlay = self._create_heatmap_overlay_image(
            original_image, mask_gray, alpha, blur_sigma
        )
        
        # Create figure with 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        
        # Panel 1: Original Image with Grid + Endpoints
        axes[0].imshow(cv2.cvtColor(original_with_grid, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image (with Region Grid)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Segmentation Mask
        axes[1].imshow(mask_gray, cmap='gray')
        axes[1].set_title('Crack Segmentation', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Heatmap Overlay
        axes[2].imshow(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Crack Location Heatmap', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add analysis results as text below the images
        analysis_text = self._format_analysis_text(result)
        
        fig.text(0.5, 0.02, analysis_text, ha='center', va='bottom',
                fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='orange', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)  # Make room for text
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Visualization saved: {output_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img_array = img_array[:, :, :3]
        
        plt.close(fig)
        
        return img_array
    
    def _create_heatmap_overlay_image(self,
                                      original_image: np.ndarray,
                                      mask: np.ndarray,
                                      alpha: float = 0.5,
                                      blur_sigma: float = 15.0) -> np.ndarray:
        """Create heatmap overlay on original image."""
        # Apply Gaussian blur to create smooth heatmap
        kernel_size = int(blur_sigma * 6) | 1
        heatmap = cv2.GaussianBlur(mask.astype(np.float32), 
                                   (kernel_size, kernel_size), 
                                   blur_sigma)
        
        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        # Apply JET colormap (blue -> green -> yellow -> red)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create overlay - only apply where there's signal
        overlay = original_image.copy()
        heatmap_mask = (heatmap > 10).astype(np.float32)
        heatmap_mask = np.stack([heatmap_mask] * 3, axis=-1)
        
        overlay = (original_image.astype(np.float32) * (1 - alpha * heatmap_mask) + 
                   heatmap_colored.astype(np.float32) * (alpha * heatmap_mask))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _format_analysis_text(self, result: LocationAnalysisResult) -> str:
        """Format analysis results as a compact text string."""
        v = result.vertical_position
        h = result.horizontal_position
        
        text = (
            f"LOCATION ANALYSIS  |  "
            f"Origin: {result.dominant_location.upper()} (intensity: {result.origin_intensity:.4f})  |  "
            f"Terminus: {result.secondary_location.upper()} (intensity: {result.terminus_intensity:.4f})  |  "
            f"Propagation: {result.propagation_direction}\n"
            f"Vertical → Upper: {v.get('upper', 0)*100:.1f}%  Middle: {v.get('middle', 0)*100:.1f}%  Lower: {v.get('lower', 0)*100:.1f}%  |  "
            f"Horizontal → Left: {h.get('left', 0)*100:.1f}%  Center: {h.get('center', 0)*100:.1f}%  Right: {h.get('right', 0)*100:.1f}%"
        )
        
        return text


class CrackLocationPipeline:
    """
    Complete pipeline for crack location analysis.
    Integrates with existing segmentation models.
    """
    
    def __init__(self,
                 segmentation_model=None,
                 grid_size: Tuple[int, int] = (5, 5),
                 smoothing_sigma: float = 1.0,
                 threshold: int = 100):
        """
        Initialize the location analysis pipeline.
        
        Args:
            segmentation_model: Optional SegmentationModel instance
            grid_size: Grid size for spatial analysis
            smoothing_sigma: Gaussian smoothing for heatmap
            threshold: Binary threshold for mask
        """
        self.segmentation_model = segmentation_model
        self.analyzer = CrackLocationAnalyzer(
            grid_size=grid_size,
            smoothing_sigma=smoothing_sigma,
            threshold=threshold
        )
        self.threshold = threshold
    
    def analyze_image(self,
                     image_path: str,
                     mask: Optional[np.ndarray] = None,
                     output_dir: Optional[str] = None,
                     save_visualization: bool = True) -> Dict[str, Any]:
        """
        Analyze crack location in an image.
        
        Args:
            image_path: Path to input image
            mask: Pre-computed segmentation mask (optional)
            output_dir: Directory to save outputs
            save_visualization: Whether to save visualization
            
        Returns:
            Dictionary with location analysis results
        """
        results = {
            'success': False,
            'image_path': image_path,
            'location_analysis': None,
            'error': None
        }
        
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                results['error'] = f"Could not load image: {image_path}"
                return results
            
            # Get segmentation mask
            if mask is None:
                if self.segmentation_model is None:
                    results['error'] = "No mask provided and no segmentation model available"
                    return results
                
                print("Running segmentation...")
                mask = self.segmentation_model.segment(image_path)
            
            # Binarize mask for analysis
            binary_mask = (mask > self.threshold).astype(np.uint8) * 255
            
            # Run location analysis
            print("Analyzing crack location...")
            analysis_result = self.analyzer.analyze(binary_mask, original_image)
            
            if not analysis_result.success:
                results['error'] = analysis_result.error
                return results
            
            results['location_analysis'] = analysis_result.to_dict()
            results['success'] = True
            
            # Save outputs
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save JSON results
                json_path = os.path.join(output_dir, 'location_analysis.json')
                with open(json_path, 'w') as f:
                    json.dump(results['location_analysis'], f, indent=2)
                print(f"Location analysis saved: {json_path}")
                
                # Save single visualization (Original + Mask + Heatmap Overlay)
                if save_visualization:
                    viz_path = os.path.join(output_dir, 'location_analysis.png')
                    self.analyzer.visualize(
                        result=analysis_result,
                        original_image=original_image,
                        mask=binary_mask,
                        output_path=viz_path,
                        alpha=0.5,
                        blur_sigma=15.0
                    )
                    results['visualization_path'] = viz_path
            
            # Print summary
            self._print_summary(analysis_result)
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return results
    
    def _print_summary(self, result: LocationAnalysisResult):
        """Print a formatted summary of the analysis."""
        v_upper = result.vertical_position.get('upper', 0) * 100
        v_middle = result.vertical_position.get('middle', 0) * 100
        v_lower = result.vertical_position.get('lower', 0) * 100
        h_left = result.horizontal_position.get('left', 0) * 100
        h_center = result.horizontal_position.get('center', 0) * 100
        h_right = result.horizontal_position.get('right', 0) * 100
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                 CRACK LOCATION ANALYSIS                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Origin (Dominant):   {result.dominant_location.upper():<40}║
║  Origin Intensity:    {result.origin_intensity:.6f} (deeper crack){' '*16}║
║  Terminus (Secondary):{result.secondary_location.upper():<41}║
║  Terminus Intensity:  {result.terminus_intensity:.6f} (shallower crack){' '*12}║
║  Propagation:         {result.propagation_direction:<40}║
╠══════════════════════════════════════════════════════════════════╣
║  VERTICAL DISTRIBUTION:                                          ║
║    Upper:  {v_upper:5.1f}%                                              ║
║    Middle: {v_middle:5.1f}%                                              ║
║    Lower:  {v_lower:5.1f}%                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  HORIZONTAL DISTRIBUTION:                                        ║
║    Left:   {h_left:5.1f}%                                              ║
║    Center: {h_center:5.1f}%                                              ║
║    Right:  {h_right:5.1f}%                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Orientation: {result.orientation:<48} ║
║  Coverage: {result.coverage_ratio*100:.4f}%{' '*43}║
╚══════════════════════════════════════════════════════════════════╝
""")


def analyze_crack_location(
    image_path: str,
    mask: Optional[np.ndarray] = None,
    segmentation_model_path: Optional[str] = None,
    grid_size: Tuple[int, int] = (5, 5),
    smoothing_sigma: float = 1.0,
    threshold: int = 100,
    output_dir: Optional[str] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Convenience function to analyze crack location.
    
    Args:
        image_path: Path to input image
        mask: Pre-computed segmentation mask (optional)
        segmentation_model_path: Path to segmentation model (if mask not provided)
        grid_size: Grid size for analysis
        smoothing_sigma: Smoothing for heatmap
        threshold: Binary threshold
        output_dir: Output directory
        device: 'cuda' or 'cpu'
        
    Returns:
        Analysis results dictionary
    """
    # Load segmentation model if needed
    seg_model = None
    if mask is None and segmentation_model_path:
        # Import from pipeline_orchestrator
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline_orchestrator import SegmentationModel
        seg_model = SegmentationModel(segmentation_model_path, device=device)
    
    # Create pipeline
    pipeline = CrackLocationPipeline(
        segmentation_model=seg_model,
        grid_size=grid_size,
        smoothing_sigma=smoothing_sigma,
        threshold=threshold
    )
    
    # Analyze
    return pipeline.analyze_image(
        image_path=image_path,
        mask=mask,
        output_dir=output_dir
    )


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze spatial location of cracks in wall images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with segmentation model
  python crack_location_analyzer.py image.jpg --seg-model path/to/model.pth

  # With pre-computed mask
  python crack_location_analyzer.py image.jpg --mask mask.png

  # Custom grid size
  python crack_location_analyzer.py image.jpg --grid-size 6 6 --smoothing 1.5
"""
    )
    
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--mask', help='Pre-computed segmentation mask (optional)')
    parser.add_argument('--seg-model', 
                       default='Masking_and_Classification_model/pretrained_net_G.pth',
                       help='Path to segmentation model')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[5, 5],
                       help='Grid size (rows cols), default: 5 5')
    parser.add_argument('--smoothing', type=float, default=1.0,
                       help='Gaussian smoothing sigma, default: 1.0')
    parser.add_argument('--threshold', type=int, default=100,
                       help='Binary threshold (0-255), default: 100')
    parser.add_argument('--output-dir', '-o', default='./location_analysis_results',
                       help='Output directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Load mask if provided
    mask = None
    if args.mask:
        if not os.path.exists(args.mask):
            print(f"Error: Mask not found: {args.mask}")
            sys.exit(1)
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    
    # Run analysis
    results = analyze_crack_location(
        image_path=args.image,
        mask=mask,
        segmentation_model_path=args.seg_model if mask is None else None,
        grid_size=tuple(args.grid_size),
        smoothing_sigma=args.smoothing,
        threshold=args.threshold,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if not results['success']:
        print(f"\nError: {results['error']}")
        sys.exit(1)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
