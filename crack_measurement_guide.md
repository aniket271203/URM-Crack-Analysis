# Brick-Calibrated Crack Measurement System - Technical Guide

## Overview

The **Brick-Calibrated Crack Measurement System** measures real-world crack dimensions (length, width, area) without requiring physical markers like ArUco codes. It uses the **brick pattern** visible in masonry wall images to establish a pixel-to-millimeter scale factor.

---

## Key Outputs

| Metric | Description | Unit |
|--------|-------------|------|
| `length_mm` | Total crack length (via skeletonization) | mm |
| `max_width` | Maximum crack width | mm |
| `mean_width` | Average crack width | mm |
| `median_width` | Median crack width | mm |
| `area_mm2` | Total crack area | mm² |
| `scale_mm_per_px` | Calibrated scale factor | mm/pixel |

---

## Why Brick HEIGHT Instead of Length?

The system focuses on measuring **brick HEIGHT** for calibration because:

1. **Brick heights are standardized** - All bricks in a wall have the same height
2. **Lengths vary** - Bricks are often cut at corners, around openings, or in bond patterns
3. **Horizontal mortar lines are clearer** - They span the entire wall width
4. **Less affected by perspective** - Height measurement is more robust to camera angle

```
Standard Brick Example (Indian Modular):
┌──────────────────────────────────┐
│                                  │  ← Height = 90mm (CONSISTENT)
│         BRICK                    │
│                                  │
└──────────────────────────────────┘
  ↑                              ↑
  Length = 190mm (but may be cut)
```

---

## Supported Brick Standards

| Type | Name | Length (mm) | Height (mm) | Mortar (mm) |
|------|------|-------------|-------------|-------------|
| `india_modular` | Indian Modular | 190 | 90 | 10 |
| `india_traditional` | Indian Traditional | 230 | 75 | 12 |
| `uk_standard` | UK Standard | 215 | 65 | 10 |
| `us_standard` | US Standard | 194 | 57 | 10 |
| `europe_nf` | European NF | 240 | 71 | 10 |

---

## Brick Detection Strategies

The system uses **three detection methods** and selects the one with highest confidence:

### Method 1: Mortar Line Detection (Primary)

Detects horizontal mortar lines using Hough transforms and projection analysis.

```
Algorithm:
1. Convert to grayscale and enhance contrast (CLAHE)
2. Apply Canny edge detection
3. Morphological operations to isolate horizontal lines
4. Project edges onto Y-axis (horizontal projection)
5. Find peaks in projection (= mortar line positions)
6. Calculate spacing between peaks = brick height in pixels
```

```python
# Simplified implementation
edges = cv2.Canny(enhanced_image, 30, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))  # Horizontal
horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Project to find line positions
h_projection = np.sum(horizontal_lines, axis=1)
peaks = find_peaks(h_projection, distance=min_brick_height)

# Brick height = median spacing between peaks
brick_height_px = np.median(np.diff(peaks))
```

**Confidence Calculation:**
```python
spacing_cv = std(spacings) / mean(spacings)  # Coefficient of variation
confidence = (1 - spacing_cv) * min(num_peaks / 5, 1.0)
```

### Method 2: Frequency Analysis (FFT)

Uses Fast Fourier Transform to detect periodic brick patterns.

```
Algorithm:
1. Apply 2D FFT to grayscale image
2. Analyze vertical frequencies (corresponds to horizontal pattern)
3. Find dominant frequency peak
4. Convert frequency to spatial period (pixels)
```

```python
# FFT analysis
f = np.fft.fft2(gray_image)
fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift)

# Analyze vertical frequency (for horizontal pattern = brick courses)
center_col = magnitude[:, width//2]
peaks = find_peaks(center_col, prominence=0.1 * max(center_col))

# Convert dominant frequency to brick height
dominant_freq = peaks[np.argmax(center_col[peaks])]
brick_height_px = image_height / dominant_freq
```

### Method 3: Edge Detection (Sobel)

Backup method using Sobel edge detection.

```
Algorithm:
1. Apply Sobel filter in Y direction (detects horizontal edges)
2. Project Sobel response onto Y-axis
3. Find peaks (= mortar line positions)
4. Calculate brick height from peak spacing
```

```python
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
h_projection = np.sum(np.abs(sobel_y), axis=1)
peaks = find_peaks(h_projection, distance=min_brick_height)
brick_height_px = np.median(np.diff(peaks))
```

### Method Selection

The system runs all three methods and selects based on confidence:

```python
results = [mortar_detection, frequency_detection, edge_detection]
best_result = max(results, key=lambda r: r.confidence)
```

---

## Scale Factor Calculation

Once brick height in pixels is detected:

```python
# Known brick dimensions (from standard)
brick_height_mm = 90  # Indian Modular

# Detected brick height in pixels
brick_height_px = 85  # From detection

# Scale factor
scale_mm_per_px = brick_height_mm / brick_height_px
# Example: 90 / 85 = 1.059 mm/pixel
```

---

## Crack Measurement Methods

### 1. Crack Length (Skeletonization)

The crack length is measured by **skeletonizing** the binary mask to a 1-pixel-wide representation.

```
Original Mask:          Skeleton:
████████████            ············
████████████    →       ············
████████████            ············
        ████                    ····
        ████                    ····
```

```python
from skimage.morphology import skeletonize

# Skeletonize binary mask
skeleton = skeletonize(binary_mask > 0)

# Length = count of skeleton pixels × scale factor
length_pixels = np.sum(skeleton)
length_mm = length_pixels * scale_mm_per_px
```

**Note:** Diagonal skeleton pixels represent √2 ≈ 1.41 units of length, but this is approximated as 1 pixel for simplicity.

### 2. Crack Width (Distance Transform)

Width is calculated using the **Euclidean Distance Transform (EDT)**.

```
Distance Transform:
┌─────────────────┐
│ 0 0 0 0 0 0 0 0 │  ← Background = 0
│ 0 1 1 1 1 1 1 0 │  ← Edge = 1
│ 0 1 2 2 2 2 1 0 │  ← Interior = distance to edge
│ 0 1 2 3 3 2 1 0 │
│ 0 1 2 2 2 2 1 0 │
│ 0 1 1 1 1 1 1 0 │
│ 0 0 0 0 0 0 0 0 │
```

```python
from scipy.ndimage import distance_transform_edt

# Compute distance transform
dist_transform = distance_transform_edt(binary_mask)

# Sample widths along skeleton
skeleton_coords = np.where(skeleton)
radii_px = dist_transform[skeleton_coords]  # Radius at each skeleton point
widths_px = radii_px * 2  # Diameter = 2 × radius

# Convert to mm
widths_mm = widths_px * scale_mm_per_px

# Statistics
max_width = np.max(widths_mm)
mean_width = np.mean(widths_mm)
median_width = np.median(widths_mm)
```

### 3. Crack Area

Area is simply the count of crack pixels converted to mm².

```python
crack_pixels = np.sum(binary_mask > 0)
area_mm2 = crack_pixels * (scale_mm_per_px ** 2)
```

---

## Pipeline Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│   BRICK DETECTION               │
│   ┌───────────────────────────┐ │
│   │ Method 1: Mortar Lines    │ │
│   │ Method 2: FFT Frequency   │ │
│   │ Method 3: Sobel Edges     │ │
│   └───────────────────────────┘ │
│   → Select highest confidence   │
│   → Calculate scale factor      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   CRACK SEGMENTATION            │
│   (U-Net based model)           │
│   → Raw mask → Binary mask      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   CRACK MEASUREMENT             │
│   ┌───────────────────────────┐ │
│   │ Skeletonization → Length  │ │
│   │ Distance Transform → Width│ │
│   │ Pixel Count → Area        │ │
│   └───────────────────────────┘ │
│   → Apply scale factor          │
└─────────────────────────────────┘
    │
    ▼
Output: Measurements in mm/mm²
```

---

## Output Files

| File | Description |
|------|-------------|
| `brick_detection.jpg` | Visualization of detected brick grid |
| `segmentation_raw.png` | Raw segmentation mask from model |
| `segmentation_binary.png` | Thresholded binary mask |
| `skeleton_only.png` | White skeleton on black background |
| `skeleton_overlay.png` | Skeleton (red) on original image |
| `skeleton_analysis.png` | Comprehensive skeleton analysis |
| `measurements.json` | Full measurements in JSON format |
| `measurements.txt` | Human-readable summary |
| `reference_grid.jpg` | Reference grid for manual calibration |

---

## Usage

### Command Line Interface

```bash
# Basic usage with auto-detection (Indian Modular bricks)
python brick_calibrated_measurement.py <image_path>

# Specify brick type
python brick_calibrated_measurement.py image.jpg --brick-type uk_standard

# Custom brick dimensions
python brick_calibrated_measurement.py image.jpg --brick-length 200 --brick-height 80

# Interactive mode (manually measure a brick)
python brick_calibrated_measurement.py image.jpg --interactive

# Full options
python brick_calibrated_measurement.py image.jpg \
    --brick-type india_modular \
    --threshold 100 \
    --output-dir ./results \
    --device cuda \
    --show-detection
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `image` | Input image path | Required |
| `--brick-type` | Standard brick type | `india_modular` |
| `--brick-length` | Custom brick length (mm) | From brick-type |
| `--brick-height` | Custom brick height (mm) | From brick-type |
| `--mortar-width` | Mortar joint width (mm) | `10` |
| `--interactive` | Manual brick selection | `False` |
| `--seg-model` | Segmentation model path | Built-in |
| `--device` | Inference device | `cuda` |
| `--threshold` | Binary threshold (0-255) | `100` |
| `--output-dir` | Output directory | `./brick_calibrated_results` |
| `--show-detection` | Display detection result | `False` |

### Python API

```python
from brick_calibrated_measurement import (
    BrickDetector, 
    CrackMeasurerWithBrickCalibration,
    BRICK_STANDARDS
)
import cv2

# Load image
image = cv2.imread("wall_with_crack.jpg")

# Initialize detector with brick dimensions
brick_info = BRICK_STANDARDS['india_modular']
detector = BrickDetector(
    brick_length_mm=brick_info['length_mm'],
    brick_height_mm=brick_info['height_mm']
)

# Detect bricks and get scale factor
detection = detector.detect(image)
if detection.success:
    print(f"Scale: {detection.avg_scale_mm_per_px:.4f} mm/pixel")
    print(f"Confidence: {detection.confidence:.2%}")

# Measure crack (assuming you have a binary mask)
measurer = CrackMeasurerWithBrickCalibration()
results = measurer.measure(image, binary_mask, detection.avg_scale_mm_per_px)

print(f"Length: {results['length_mm']:.2f} mm")
print(f"Max Width: {results['width_stats']['max_mm']:.2f} mm")
print(f"Area: {results['area_mm2']:.2f} mm²")
```

### Interactive Mode

When automatic detection fails, use interactive mode:

```bash
python brick_calibrated_measurement.py image.jpg --interactive
```

**GUI Mode (if available):**
1. Click on the TOP edge of a brick
2. Click on the BOTTOM edge of the same brick
3. Close the window

**Terminal Mode (fallback):**
1. A reference grid image is saved
2. Open the image and measure brick height in pixels
3. Enter the value when prompted

---

## Example Output

### Console Output
```
╔══════════════════════════════════════════════════════════════════╗
║                    CRACK MEASUREMENTS                             ║
║              (Brick-Calibrated - REAL Dimensions)                 ║
╠══════════════════════════════════════════════════════════════════╣
║  CALIBRATION:                                                    ║
║    Brick type: Indian Modular Brick                              ║
║    Reference: 190 x 90 mm = 180 x 85 px                          ║
║    Scale: 1.0588 mm/pixel (confidence: 87%)                      ║
╠══════════════════════════════════════════════════════════════════╣
║  CRACK DIMENSIONS:                                               ║
║    Length:           245.32 mm                                   ║
║    Max Width:          3.45 mm                                   ║
║    Mean Width:         1.87 mm                                   ║
║    Median Width:       1.62 mm                                   ║
║    Area:             458.91 mm²                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### JSON Output
```json
{
  "success": true,
  "length_mm": 245.32,
  "area_mm2": 458.91,
  "width_stats": {
    "max_mm": 3.45,
    "min_mm": 0.53,
    "mean_mm": 1.87,
    "median_mm": 1.62,
    "std_mm": 0.68,
    "p95_mm": 3.12
  },
  "scale_mm_per_px": 1.0588,
  "calibration": {
    "method": "mortar_lines_height",
    "brick_type": "Indian Modular Brick",
    "brick_height_mm": 90,
    "brick_height_px": 85,
    "confidence": 0.87
  }
}
```

---

## Accuracy Considerations

### Factors Affecting Accuracy

1. **Brick Type Selection**
   - Using wrong brick dimensions will scale all measurements incorrectly
   - Verify brick type matches the actual wall

2. **Detection Confidence**
   - Low confidence (< 50%) indicates unreliable detection
   - Use interactive mode for difficult images

3. **Camera Angle / Perspective**
   - Non-perpendicular camera angle introduces distortion
   - Best results with camera parallel to wall surface

4. **Image Quality**
   - Low resolution reduces measurement precision
   - Motion blur affects edge detection

5. **Segmentation Threshold**
   - Too low: Captures noise, inflates measurements
   - Too high: Misses thin crack sections
   - Check `skeleton_analysis.png` to verify threshold

### Recommended Workflow

1. **First Run**: Use auto-detection
   ```bash
   python brick_calibrated_measurement.py image.jpg
   ```

2. **Check Outputs**:
   - `brick_detection.jpg` - Are grid lines aligned with actual bricks?
   - `skeleton_analysis.png` - Is skeleton continuous along crack?

3. **Adjust if Needed**:
   ```bash
   # If detection failed, use interactive mode
   python brick_calibrated_measurement.py image.jpg --interactive
   
   # If skeleton is broken, lower threshold
   python brick_calibrated_measurement.py image.jpg --threshold 80
   
   # If too much noise, raise threshold
   python brick_calibrated_measurement.py image.jpg --threshold 120
   ```

---

## Threshold Calibration

The `--threshold` parameter controls binary mask creation:

| Threshold | Effect | When to Use |
|-----------|--------|-------------|
| Low (50-80) | More pixels classified as crack | Faint/thin cracks |
| Medium (80-120) | Balanced | Most cases |
| High (120-150) | Fewer pixels, only strong signals | Noisy images |

**Visual Check**: Open `skeleton_analysis.png` to see:
- Is the skeleton continuous along the visible crack?
- Are there disconnected segments (threshold too high)?
- Is there noise/extra regions (threshold too low)?

---

## Dependencies

```
numpy
opencv-python
scipy
scikit-image
matplotlib
torch (for segmentation model)
```

---

## Files

| File | Purpose |
|------|---------|
| `brick_calibrated_measurement.py` | Main measurement module |
| `pipeline_orchestrator.py` | Segmentation model wrapper |
| `Masking_and_Classification_model/pretrained_net_G.pth` | U-Net segmentation model |

---

## Theory: Why This Approach Works

### Brick Pattern as Natural Scale Reference

Unlike ArUco markers which require physical placement, brick walls contain a **built-in scale reference**:

1. **Bricks are manufactured to standard sizes** - Industrial bricks have consistent dimensions
2. **Mortar joints are visible** - Create clear horizontal lines across the wall
3. **Pattern is periodic** - Multiple measurements improve accuracy

### Height vs Length Trade-off

| Dimension | Pros | Cons |
|-----------|------|------|
| Height | Consistent, clear mortar lines | Requires multiple brick courses |
| Length | More bricks visible | Variable (cuts, bonds, headers) |

**Conclusion**: Height-based calibration is more reliable.

### Skeleton-Based Length Measurement

Skeletonization reduces a region to its **medial axis** (centerline):

- Captures the true "path length" of the crack
- Independent of crack width
- Each skeleton pixel represents 1 unit of linear distance

### Distance Transform for Width

The EDT computes distance to nearest boundary:

- At the skeleton (centerline), EDT value = half-width
- Diameter = 2 × EDT value at skeleton point
- Gives width at every point along the crack
