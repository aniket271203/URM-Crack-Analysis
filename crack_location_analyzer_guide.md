# Crack Location Analyzer - Technical Guide

## Overview

The **Crack Location Analyzer** is a spatial analysis module that determines where cracks are located within a wall image and identifies their propagation direction. It uses a combination of soft spatial grids, Gaussian heatmap smoothing, and intensity-based endpoint analysis to provide both **positional classification** and **propagation direction** of structural cracks.

---

## Key Outputs

| Field | Description |
|-------|-------------|
| `dominant_location` | Origin of crack (deeper end, e.g., "upper-left") |
| `secondary_location` | Terminus of crack (shallower end, e.g., "middle-right") |
| `propagation_direction` | Direction string (e.g., "upper-left → middle-right") |
| `origin_intensity` | Crack density at origin (higher = deeper crack) |
| `terminus_intensity` | Crack density at terminus (lower = shallower crack) |
| `vertical_position` | Probability distribution: upper/middle/lower |
| `horizontal_position` | Probability distribution: left/center/right |

---

## Strategy & Approach

### 1. Spatial Grid Division (Equal Thirds)

The image is divided into a **3×3 semantic grid** using equal thirds:

```
┌─────────────┬─────────────┬─────────────┐
│ UPPER-LEFT  │UPPER-CENTER │ UPPER-RIGHT │
│   (0-33%)   │  (33-67%)   │  (67-100%)  │
├─────────────┼─────────────┼─────────────┤
│ MIDDLE-LEFT │MIDDLE-CENTER│MIDDLE-RIGHT │
│   (0-33%)   │  (33-67%)   │  (67-100%)  │
├─────────────┼─────────────┼─────────────┤
│ LOWER-LEFT  │LOWER-CENTER │ LOWER-RIGHT │
│   (0-33%)   │  (33-67%)   │  (67-100%)  │
└─────────────┴─────────────┴─────────────┘
        ← Horizontal (X) →
```

**Thresholds:**
- Vertical (Y-axis): `upper < 0.333`, `0.333 ≤ middle ≤ 0.667`, `lower > 0.667`
- Horizontal (X-axis): `left < 0.333`, `0.333 ≤ center ≤ 0.667`, `right > 0.667`

### 2. Crack Pixel Density Analysis

1. **Segmentation**: The crack segmentation mask identifies all crack pixels
2. **Density Grid**: A fine-grained 5×5 grid accumulates crack pixel counts per cell
3. **Gaussian Smoothing**: Applied to create a smooth probability heatmap
4. **Normalization**: Heatmap values sum to 1.0 for probabilistic interpretation

```python
# Density calculation
density_grid[row, col] = count of crack pixels in cell / total crack pixels

# Smoothing
heatmap = gaussian_filter(density_grid, sigma=1.0)
heatmap = heatmap / heatmap.sum()  # Normalize
```

### 3. Crack Endpoint Detection (PCA-based)

To find the two ends of the crack:

1. **Collect all crack pixel coordinates** from the segmentation mask
2. **Apply PCA** (Principal Component Analysis) to find the main axis of the crack
3. **Project all pixels** onto the principal axis
4. **Identify extremes**: The pixels with minimum and maximum projections are the endpoints

```python
# PCA to find main crack direction
coords_centered = crack_coords - centroid
cov_matrix = np.cov(coords_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
principal_axis = eigenvectors[:, argmax(eigenvalues)]

# Find endpoints
projections = coords_centered @ principal_axis
endpoint1 = crack_coords[argmin(projections)]
endpoint2 = crack_coords[argmax(projections)]
```

### 4. Intensity-Based Origin/Terminus Classification

**Theory**: Cracks originate where stress is highest, creating deeper/wider cracks. As the crack propagates, it becomes shallower at the terminus.

**Implementation**:
1. Compute **local intensity** at each endpoint (average crack density in a neighborhood)
2. **Higher intensity endpoint** = **Origin** (dominant location, deeper crack)
3. **Lower intensity endpoint** = **Terminus** (secondary location, shallower crack)

```python
def compute_local_intensity(mask, point, radius=20):
    """Average crack pixel density in neighborhood around point"""
    neighborhood = mask[y-radius:y+radius, x-radius:x+radius]
    return mean(neighborhood)

# Determine origin vs terminus
intensity_1 = compute_local_intensity(mask, endpoint1)
intensity_2 = compute_local_intensity(mask, endpoint2)

if intensity_1 > intensity_2:
    origin = endpoint1      # Deeper crack
    terminus = endpoint2    # Shallower crack
else:
    origin = endpoint2
    terminus = endpoint1
```

### 5. Propagation Direction

The propagation direction is formatted as:
```
"{origin_region} → {terminus_region}"
```

For example: `"upper-left → middle-right"`

This indicates:
- Crack **started** at upper-left (high stress origin)
- Crack **propagated to** middle-right (stress dissipation)

---

## Algorithm Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│   Crack Segmentation        │ ← Uses pretrained U-Net model
│   (Binary Mask)             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Density Grid (5×5)        │ ← Count crack pixels per cell
│   + Gaussian Smoothing      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Semantic Aggregation      │ ← upper/middle/lower
│   (Vertical & Horizontal)   │   left/center/right
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   PCA Endpoint Detection    │ ← Find crack extremities
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Intensity-Based Ordering  │ ← Higher intensity = Origin
│   Origin → Terminus         │   Lower intensity = Terminus
└─────────────────────────────┘
    │
    ▼
Output: Location Analysis Result
```

---

## Output Files

| File | Description |
|------|-------------|
| `location_analysis.json` | Full analysis data in JSON format |
| `location_analysis.png` | 3-panel visualization (Original + Mask + Heatmap) |

### Visualization Panels

1. **Original Image with Grid**: Shows 3×3 region boundaries, origin (green), terminus (red), and propagation arrow
2. **Crack Segmentation**: Binary mask showing detected crack pixels
3. **Crack Location Heatmap**: Color-coded density overlay (blue=low, red=high)

---

## Usage

### Command Line

```bash
# Basic usage
python crack_location_analyzer.py <image_path> --output-dir <output_directory>

# Example
python crack_location_analyzer.py Test/test_98.jpg --output-dir results/

# With custom threshold
python crack_location_analyzer.py image.jpg --output-dir output/ --threshold 100
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `image_path` | Path to input image | Required |
| `--output-dir` | Output directory for results | `./crack_location_output` |
| `--threshold` | Binary threshold for mask | `127` |
| `--model-path` | Custom segmentation model path | Built-in model |

### Python API

```python
from crack_location_analyzer import CrackLocationPipeline, analyze_crack_location

# Option 1: Using the pipeline class
pipeline = CrackLocationPipeline(
    grid_size=(5, 5),
    smoothing_sigma=1.0,
    threshold=127
)

results = pipeline.analyze_image(
    image_path="path/to/image.jpg",
    output_dir="output/",
    save_visualization=True
)

print(f"Origin: {results['analysis']['dominant_location']}")
print(f"Terminus: {results['analysis']['secondary_location']}")
print(f"Propagation: {results['analysis']['propagation_direction']}")

# Option 2: Using convenience function
results = analyze_crack_location(
    image_path="path/to/image.jpg",
    output_dir="output/",
    device='cuda'  # or 'cpu'
)
```

### Using Pre-computed Mask

```python
import cv2
from crack_location_analyzer import CrackLocationAnalyzer

# Load your own segmentation mask
mask = cv2.imread("crack_mask.png", cv2.IMREAD_GRAYSCALE)
original = cv2.imread("original_image.jpg")

# Analyze
analyzer = CrackLocationAnalyzer(grid_size=(5, 5))
result = analyzer.analyze(mask, original_image=original)

print(f"Origin: {result.dominant_location} (intensity: {result.origin_intensity:.4f})")
print(f"Terminus: {result.secondary_location} (intensity: {result.terminus_intensity:.4f})")
print(f"Propagation: {result.propagation_direction}")
```

---

## Example Output

### Console Output
```
╔══════════════════════════════════════════════════════════════════╗
║                 CRACK LOCATION ANALYSIS                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Origin (Dominant):   UPPER-LEFT                                  ║
║  Origin Intensity:    0.007258 (deeper crack)                     ║
║  Terminus (Secondary):MIDDLE-RIGHT                                ║
║  Terminus Intensity:  0.000862 (shallower crack)                  ║
║  Propagation:         upper-left → middle-right                   ║
╠══════════════════════════════════════════════════════════════════╣
║  VERTICAL DISTRIBUTION:                                           ║
║    Upper:   35.5%                                                 ║
║    Middle:  25.7%                                                 ║
║    Lower:   38.9%                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  HORIZONTAL DISTRIBUTION:                                         ║
║    Left:    33.8%                                                 ║
║    Center:  18.5%                                                 ║
║    Right:   47.7%                                                 ║
╚══════════════════════════════════════════════════════════════════╝
```

### JSON Output
```json
{
  "success": true,
  "dominant_location": "upper-left",
  "secondary_location": "middle-right",
  "propagation_direction": "upper-left → middle-right",
  "origin_intensity": 0.007258,
  "terminus_intensity": 0.000862,
  "vertical_position": {
    "upper": 0.355,
    "middle": 0.257,
    "lower": 0.389
  },
  "horizontal_position": {
    "left": 0.338,
    "center": 0.185,
    "right": 0.477
  },
  "crack_endpoints": {
    "start": [0.014, 0.267],
    "end": [0.988, 0.495]
  },
  "orientation": "diagonal",
  "coverage_ratio": 0.0042
}
```

---

## Interpretation Guide

### Intensity Ratio
- **High ratio** (origin >> terminus): Clear propagation direction, crack originated from a stressed point
- **Low ratio** (origin ≈ terminus): Uniform crack depth, possibly stress from multiple sources

### Coverage Ratio
- `< 0.5%`: Minor surface crack
- `0.5% - 2%`: Moderate crack
- `> 2%`: Significant structural crack

### Orientation
- `horizontal`: Crack runs left-right (possible settlement or load distribution issues)
- `vertical`: Crack runs top-bottom (possible foundation or structural issues)
- `diagonal`: Crack runs diagonally (possible shear stress or differential settlement)

---

## Dependencies

```
numpy
opencv-python
scipy
matplotlib
torch (for segmentation model)
```

---

## Files

| File | Purpose |
|------|---------|
| `crack_location_analyzer.py` | Main analyzer module |
| `Masking_and_Classification_model/pretrained_net_G.pth` | Segmentation model |
| `Masking_and_Classification_model/model_utils.py` | Model utilities |

---

## Theory: Why Intensity Indicates Crack Origin

1. **Stress Concentration**: Cracks initiate at points of maximum stress
2. **Energy Release**: Initial fracture releases significant energy, creating deeper/wider cracks
3. **Propagation Attenuation**: As the crack propagates, stress is redistributed
4. **Terminus Formation**: The crack tip becomes shallower as energy dissipates

This physical principle allows us to use **local crack intensity** (pixel density) as a proxy for **crack depth**, enabling identification of the crack origin vs. terminus.
