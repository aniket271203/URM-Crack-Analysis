# Crack Analysis Pipeline: Validation & Test Matrix Plan

This document outlines the comprehensive validation strategy for the URM Crack Analysis system. To support a scientific publication, we must demonstrate the system's accuracy and robustness under varying environmental and geometric conditions.

## 1. Objectives
- Validate **Brick Dimension Accuracy**: Accuracy of automated brick detection and scale calibration.
- Validate **Crack Dimension Accuracy**: Precision of length and width measurements in millimeters.
- Assess **Robustness**: Performance consistency across different camera angles, lighting, and zoom levels.

---

## 2. Variables for Testing Matrix

| Parameter | Controlled Values | Description |
|-----------|-------------------|-------------|
| **Camera Angle ($\theta$)** | $90^\circ$ (Orthogonal), $75^\circ$, $60^\circ$, $45^\circ$ | Measures tilt-robustness and perspective correction capability. |
| **Lighting (L)** | Bright Daylight, Overcast, Low Light, Artificial (Shadowed) | Measures robustness to contrast and texture variations. |
| **Exposure (E)** | Underexposed (-1EV), Normal (0EV), Overexposed (+1EV) | Tests the sensitivity of segmentation to brightness levels. |
| **Zoom/Distance (D)** | 1m (High Detail), 3m (Standard), 5m (Wide View) | Tests scale-invariance and resolution dependency. |
| **Brick Type (B)** | Modular, Non-modular, Painted, weathered | Tests the robustness of the brick detection algorithms (Hough, FFT). |

---

## 3. The Validation Matrix ($4 \times 4 \times 3$)

To achieve statistical significance, a subset of the most critical combinations should be processed:

| Test ID | Angle | Lighting | Distance | Target Metrics |
|---------|-------|----------|----------|----------------|
| V-01-A | $90^\circ$ | Bright | 1m | Baseline Accuracy (Optimal) |
| V-01-B | $90^\circ$ | Overcast | 3m | Standard Field Condition |
| V-01-C | $90^\circ$ | Low Light | 3m | Adverse Condition (Texture Loss) |
| V-02-A | $75^\circ$ | Normal | 2m | Perspective Test (Minor) |
| V-02-B | $60^\circ$ | Normal | 2m | Perspective Test (Moderate) |
| V-02-C | $45^\circ$ | Normal | 2m | Perspective Test (Extreme - Failure Limit) |
| V-03-A | $90^\circ$ | Shadows | 2m | Non-uniform lighting test |
| V-04-A | $90^\circ$ | Normal | 5m | Resolution/Pixel-density limit test |

---

## 4. Ground Truth Collection (The "Gold Standard")

For each test image, the following physical measurements must be recorded using high-precision instruments (e.g., Digital Caliper with 0.1mm accuracy):

1. **Brick Dimensions**: 
   - $L_{phys}$: Real length of 5 sample bricks.
   - $H_{phys}$: Real height of 5 sample bricks.
2. **Crack Dimensions**:
   - $L_{crack}$: Total length measured along the crack skeleton.
   - $W_{max}$: Maximum width at the widest point.
   - $W_{avg}$: Average width across 5 distinct points.

---

## 5. Performance Metrics for Measurement

To evaluate the system, we will use the following error metrics:

### A. Mean Absolute Percentage Error (MAPE)
$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{Measurement_{pred} - Measurement_{true}}{Measurement_{true}} \right|$$
*Target Move: < 5% for optimal conditions, < 10% for adverse.*

### B. Root Mean Square Error (RMSE)
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Measurement_{pred} - Measurement_{true})^2}$$
*High RMSE indicates lack of robustness (outliers).*

### C. Scale Bias
Calculates if the system consistently overestimates or underestimates based on the angle.

---

## 6. Sensitivity Analysis
The paper should include a "Sensitivity Analysis" section. This determines how much the output changes with respect to small changes in input parameters.

### A. Angle Sensitivity
- Plot **Measurement Error** vs **Angle**. 
- Identify the "Critical Angle" beyond which the error exceeds 10%. 
- This defines the "Operating Envelope" of your system.

### B. Lighting Sensitivity
- Compare results under "Uniform" vs "Non-uniform" (shadowed) lighting.
- Report the **Standard Deviation** of measurements for the same crack under different lightings. Low SD = High Robustness.

---

## 7. Implementation of the Test Runner

We have provided a specific script `Benchmark/evaluate_measurements.py` that:
1. Matches your ground truth CSV with the system's JSON outputs.
2. Calculates **MAPE** (Mean Absolute Percentage Error) and **RMSE** (Root Mean Square Error).
3. Generates **Parity Plots** (Predicted vs Ground Truth) which are essential for publication.
4. Generates **Error vs. Angle** boxplots to show robustness.

---

## 8. Step-by-Step Validation Workflow

1. **Setup**: Select a wall with 3-5 distinct cracks.
2. **Measure**: Physically measure each crack and brick using a caliper.
3. **Template**: Fill the `Benchmark/measurement_ground_truth_template.csv` with these values.
4. **Capture**: Take photos following the combinations in the [Validation Matrix](#3-the-validation-matrix-4--4--3).
5. **Run Pipeline**: Process all images using `batch_process_darbhanga.py` (or similar orchestrator) to generate `measurements.json` for each.
6. **Evaluate**: Run the evaluation script:
   ```bash
   python Benchmark/evaluate_measurements.py --gt ground_truth.csv --results ./darbhanga_results/intermediate/
   ```
7. **Publish**: Copy the generated plots and summary table into your LaTeX document.
