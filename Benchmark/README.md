# Crack Analysis Pipeline Benchmarking System

This directory contains a comprehensive benchmarking framework for evaluating the performance of the crack analysis pipeline across all components: YOLO Detection, Segmentation, Classification, and RAG Analysis.

## 🎯 Overview

The benchmarking system provides:
- **Systematic Test Data Generation**: Creates artificial crack images with ground truth
- **Multi-Model Evaluation**: Tests YOLO detection, segmentation, and classification
- **Performance Metrics**: Comprehensive evaluation with industry-standard metrics
- **Visualizations**: Detailed plots and interactive reports
- **Failure Analysis**: Identifies and analyzes failure cases

## 📁 Structure

```
Benchmark/
├── README.md                        # This documentation
├── setup_benchmark.sh              # Quick setup script
├── generate_simple_test_data.py    # Simple test data generator (working)
├── simple_benchmark.py             # Simplified benchmark runner (working)
├── generate_test_data.py           # Advanced test data generator
├── benchmark_pipeline.py           # Full benchmarking suite
├── evaluation_metrics.py           # Metrics calculation utilities
├── visualization.py                # Plotting and visualization
├── config_benchmark.py             # Configuration settings
├── test_data/                      # Generated test dataset
│   ├── images/                     # Test images (300 total)
│   ├── annotations/                # YOLO format annotations
│   └── metadata/                   # Ground truth and metadata
├── results/                        # Benchmark results
└── reports/                        # Generated reports and plots
```

## 🎯 Test Dataset Composition

### Crack Types Distribution
- **Vertical Cracks**: 50 images (temperature-induced expansion/contraction)
- **Horizontal Cracks**: 50 images (flexural stress, beam overloading) 
- **Diagonal Cracks**: 50 images (foundation settlement, shear stress)
- **Step Cracks**: 50 images (differential movement in masonry)
- **No Cracks**: 100 images (negative samples for false positive testing)
- **Total**: 300 images

### Image Characteristics
- **Base Materials**: Real brick wall images from DATA_bricks directory
- **Crack Severities**: Light, medium, severe (varying thickness 2-10 pixels)
- **Augmentations**: Random noise, blur, brightness/contrast variations
- **Annotations**: YOLO format bounding boxes + metadata
- **Ground Truth**: Complete labels for detection, classification, and segmentation

## 🔬 Evaluation Metrics

### 1. YOLO Detection Performance
- **Precision**: TP / (TP + FP) - accuracy of positive predictions
- **Recall**: TP / (TP + FN) - ability to find all positive cases
- **F1-Score**: 2 * (Precision × Recall) / (Precision + Recall)
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Detection Rate**: Percentage of crack images correctly identified

### 2. Segmentation Performance
- **IoU (Intersection over Union)**: Overlap between predicted and true masks
- **Dice Coefficient**: 2 * |A ∩ B| / (|A| + |B|)
- **Pixel Accuracy**: Correctly classified pixels / total pixels
- **Mean IoU**: Average IoU across all test images

### 3. Classification Performance
- **Accuracy**: Correct classifications / total classifications
- **Per-Class Precision/Recall**: Performance for each crack type
- **Confusion Matrix**: Detailed error analysis
- **Weighted F1**: F1-score weighted by class frequency

### 4. End-to-End Pipeline Metrics
- **Overall Accuracy**: Complete pipeline correctness
- **Processing Time**: Inference speed per image
- **Throughput**: Images processed per second
- **Memory Usage**: Resource consumption during inference

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the trained models:
# - Crack_Detection_YOLO/crack_yolo_train/weights/best.pt
# - Masking_and_Classification_model/pretrained_net_G.pth  
# - Masking_and_Classification_model/crack_orientation_classifier.h5
```

### Option 1: Automated Setup (Recommended)
```bash
cd Benchmark
./setup_benchmark.sh
```

### Option 2: Manual Setup
```bash
cd Benchmark

# 1. Generate test data
python generate_simple_test_data.py --seed 42

# 2. Run benchmark
python simple_benchmark.py --device cuda

# 3. With RAG analysis (slower)
python simple_benchmark.py --device cuda --use-rag
```

### Advanced Usage
```bash
# CPU-only benchmarking
python simple_benchmark.py --device cpu

# Custom test data directory
python simple_benchmark.py --test-dir /path/to/test_data

# Custom output directory
python simple_benchmark.py --output /path/to/results
```

## 📊 Expected Performance Baselines

Based on the pipeline architecture, expected performance ranges:

### Detection (YOLO)
- **Precision**: 0.85-0.95 (low false positives)
- **Recall**: 0.80-0.90 (good crack detection)
- **F1-Score**: 0.82-0.92 (balanced performance)
- **Processing Time**: 0.1-0.3s per image (GPU)

### Classification
- **Accuracy**: 0.75-0.90 (4-class problem)
- **Vertical/Horizontal**: Higher accuracy (distinct patterns)
- **Diagonal/Step**: Lower accuracy (similar features)

### Pipeline End-to-End
- **Overall Accuracy**: 0.70-0.85 (detection + classification)
- **Processing Time**: 0.5-2.0s per image (full pipeline)
- **Throughput**: 0.5-2.0 images/sec

## 📈 Benchmark Results Interpretation

### Detection Analysis
- **High Precision, Low Recall**: Model is conservative (misses some cracks)
- **Low Precision, High Recall**: Model is aggressive (false positives)
- **Balanced F1**: Good overall detection performance

### Classification Analysis
- **High Accuracy**: Model generalizes well to different crack types
- **Class Imbalance**: Some crack types may be over/under-represented
- **Confusion Patterns**: Reveals which crack types are commonly confused

### Failure Case Analysis
- **False Negatives**: Subtle cracks, poor image quality
- **False Positives**: Shadows, texture patterns, image artifacts
- **Misclassifications**: Similar crack patterns, ambiguous orientations

## 🛠️ Customization Options

### Test Data Modification
Edit `generate_simple_test_data.py`:
```python
TEST_CONFIG = {
    'vertical': 50,      # Adjust quantities
    'horizontal': 50,
    'diagonal': 50, 
    'step': 50,
    'no_crack': 100,
}
```

### Threshold Tuning
Edit `simple_benchmark.py`:
```python
# YOLO confidence threshold
confidence_threshold = 0.25  # Lower = more detections

# Classification confidence threshold  
classification_threshold = 0.5  # Higher = more conservative
```

### Device Configuration
```bash
# Force CPU usage (for comparison)
python simple_benchmark.py --device cpu

# GPU with specific CUDA device
CUDA_VISIBLE_DEVICES=0 python simple_benchmark.py --device cuda
```

## 📋 Benchmark Report Contents

The benchmark generates:

### 1. Console Output
- Real-time progress bars
- Live metric updates
- Summary statistics
- Error notifications

### 2. JSON Results
- `benchmark_results.json`: Complete metrics and metadata
- Detailed predictions vs ground truth
- Timing and performance statistics
- Configuration parameters

### 3. CSV Export
- `detailed_results.csv`: Per-image results
- Columns: filename, ground_truth, prediction, confidence, timing
- Easy import into analysis tools

### 4. Future Enhancements
- Interactive HTML dashboard
- Precision-Recall curves
- ROC curves and AUC metrics
- Confusion matrix heatmaps

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found**
```bash
❌ Error: Model file not found
✅ Solution: Train models first or check file paths
```

2. **CUDA Out of Memory**
```bash
❌ Error: CUDA out of memory
✅ Solution: Use --device cpu or reduce batch size
```

3. **Import Errors**
```bash
❌ Error: Cannot import pipeline_orchestrator
✅ Solution: Run from correct directory, check Python path
```

4. **No Brick Images**
```bash
❌ Error: No brick images found
✅ Solution: Ensure DATA_bricks directory contains images
```

### Performance Optimization
- **GPU Usage**: Ensure CUDA is available and drivers are updated
- **Memory**: Close other applications to free RAM/VRAM
- **Disk Space**: Ensure sufficient space for test data and results
- **CPU Cores**: Benchmark will use available cores automatically

## 📞 Support

For issues or questions:
1. Check model files are trained and available
2. Verify Python dependencies are installed
3. Ensure sufficient system resources (GPU memory, disk space)
4. Review error messages in benchmark output

## 🎯 Future Roadmap

Planned enhancements:
- **Real Image Testing**: Benchmark on actual construction photos
- **Cross-Validation**: K-fold validation for robust metrics
- **Model Comparison**: A/B testing different model versions
- **Deployment Metrics**: Latency, throughput, resource usage
- **Continuous Integration**: Automated benchmarking on model updates
