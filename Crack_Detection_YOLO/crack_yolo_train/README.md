# Crack Detection Inference

This directory contains scripts to run inference using your trained YOLOv8 crack detection model.

## Files

- `crack_detection_inference.py` - Main inference script with full functionality
- `test_crack_detection.py` - Simple test script for quick testing
- `weights/best.pt` - Your trained YOLOv8 model weights
- `weights/last.pt` - Last epoch model weights

## Quick Start

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install ultralytics opencv-python matplotlib pillow torch torchvision
```

### 2. Quick Test

Run a quick test with validation images:

```bash
python test_crack_detection.py
```

### 3. Interactive Test

Test with your own images interactively:

```bash
python test_crack_detection.py --interactive
```

## Main Inference Script Usage

### Single Image

```bash
python crack_detection_inference.py --input /path/to/your/image.jpg
```

### Directory of Images

```bash
python crack_detection_inference.py --input /path/to/image/directory/
```

### Custom Parameters

```bash
python crack_detection_inference.py \
    --input /path/to/image.jpg \
    --model weights/best.pt \
    --conf 0.5 \
    --iou 0.45 \
    --output my_results/
```

## Parameters

- `--model`: Path to model weights (default: `weights/best.pt`)
- `--input`: Input image or directory (required)
- `--output`: Output directory for results (default: `inference_results`)
- `--conf`: Confidence threshold 0-1 (default: 0.25)
- `--iou`: IoU threshold for NMS 0-1 (default: 0.45)
- `--no-save`: Don't save annotated images (flag)

## Output

The script will:

1. **Process images** and detect cracks
2. **Save annotated images** with bounding boxes in the output directory
3. **Print detection summary** with statistics
4. **Show confidence scores** for each detection

### Example Output

```
CRACK DETECTION SUMMARY
==================================================
Total images processed: 5
Images with cracks detected: 3
Images without cracks: 2
Total cracks detected: 7
Average cracks per image: 1.40
Average cracks per positive image: 2.33

Detailed Results:
------------------------------
image1.jpg: 2 crack(s)
image2.jpg: 0 crack(s)
image3.jpg: 3 crack(s)
image4.jpg: 0 crack(s)
image5.jpg: 2 crack(s)
```

## Model Performance

Based on your training results (`results.csv`), your model achieved:
- **mAP50**: 96.26% (very good detection accuracy)
- **mAP50-95**: 71.54% (good across different IoU thresholds)
- **Precision**: 93.37% (low false positives)
- **Recall**: 93.10% (good at finding actual cracks)

## Tips for Best Results

1. **Adjust confidence threshold**: Lower values (0.1-0.3) for more sensitive detection, higher values (0.5-0.8) for more confident detections only

2. **Image quality**: Best results with clear, well-lit images similar to your training data

3. **Image size**: The model was trained on 640x640 images, so similar aspect ratios work best

4. **Batch processing**: Use directory mode for processing multiple images efficiently

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Make sure `weights/best.pt` exists in the current directory
   - Check the model path in your command

2. **"CUDA out of memory"**
   - The script will automatically fall back to CPU if needed
   - For large images, consider resizing them first

3. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

4. **Poor detection results**
   - Try adjusting the confidence threshold (`--conf`)
   - Ensure your test images are similar to training data
   - Check if the image quality is good

### Performance Notes

- **GPU**: Will automatically use CUDA if available for faster inference
- **CPU**: Falls back to CPU if CUDA is not available
- **Memory**: Large images may require more memory

## Example Commands

```bash
# Basic usage
python crack_detection_inference.py --input test_image.jpg

# High confidence detections only
python crack_detection_inference.py --input test_image.jpg --conf 0.7

# Process entire directory
python crack_detection_inference.py --input /path/to/test/images/ --output results/

# Don't save images, just get detection info
python crack_detection_inference.py --input test_image.jpg --no-save

# Use last epoch weights instead of best
python crack_detection_inference.py --input test_image.jpg --model weights/last.pt
```
