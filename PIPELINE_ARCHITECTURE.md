# Pipeline Architecture Documentation

## 🏗️ System Overview

The Crack Analysis Pipeline integrates three independent models into a unified workflow:

```
Input Image
    ↓
[1] YOLO Detection → Detects cracks with bounding boxes
    ↓ (if cracks detected)
[2] Segmentation → Extracts crack mask
    ↓
[3] Classification → Classifies crack type (4 categories)
    ↓
[4] RAG Analysis → Determines cause and provides recommendations
    ↓
Final Report
```

## 📦 Component Details

### 1. YOLO Crack Detection

**Location**: `Crack_Detection_YOLO/crack_yolo_train/`

**Model**: YOLOv8 (Ultralytics)

**Input**: RGB image (any size)

**Output**: 
- Bounding boxes for detected cracks
- Confidence scores
- Class labels

**Key Files**:
- `crack_detection_inference.py`: Inference script
- `weights/best.pt`: Trained model weights

**Integration**: 
- Uses `ultralytics.YOLO` for inference
- Returns detection results with bounding boxes
- Pipeline stops if no cracks detected

### 2. Segmentation Model

**Location**: `Masking_and_Classification_model/`

**Model**: DeepCrack (Deep Hierarchical Feature Learning)

**Architecture**: 
- Multi-scale feature extraction
- Side outputs at different scales
- Fused output for final mask

**Input**: RGB image (normalized)

**Output**: Binary mask (grayscale, 0-255)

**Key Files**:
- `pretrained_net_G.pth`: Trained model weights
- `model_utils.py`: Model architecture definition

**Integration**:
- Wrapped in `SegmentationModel` class
- Handles GPU/CPU automatically
- Supports FP16 for faster inference

### 3. Classification Model

**Location**: `Masking_and_Classification_model/`

**Model**: CNN Classifier (TensorFlow/Keras)

**Architecture**:
- 3 Conv2D layers with MaxPooling
- Flatten + Dense layers
- Softmax output (5 classes, but uses 4)

**Input**: Segmented mask (256x256, grayscale)

**Output**: 
- Crack type: `vertical`, `horizontal`, `diagonal`, `step`
- Confidence score (0-1)

**Key Files**:
- `crack_orientation_classifier.h5`: Trained model weights

**Integration**:
- Wrapped in `ClassificationModel` class
- Preprocesses mask to 256x256
- Returns type and confidence

### 4. RAG Analysis System

**Location**: `Rag_and_Reasoning/crack_analysis_rag/`

**Components**:
- **Document Manager**: Manages knowledge base
- **Enhanced Retriever**: Hybrid search (semantic + keyword)
- **Gemini Analyzer**: Multi-step reasoning with Gemini AI

**Input**: 
- Segmented mask image
- Crack type (from classification)

**Output**:
- Location analysis
- Root cause determination
- Safety assessment
- Comprehensive report

**Key Files**:
- `src/main_rag.py`: Main RAG orchestrator
- `src/gemini_analyzer.py`: Gemini API integration
- `src/enhanced_retrieval.py`: Document retrieval
- `data/`: Knowledge base documents

**Integration**:
- Optional component (can be disabled)
- Requires Gemini API key
- Uses retrieved documents for context

## 🔄 Pipeline Flow

### Step-by-Step Execution

1. **Image Input**
   ```python
   image_path → PIL Image → Validation
   ```

2. **YOLO Detection**
   ```python
   image → YOLO model → detections[]
   if detections.empty():
       return "No cracks detected"
   ```

3. **Segmentation**
   ```python
   image → DeepCrack → mask (numpy array)
   mask → Save to temp file (for RAG)
   ```

4. **Classification**
   ```python
   mask → Resize(256x256) → CNN → crack_type, confidence
   ```

5. **RAG Analysis** (if enabled)
   ```python
   mask + crack_type → RAG system:
       → Document retrieval (hybrid search)
       → Location analysis (Gemini)
       → Cause determination (Gemini + context)
       → Report generation (Gemini)
   ```

6. **Result Compilation**
   ```python
   results = {
       'detection': {...},
       'segmentation': {...},
       'classification': {...},
       'rag_analysis': {...}
   }
   ```

## 🧩 Class Structure

### `CrackAnalysisPipeline`

Main orchestrator class that coordinates all models.

**Methods**:
- `__init__()`: Initialize all models
- `process_image()`: Run complete pipeline
- `get_summary()`: Generate human-readable summary

### `SegmentationModel`

Wrapper for DeepCrack segmentation model.

**Methods**:
- `__init__()`: Load model and weights
- `segment()`: Process single image

### `ClassificationModel`

Wrapper for CNN classifier.

**Methods**:
- `__init__()`: Load model
- `classify()`: Classify mask image

## 🔌 Integration Points

### Model Loading

All models are loaded at pipeline initialization:
- YOLO: Direct from `.pt` file
- Segmentation: Architecture + weights from `.pth`
- Classification: Keras model from `.h5`
- RAG: Document manager + Gemini client

### Data Flow

```
Image (file path)
    ↓
YOLO → detections (list of bboxes)
    ↓
Original Image → Segmentation → mask (numpy)
    ↓
Mask → Classification → type, confidence
    ↓
Mask + Type → RAG → analysis report
```

### Error Handling

- Each step has try-except blocks
- Errors are captured in results dictionary
- Pipeline continues if optional steps fail
- Detailed error messages in output

## 🎯 Key Design Decisions

1. **Modular Architecture**: Each model is independent and can be used separately

2. **Lazy Loading**: Models loaded only when pipeline is initialized

3. **Optional RAG**: RAG analysis can be disabled for faster processing

4. **Memory Management**: 
   - GPU memory cleared after each step
   - Temporary files cleaned up
   - Support for CPU fallback

5. **Flexible Input**: 
   - Accepts file paths
   - Handles various image formats
   - Automatic preprocessing

6. **Comprehensive Output**: 
   - Structured results dictionary
   - Human-readable summaries
   - Exportable reports

## 🔧 Configuration

### Model Paths
Set via:
- Streamlit sidebar (UI)
- Command-line arguments (CLI)
- Code initialization (API)

### Device Selection
- Automatic: Uses GPU if available
- Manual: Set via `device` parameter
- Environment: `USE_GPU` in `.env`

### RAG Configuration
- Enable/disable via `use_rag` parameter
- API key via `.env` file
- Data directory configurable

## 📊 Performance Considerations

### Speed
- YOLO: ~100-200ms per image
- Segmentation: ~500-1000ms per image
- Classification: ~50-100ms per image
- RAG: ~5-10 seconds (API call)

### Memory
- YOLO: ~500MB GPU
- Segmentation: ~1-2GB GPU
- Classification: ~200MB RAM
- RAG: ~500MB RAM (embeddings)

### Optimization
- FP16 for segmentation (if GPU supports)
- Batch processing (future enhancement)
- Model quantization (future enhancement)

## 🚀 Future Enhancements

1. **Batch Processing**: Process multiple images
2. **Real-time Video**: Process video streams
3. **Model Optimization**: Quantization, pruning
4. **Additional Classifiers**: More crack types
5. **Enhanced RAG**: More document sources
6. **API Server**: REST API for integration
7. **Database**: Store analysis history
8. **Dashboard**: Analytics and trends

## 📝 Notes

- All models are inference-only (no training)
- Models can be swapped independently
- Pipeline is stateless (except model loading)
- Thread-safe for concurrent requests (with proper setup)





