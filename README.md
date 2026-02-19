# Structural Crack Analysis System

A comprehensive pipeline for structural crack detection, segmentation, classification, and analysis using deep learning and RAG (Retrieval-Augmented Generation).

## 🎯 Overview

This system integrates three independent models into a unified pipeline:

1. **Crack Detection (YOLO)**: Detects cracks in images with bounding boxes
2. **Segmentation & Classification**: Segments crack regions and classifies them into 4 categories (vertical, horizontal, diagonal, step)
3. **RAG Analysis**: Uses Gemini AI with document retrieval to determine crack causes and provide recommendations

## 📁 Project Structure

```
Final_Combined_Model/
├── Crack_Detection_YOLO/
│   └── crack_yolo_train/
│       ├── weights/
│       │   └── best.pt              # YOLO detection model
│       └── crack_detection_inference.py
├── Masking_and_Classification_model/
│   ├── pretrained_net_G.pth          # Segmentation model
│   ├── crack_orientation_classifier.h5 # Classification model
│   └── model_utils.py                 # Model architecture utilities
├── Rag_and_Reasoning/
│   └── crack_analysis_rag/
│       ├── src/                       # RAG system source code
│       └── data/                      # RAG knowledge base
├── pipeline_orchestrator.py           # Main pipeline orchestrator
├── streamlit_app.py                   # Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Installation

### 1. Clone the repository

```bash
cd /path/to/Final_Combined_Model
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have a GPU and want to use FAISS with GPU support:
```bash
pip install faiss-gpu  # Instead of faiss-cpu
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
USE_GPU=true  # or false for CPU-only
```

To get a Gemini API key:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🎮 Usage

### Option 1: Streamlit Web App (Recommended)

1. **Start the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

2. **In the web interface**:
   - Configure model paths in the sidebar (defaults should work)
   - Click "🔄 Load Pipeline" to initialize all models
   - Upload an image using the file uploader
   - Click "🔍 Analyze Image" to run the complete pipeline
   - View results in the main panel

### Option 2: Python Script

```python
from pipeline_orchestrator import CrackAnalysisPipeline

# Initialize pipeline
pipeline = CrackAnalysisPipeline(
    yolo_model_path="Crack_Detection_YOLO/crack_yolo_train/weights/best.pt",
    segmentation_model_path="Masking_and_Classification_model/pretrained_net_G.pth",
    classification_model_path="Masking_and_Classification_model/crack_orientation_classifier.h5",
    rag_data_dir="Rag_and_Reasoning/crack_analysis_rag/data",
    device='cuda'  # or 'cpu'
)

# Process an image
results = pipeline.process_image(
    image_path="path/to/your/image.jpg",
    use_rag=True,
    save_intermediate=True,
    output_dir="output"
)

# Get summary
summary = pipeline.get_summary(results)
print(summary)
```

## 🔄 Pipeline Flow

The complete pipeline follows these steps:

1. **Crack Detection (YOLO)**
   - Input: Original image
   - Output: Bounding boxes around detected cracks
   - If no cracks detected, pipeline stops

2. **Image Segmentation**
   - Input: Original image
   - Output: Binary mask highlighting crack regions
   - Uses DeepCrack segmentation model

3. **Crack Classification**
   - Input: Segmented mask
   - Output: Crack type (vertical/horizontal/diagonal/step) with confidence
   - Uses CNN classifier

4. **RAG Analysis** (Optional)
   - Input: Segmented mask + crack type
   - Output: Comprehensive analysis including:
     - Location analysis
     - Root cause determination
     - Safety assessment
     - Recommendations

## 📊 Output Format

The pipeline returns a dictionary with the following structure:

```python
{
    'success': bool,
    'image_path': str,
    'detection': {
        'detections': list,  # List of bounding boxes
        'num_cracks': int,
        'crack_detected': bool
    },
    'segmentation': {
        'mask': np.ndarray,  # Segmented mask
        'mask_shape': tuple
    },
    'classification': {
        'crack_type': str,  # 'vertical', 'horizontal', 'diagonal', 'step'
        'confidence': float
    },
    'rag_analysis': {
        'success': bool,
        'summary': {
            'location_analysis': str,
            'cause_analysis': str,
            'comprehensive_report': str
        }
    }
}
```

## ⚙️ Configuration

### Model Paths

Default paths (can be changed in Streamlit sidebar or code):
- YOLO: `Crack_Detection_YOLO/crack_yolo_train/weights/best.pt`
- Segmentation: `Masking_and_Classification_model/pretrained_net_G.pth`
- Classification: `Masking_and_Classification_model/crack_orientation_classifier.h5`
- RAG Data: `Rag_and_Reasoning/crack_analysis_rag/data`

### Device Selection

- **GPU (CUDA)**: Faster processing, requires CUDA-compatible GPU
- **CPU**: Slower but works on any machine

Set via environment variable or in code:
```python
device = 'cuda'  # or 'cpu'
```

## 🐛 Troubleshooting

### Common Issues

1. **Model loading errors**
   - Ensure all model files exist at specified paths
   - Check file permissions

2. **CUDA/GPU errors**
   - Install appropriate PyTorch version for your CUDA version
   - Set `USE_GPU=false` in `.env` to use CPU

3. **RAG system errors**
   - Verify `GOOGLE_API_KEY` is set in `.env`
   - Check internet connection (for Gemini API)
   - Ensure RAG data directory exists and contains documents

4. **Memory errors**
   - Reduce batch size in model inference
   - Use CPU instead of GPU
   - Process smaller images

### Dependencies Issues

If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Notes

- The first run may take longer as models are loaded into memory
- RAG analysis requires internet connection for Gemini API
- All models are loaded into memory at startup for faster inference
- Intermediate results can be saved for debugging

## 🔧 Development

### Adding New Models

To integrate additional models:
1. Create a wrapper class similar to `SegmentationModel` or `ClassificationModel`
2. Add it to `CrackAnalysisPipeline.__init__()`
3. Integrate into `process_image()` method

### Customizing RAG

- Add documents to `Rag_and_Reasoning/crack_analysis_rag/data/`
- Modify prompts in `gemini_analyzer.py`
- Adjust retrieval parameters in `enhanced_retrieval.py`

## 📄 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepCrack segmentation model
- Google Gemini API
- Streamlit framework




