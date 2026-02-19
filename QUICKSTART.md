# Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser automatically!

## 📋 Step-by-Step Usage

### In the Streamlit App:

1. **Load Pipeline** (Sidebar)
   - Model paths are pre-configured
   - Click "🔄 Load Pipeline"
   - Wait for "✅ Pipeline Ready"

2. **Upload Image**
   - Use the file uploader
   - Supported: JPG, PNG, JPEG, BMP

3. **Analyze**
   - Click "🔍 Analyze Image"
   - Wait for processing (30-60 seconds)

4. **View Results**
   - Detection: Number of cracks found
   - Segmentation: Mask visualization
   - Classification: Crack type and confidence
   - RAG Analysis: Comprehensive report

## 🖥️ Command Line Usage

```bash
python run_pipeline.py --image path/to/image.jpg
```

With custom paths:
```bash
python run_pipeline.py \
  --image test.jpg \
  --yolo Crack_Detection_YOLO/crack_yolo_train/weights/best.pt \
  --seg Masking_and_Classification_model/pretrained_net_G.pth \
  --class Masking_and_Classification_model/crack_orientation_classifier.h5 \
  --rag-dir Rag_and_Reasoning/crack_analysis_rag/data
```

## ⚡ Troubleshooting

### "Pipeline not loaded"
- Check model file paths exist
- Ensure all dependencies installed
- Check console for error messages

### "RAG analysis failed"
- Verify `GOOGLE_API_KEY` in `.env`
- Check internet connection
- Ensure RAG data directory exists

### "CUDA out of memory"
- Set `USE_GPU=false` in `.env`
- Or reduce image size before uploading

### Models not found
- Verify file paths in sidebar
- Check file permissions
- Ensure models are in correct directories

## 📊 Expected Output

When successful, you'll see:

1. ✅ **Detection**: "X crack(s) detected"
2. ✅ **Segmentation**: Mask image displayed
3. ✅ **Classification**: "Classified as: [TYPE] (confidence: XX%)"
4. ✅ **RAG Analysis**: Comprehensive report with:
   - Location analysis
   - Root cause
   - Safety assessment
   - Recommendations

## 💡 Tips

- First run takes longer (model loading)
- Use GPU for faster processing
- Save intermediate results for debugging
- Export reports for documentation

## 🆘 Need Help?

Check the full [README.md](README.md) for detailed documentation.




