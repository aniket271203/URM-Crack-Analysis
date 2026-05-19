# URM Crack Analysis

Automated structural assessment pipeline for **Unreinforced Masonry (URM)** walls. Detects, segments, measures, and diagnoses cracks from photographs using deep learning, brick-calibrated scaling, and FEMA 306-grounded RAG reasoning.

## Pipeline Overview

```
Image → YOLO Detection → DeepCrack Segmentation → Orientation Classification
                              ↓                            ↓
                    Brick-Calibrated Measurement    Crack Location Analysis
                       (length, width, area)       (spatial mapping, PCA)
                              ↓                            ↓
                         ┌─────────────────────────────────┘
                         ↓
                  FEMA 306 RAG Diagnosis
                  (damage type, severity, recommendations)
```

### Stage 1 — Crack Detection (YOLOv8)
Fine-tuned `yolov8n` localises cracks in the image and returns bounding boxes.  
mAP₅₀ = 0.962 on the validation set.

### Stage 2 — Pixel-Level Segmentation (DeepCrack)
A VGG-16-topology encoder with Hierarchical Side-Output fusion produces a probability map of crack pixels. Trained with Binary Focal Loss (γ = 2) for class-imbalance robustness.

### Stage 3 — Orientation Classification
A CNN classifier operating on the binary mask categorises the crack as **Vertical**, **Horizontal**, **Diagonal**, or **Stepped**.

### Stage 4 — Brick-Calibrated Measurement
Detects the brick coursing pattern (mortar lines, FFT, or interactive selection) to compute a **pixel → mm** scale factor. Then:
- **Length** — skeleton extracted via morphological thinning; chain-code walk sums 1.0 for orthogonal and √2 for diagonal steps.
- **Width** — dual method: Euclidean distance transform + perpendicular intensity profiling.
- **Area** — crack pixel count × scale².

### Stage 5 — Spatial Location Analysis
Crack pixels are mapped onto a 5×5 spatial grid to produce a density heatmap. PCA-based endpoint detection and intensity-based origin inference determine propagation direction.

### Stage 6 — FEMA 306 RAG Diagnosis
A two-layer Retrieval-Augmented Generation engine (Gemini LLM + ChromaDB vector store over FEMA 306 documents) classifies the damage type, severity level, and produces an engineering-grade diagnostic report.

## Project Structure

```
├── streamlit_app.py                  # Streamlit web interface
├── pipeline_orchestrator.py          # End-to-end pipeline orchestrator
├── brick_calibrated_measurement.py   # Measurement with brick-scale calibration
├── crack_location_analyzer.py        # Spatial mapping & propagation analysis
│
├── Crack_Detection_YOLO/             # YOLOv8 detection model + weights
├── Masking_and_Classification_model/
│   ├── pretrained_net_G.pth          # DeepCrack segmentation weights
│   ├── crack_orientation_classifier.h5
│   └── model_utils.py               # DeepCrackNet architecture
│
├── Rag_and_Reasoning/                # Original RAG system
├── Independent study/
│   ├── RAG_updated/                  # FEMA 306 RAG agent (ChromaDB + Gemini)
│   └── WALL_Model/                   # Structural component detection (VLM)
│
├── Benchmark/                        # Evaluation scripts & test data
├── Paper/                            # LaTeX source for the research paper
├── Dataset_Generation/               # Synthetic crack generation tools
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
# CPU-only (recommended if no GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu

# Then install the rest:
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then in the browser:
1. Upload an image of a cracked wall
2. (Optional) Select brick type and calibration method
3. Click **Analyze** — results display inline

### 4. CLI usage

```bash
# Brick-calibrated measurement (interactive brick selection)
python brick_calibrated_measurement.py image.jpg --interactive --join-dist 30

# Auto-detect bricks with a specific standard
python brick_calibrated_measurement.py image.jpg --brick-type india_modular
```

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `--brick-type` | `india_modular` | Brick standard (`india_modular`, `uk_standard`, `us_standard`, `europe_nf`) |
| `--join-dist` | `30` | Max pixel gap to bridge disconnected crack fragments |
| `--threshold` | `40` | Segmentation binarisation threshold (0–255) |
| `--interactive` | off | Manually click brick top/bottom edges for calibration |
| `--device` | `cuda` | `cuda` or `cpu` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` | `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `No module named 'tensorflow'` | `pip uninstall tensorflow -y && pip install tensorflow` (or `tensorflow-cpu`) |
| RAG agent fails to load | Ensure `GEMINI_API_KEY` is set in `.env` |
| Brick auto-detection fails | Use `--interactive` flag to manually select a brick |

## Acknowledgments

- [DeepCrack](https://github.com/yhlleo/DeepCrack) — Hierarchical feature learning for crack segmentation
- [YOLOv8](https://github.com/ultralytics/ultralytics) — Real-time object detection
- [FEMA 306](https://www.fema.gov/) — Evaluation of earthquake-damaged concrete and masonry wall buildings
- [Google Gemini](https://ai.google.dev/) — Large language model for RAG reasoning
- [Streamlit](https://streamlit.io/) — Web application framework
