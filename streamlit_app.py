"""
Streamlit App for Complete Crack Analysis Pipeline
Integrates YOLO Detection, Segmentation, Classification, 
Crack Measurement, Location Analysis, and RAG Analysis
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import time
import json
import traceback
from datetime import datetime
from io import BytesIO

# PDF Generation
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor, black, gray, red, green, blue
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("ReportLab not available for PDF generation")

# Try to import streamlit-image-coordinates for interactive click mode
IMAGE_COORDINATES_AVAILABLE = False
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    IMAGE_COORDINATES_AVAILABLE = True
except ImportError:
    IMAGE_COORDINATES_AVAILABLE = False

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Independent study', 'RAG_updated'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Independent study', 'WALL_Model'))
from pipeline_orchestrator import CrackAnalysisPipeline

# Import crack measurement and location analysis modules
from brick_calibrated_measurement import (
    BrickDetector, 
    CrackMeasurerWithBrickCalibration,
    BRICK_STANDARDS,
    visualize_skeleton
)
from crack_location_analyzer import CrackLocationAnalyzer, CrackLocationPipeline

# Import WALL_Model for structural component detection
WALL_MODEL_AVAILABLE = False
try:
    from src.vision.component_classifier import WallClassifier, WallComponent
    WALL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"WALL_Model not available: {e}")
    WALL_MODEL_AVAILABLE = False

# Import FEMA 306 RAG components
FEMA_RAG_AVAILABLE = False
try:
    from src.retriever import SectionRetriever
    from src.llm_service import GeminiLLMService
    from src.structural_agent import StructuralAgent
    from src.schema import DiagnosisResult
    FEMA_RAG_AVAILABLE = True
except ImportError as e:
    print(f"FEMA RAG not available: {e}")
    FEMA_RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Crack Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# STANDARD IMAGE SIZE - All uploaded images are resized to this width
# to ensure consistent behavior across small and large images.
# Aspect ratio is always preserved.
# =========================================================================
STANDARD_IMAGE_WIDTH = 1024  # pixels


def resize_to_standard(pil_image: Image.Image, target_width: int = STANDARD_IMAGE_WIDTH) -> Image.Image:
    """
    Resize a PIL image to a standard width while preserving aspect ratio.
    Both upsizes and downsizes to ensure all images are at the same scale.
    
    Args:
        pil_image: Input PIL Image.
        target_width: Desired width in pixels.
    
    Returns:
        Resized PIL Image (or original if already at target width).
    """
    w, h = pil_image.size
    if w == target_width:
        return pil_image
    scale = target_width / w
    new_h = int(h * scale)
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return pil_image.resize((target_width, new_h), resample)


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_image_path' not in st.session_state:
    st.session_state.uploaded_image_path = None
# Brick calibration state
if 'brick_height_px' not in st.session_state:
    st.session_state.brick_height_px = None
if 'scale_mm_per_px' not in st.session_state:
    st.session_state.scale_mm_per_px = None
if 'brick_calibration_done' not in st.session_state:
    st.session_state.brick_calibration_done = False
if 'calibration_mode' not in st.session_state:
    st.session_state.calibration_mode = "Automatic"
# Interactive calibration click points
if 'click_points' not in st.session_state:
    st.session_state.click_points = []
# Measurement and location results
if 'measurement_results' not in st.session_state:
    st.session_state.measurement_results = None
if 'location_results' not in st.session_state:
    st.session_state.location_results = None
# FEMA RAG state
if 'fema_rag_agent' not in st.session_state:
    st.session_state.fema_rag_agent = None
if 'fema_diagnosis' not in st.session_state:
    st.session_state.fema_diagnosis = None
# WALL_Model state for structural component detection
if 'wall_components' not in st.session_state:
    st.session_state.wall_components = None
if 'wall_classifier' not in st.session_state:
    st.session_state.wall_classifier = None
# PDF report state
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None


def reset_image_state():
    """Reset all image-specific results in session state."""
    st.session_state.results = None
    st.session_state.uploaded_image = None
    st.session_state.uploaded_image_path = None
    st.session_state.brick_height_px = None
    st.session_state.scale_mm_per_px = None
    st.session_state.brick_calibration_done = False
    st.session_state.click_points = []
    st.session_state.measurement_results = None
    st.session_state.location_results = None
    st.session_state.fema_diagnosis = None
    st.session_state.wall_components = None
    st.session_state.pdf_report = None


def detect_structural_component(image: np.ndarray, crack_centroid: tuple = None) -> dict:
    """
    Use WALL_Model to detect which structural component contains the crack.
    
    Args:
        image: Original wall image (BGR)
        crack_centroid: Normalized (x, y) centroid of crack (0-1 range)
    
    Returns:
        Dict with component_type, component_id, confidence
    """
    if not WALL_MODEL_AVAILABLE:
        return {"component_type": "Unknown", "component_id": None, "confidence": 0.0}
    
    try:
        # Initialize classifier if needed
        if st.session_state.wall_classifier is None:
            st.session_state.wall_classifier = WallClassifier()  # Use heuristic for speed
        
        classifier = st.session_state.wall_classifier
        
        # Detect openings
        detections = classifier.detect_openings(image)
        
        # Infer structural components
        components = classifier.infer_structure(image.shape, detections)
        st.session_state.wall_components = components
        
        if not components or crack_centroid is None:
            print("No components detected or no crack centroid provided.")
            return {"component_type": "Wall Panel", "component_id": None, "confidence": 0.5}
        
        # Convert normalized centroid to pixel coordinates
        h, w = image.shape[:2]
        crack_x = int(crack_centroid[0] * w)
        crack_y = int(crack_centroid[1] * h)
        
        # Find which component contains the crack
        for comp in components:
            x, y, cw, ch = comp.bbox
            if x <= crack_x <= x + cw and y <= crack_y <= y + ch:
                return {
                    "component_type": comp.type,
                    "component_id": comp.id,
                    "confidence": 0.8,
                    "description": comp.description
                }
        
        # If not inside any component, determine relative position
        # Check if near any pier or spandrel
        piers = [c for c in components if c.type == "Pier"]
        spandrels = [c for c in components if c.type == "Spandrel"]
        
        if piers:
            # Check vertical position relative to piers
            pier = piers[0]
            px, py, pw, ph = pier.bbox
            if crack_y < py + ph * 0.3:
                return {"component_type": "Top of Pier", "component_id": pier.id, "confidence": 0.6}
            elif crack_y > py + ph * 0.7:
                return {"component_type": "Bottom of Pier", "component_id": pier.id, "confidence": 0.6}
            else:
                return {"component_type": "Mid-height of Pier", "component_id": pier.id, "confidence": 0.6}
        
        return {"component_type": "Wall Panel", "component_id": None, "confidence": 0.5}
        
    except Exception as e:
        print(f"WALL_Model detection error: {e}")
        return {"component_type": "Unknown", "component_id": None, "confidence": 0.0, "error": str(e)}


def map_location_to_fema(grid_location: str, orientation: str = None, 
                         structural_component: dict = None) -> str:
    """
    Map grid-based location to FEMA 306 structural terms.
    Enhanced with WALL_Model structural component detection.
    
    Args:
        grid_location: Location from crack_location_analyzer (e.g., 'upper-left', 'middle-center')
        orientation: Crack orientation ('horizontal', 'vertical', 'diagonal')
        structural_component: Optional dict from WALL_Model with component_type
    
    Returns:
        FEMA 306 structural location term
    """
    # If WALL_Model detected a specific structural component, use it
    if structural_component and structural_component.get('component_type'):
        comp_type = structural_component['component_type']
        if comp_type in ['Pier', 'Spandrel', 'Opening', 'Top of Pier', 'Bottom of Pier', 'Mid-height of Pier']:
            return comp_type
    
    # Fallback to grid-based location mapping
    location_lower = grid_location.lower() if grid_location else ""
    
    # Vertical position mappings
    if 'top' in location_lower:
        if 'left' in location_lower or 'right' in location_lower:
            return "Top of Pier"
        return "Top of Wall"
    elif 'bottom' in location_lower:
        if 'left' in location_lower or 'right' in location_lower:
            return "Bottom of Pier"
        return "Bottom of Wall"
    elif 'center' in location_lower or 'middle' in location_lower:
        if orientation and orientation.lower() == 'diagonal':
            return "Center of Pier"
        return "Mid-height of Pier"
    elif 'left' in location_lower or 'right' in location_lower:
        return "Spandrel Ends"
    else:
        return "Wall Panel"


def build_fema_input(classification_result: dict, measurement_result: dict, 
                     location_result, structural_component: dict = None,
                     original_image: np.ndarray = None) -> dict:
    """
    Build input dictionary for FEMA 306 StructuralAgent from all calculated metrics.
    
    Data Sources (Priority Order):
    1. ORIENTATION: Classification model (crack_type) → Location analyzer (fallback)
    2. WIDTH/LENGTH: Brick-calibrated measurement 
    3. LOCATION: WALL_Model (structural component) → Location analyzer grid (fallback)
    
    Args:
        classification_result: Classification output with crack_type from classifier model
        measurement_result: Measurement output with width_stats from brick_calibrated_measurement
        location_result: LocationAnalysisResult object from crack_location_analyzer
        structural_component: Optional dict from WALL_Model with component_type
        original_image: Original image for WALL_Model detection if needed
    
    Returns:
        Dictionary formatted for FEMA 306 diagnosis with all extracted metrics
    """
    # =========================================================================
    # 1. ORIENTATION - Priority: Classification Model > Location Analyzer
    # =========================================================================
    orientation = "Unknown"
    orientation_source = "none"
    orientation_confidence = 0.0
    
    # First try: Classification model result (primary source)
    if classification_result and 'crack_type' in classification_result:
        crack_type = classification_result['crack_type'].lower()
        orientation_confidence = classification_result.get('confidence', 0.0)
        
        if crack_type == 'horizontal':
            orientation = "Horizontal"
            orientation_source = "classification_model"
        elif crack_type == 'vertical':
            orientation = "Vertical"
            orientation_source = "classification_model"
        elif crack_type == 'diagonal':
            orientation = "Diagonal"
            orientation_source = "classification_model"
        elif crack_type == 'step':
            orientation = "Diagonal"  # Step cracks are diagonal along mortar joints
            orientation_source = "classification_model (step→diagonal)"
    
    # Fallback: Location analyzer orientation
    if orientation == "Unknown" and location_result:
        if hasattr(location_result, 'orientation') and location_result.orientation:
            orientation = location_result.orientation.title()
            orientation_source = "location_analyzer"
            orientation_confidence = getattr(location_result, 'confidence', 0.5)
    
    # =========================================================================
    # 2. WIDTH & LENGTH - From Brick-Calibrated Measurement
    # =========================================================================
    width_str = "Unknown"
    max_width_mm = 0.0
    mean_width_mm = 0.0
    length_mm = 0.0
    
    if measurement_result:
        width_stats = measurement_result.get('width_stats', {})
        max_width_mm = width_stats.get('max_mm', 0)
        mean_width_mm = width_stats.get('mean_mm', 0)
        length_mm = measurement_result.get('length_mm', 0)
        
        if max_width_mm > 0:
            width_str = f"{max_width_mm:.1f}mm"
    
    # =========================================================================
    # 3. LOCATION - Priority: WALL_Model > Location Analyzer Grid
    # =========================================================================
    fema_location = "Unknown"
    location_source = "none"
    
    # Try WALL_Model structural component detection first
    if structural_component and structural_component.get('component_type'):
        comp_type = structural_component['component_type']
        if comp_type not in ['Unknown', None]:
            fema_location = comp_type
            location_source = "WALL_Model"
    
    # Fallback: Location analyzer with grid mapping
    if fema_location == "Unknown" and location_result:
        if hasattr(location_result, 'dominant_location') and location_result.dominant_location:
            fema_location = map_location_to_fema(
                location_result.dominant_location, 
                orientation,
                structural_component
            )
            location_source = "location_analyzer_grid"
    
    # =========================================================================
    # 4. BUILD DESCRIPTION - Comprehensive crack description
    # =========================================================================
    description_parts = []
    
    # Add crack type/pattern info from classification
    crack_type_raw = classification_result.get('crack_type', '') if classification_result else ''
    if crack_type_raw:
        if crack_type_raw.lower() == 'step':
            description_parts.append("Step-pattern cracks along mortar joints")
        elif crack_type_raw.lower() == 'diagonal':
            description_parts.append("Diagonal cracking pattern through wall")
        elif crack_type_raw.lower() == 'horizontal':
            description_parts.append("Horizontal crack pattern in bed joints")
        elif crack_type_raw.lower() == 'vertical':
            description_parts.append("Vertical crack pattern through head joints")
    
    # Add dimensional info
    if max_width_mm > 0:
        if max_width_mm < 0.5:
            description_parts.append("hairline width crack")
        elif max_width_mm < 2.0:
            description_parts.append("fine crack width")
        elif max_width_mm < 5.0:
            description_parts.append("moderate crack width")
        else:
            description_parts.append("wide crack opening")
    
    # Add propagation info from location analyzer
    if location_result:
        if hasattr(location_result, 'propagation_direction') and location_result.propagation_direction:
            description_parts.append(f"propagating {location_result.propagation_direction}")
        
        if hasattr(location_result, 'spread_type') and location_result.spread_type:
            spread = location_result.spread_type.replace('_', ' ')
            description_parts.append(f"{spread} spread pattern")
    
    # Combine description
    description = ". ".join(description_parts) if description_parts else "Crack observed in masonry wall"
    description = description[0].upper() + description[1:]  # Capitalize first letter
    
    # =========================================================================
    # 5. BUILD FINAL INPUT DICT
    # =========================================================================
    fema_input = {
        "material": "URM",  # Unreinforced Masonry - default for brick walls
        "orientation": orientation,
        "width": width_str,
        "location": fema_location,
        "description": description
    }
    
    # Also include metadata for display/debugging
    fema_input["_metadata"] = {
        "orientation_source": orientation_source,
        "orientation_confidence": orientation_confidence,
        "location_source": location_source,
        "max_width_mm": max_width_mm,
        "mean_width_mm": mean_width_mm,
        "length_mm": length_mm,
        "crack_type_raw": crack_type_raw
    }
    
    return fema_input


def generate_analysis_pdf(
    image_path: str,
    detection_results: dict,
    classification_results: dict,
    measurement_results: dict,
    location_results,
    fema_input: dict,
    fema_diagnosis,
    output_path: str = None
) -> bytes:
    """
    Generate a comprehensive PDF report of the crack analysis.
    
    Args:
        image_path: Path to the analyzed image
        detection_results: YOLO detection output
        classification_results: Crack classification output
        measurement_results: Brick-calibrated measurement output
        location_results: Crack location analysis output
        fema_input: Input used for FEMA diagnosis
        fema_diagnosis: DiagnosisResult from FEMA 306 RAG
        output_path: Optional path to save PDF
    
    Returns:
        PDF as bytes
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("ReportLab is not installed. Install with: pip install reportlab")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1*cm,
        leftMargin=1*cm,
        topMargin=1*cm,
        bottomMargin=1*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1f77b4'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#2c3e50'),
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=5
    )
    
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#34495e'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    citation_style = ParagraphStyle(
        'Citation',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor('#555555'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        borderColor=HexColor('#cccccc'),
        borderWidth=1,
        borderPadding=8,
        backColor=HexColor('#f9f9f9')
    )
    
    elements = []
    
    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("🔍 STRUCTURAL CRACK ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("FEMA 306 Compliant Assessment", ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER, textColor=gray
    )))
    elements.append(Spacer(1, 30))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %H:%M")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
    elements.append(Spacer(1, 10))
    
    # Add analyzed image if available
    if image_path and os.path.exists(image_path):
        try:
            img = RLImage(image_path, width=12*cm, height=9*cm)
            elements.append(Spacer(1, 20))
            elements.append(img)
            elements.append(Paragraph("<i>Analyzed Wall Image</i>", ParagraphStyle(
                'Caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=gray
            )))
        except Exception as e:
            pass
    
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=2, color=HexColor('#3498db')))
    elements.append(PageBreak())
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    elements.append(Paragraph("📋 EXECUTIVE SUMMARY", header_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))
    elements.append(Spacer(1, 10))
    
    # Severity assessment
    if fema_diagnosis:
        severity = fema_diagnosis.severity if fema_diagnosis.severity else "Unknown"
        damage_level = fema_diagnosis.damage_level if fema_diagnosis.damage_level else "Unknown"
        failure_mode = fema_diagnosis.failure_mode.name if fema_diagnosis.failure_mode else "Not Determined"
        hybrid_score = fema_diagnosis.hybrid_score if fema_diagnosis.hybrid_score else 0.0
        
        # Severity color coding
        severity_color = "#27ae60" if severity.lower() in ['minor', 'low'] else \
                        "#f39c12" if severity.lower() in ['moderate', 'medium'] else \
                        "#e74c3c" if severity.lower() in ['severe', 'high', 'critical'] else "#7f8c8d"
        
        summary_data = [
            ["Identified Failure Mode", failure_mode],
            ["Damage Level", damage_level],
            ["Severity Assessment", severity],
            ["Confidence Score", f"{hybrid_score:.1%}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[6*cm, 10*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#ecf0f1')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (1, 2), (1, 2), HexColor(severity_color)),  # Severity row
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
    else:
        elements.append(Paragraph("<i>FEMA 306 Diagnosis was not performed.</i>", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # =========================================================================
    # CRACK MEASUREMENTS
    # =========================================================================
    elements.append(Paragraph("📏 CRACK MEASUREMENTS", header_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))
    elements.append(Spacer(1, 10))
    
    if measurement_results:
        width_stats = measurement_results.get('width_stats', {})
        measurement_data = [
            ["Parameter", "Value", "Unit"],
            ["Crack Length", f"{measurement_results.get('length_mm', 0):.2f}", "mm"],
            ["Maximum Width", f"{width_stats.get('max_mm', 0):.2f}", "mm"],
            ["Average Width", f"{width_stats.get('mean_mm', 0):.2f}", "mm"],
            ["Minimum Width", f"{width_stats.get('min_mm', 0):.2f}", "mm"],
            ["95th Percentile Width", f"{width_stats.get('p95_mm', 0):.2f}", "mm"],
            ["Crack Area", f"{measurement_results.get('area_mm2', 0):.2f}", "mm²"]
        ]
        
        meas_table = Table(measurement_data, colWidths=[6*cm, 4*cm, 3*cm])
        meas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f8f9fa')])
        ]))
        elements.append(meas_table)
        
        # Calibration info
        elements.append(Spacer(1, 10))
        calibration = measurement_results.get('calibration', {})
        scale = calibration.get('avg_scale_mm_per_px', 0) if calibration else 0
        elements.append(Paragraph(
            f"<i>Calibration: Brick-based scale factor = {scale:.4f} mm/pixel</i>", 
            ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=gray)
        ))
    else:
        elements.append(Paragraph("<i>Measurements not available.</i>", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # =========================================================================
    # CRACK CLASSIFICATION & LOCATION
    # =========================================================================
    elements.append(Paragraph("🎯 CRACK CHARACTERISTICS", header_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))
    elements.append(Spacer(1, 10))
    
    # Classification
    if classification_results:
        best_class = classification_results.get('best_classification', {})
        if best_class:
            crack_type = best_class.get('crack_type', 'Unknown').upper()
            confidence = best_class.get('confidence', 0)
            elements.append(Paragraph(f"<b>Crack Type:</b> {crack_type} (Confidence: {confidence:.1%})", normal_style))
    
    # Location
    if location_results and hasattr(location_results, 'success') and location_results.success:
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Primary Location:</b> {location_results.dominant_location.replace('-', ' ').title() if location_results.dominant_location else 'Unknown'}", normal_style))
        elements.append(Paragraph(f"<b>Orientation:</b> {location_results.orientation.title() if location_results.orientation else 'Unknown'}", normal_style))
        elements.append(Paragraph(f"<b>Propagation Direction:</b> {location_results.propagation_direction if location_results.propagation_direction else 'Unknown'}", normal_style))
    
    elements.append(Spacer(1, 20))
    
    # =========================================================================
    # FEMA 306 DIAGNOSIS DETAILS
    # =========================================================================
    if fema_diagnosis:
        elements.append(Paragraph("🏛️ FEMA 306 STRUCTURAL DIAGNOSIS", header_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))
        elements.append(Spacer(1, 10))
        
        # Input observations
        elements.append(Paragraph("Input Observations", subheader_style))
        if fema_input:
            obs_data = [
                ["Material", fema_input.get('material', 'URM')],
                ["Crack Orientation", fema_input.get('orientation', 'Unknown')],
                ["Crack Width", fema_input.get('width', 'Unknown')],
                ["Location", fema_input.get('location', 'Unknown')],
            ]
            obs_table = Table(obs_data, colWidths=[5*cm, 11*cm])
            obs_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), HexColor('#ecf0f1')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(obs_table)
        
        elements.append(Spacer(1, 15))
        
        # Failure mode details
        if fema_diagnosis.failure_mode:
            elements.append(Paragraph("Identified Failure Mode", subheader_style))
            fm = fema_diagnosis.failure_mode
            
            # Basic info table
            fm_data = [
                ["Property", "Value"],
                ["Name", fm.name or "Unknown"],
                ["ID", fm.id or "Unknown"],
                ["Material Type", fm.material or "URM"],
                ["Failure Type", fm.type or "Unknown"],
            ]
            
            fm_table = Table(fm_data, colWidths=[5*cm, 11*cm])
            fm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('BACKGROUND', (0, 1), (0, -1), HexColor('#ecf0f1')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(fm_table)
            elements.append(Spacer(1, 10))
            
            # Parse and format description
            elements.append(Paragraph("<b>Description:</b>", normal_style))
            description = fm.description or ""
            
            # Try to parse as JSON and extract key information
            try:
                if description.strip().startswith('{'):
                    desc_data = json.loads(description)
                    
                    # Extract key fields from the JSON structure
                    if 'identification_guidelines' in desc_data:
                        guidelines = desc_data['identification_guidelines']
                        if 'by_observation' in guidelines:
                            elements.append(Paragraph("<b>Visual Identification:</b>", normal_style))
                            elements.append(Paragraph(guidelines['by_observation'], normal_style))
                            elements.append(Spacer(1, 5))
                        if 'by_analysis' in guidelines:
                            elements.append(Paragraph("<b>Analysis Notes:</b>", normal_style))
                            elements.append(Paragraph(guidelines['by_analysis'], normal_style))
                            elements.append(Spacer(1, 5))
                    
                    # Extract damage levels summary
                    if 'damage_levels' in desc_data and desc_data['damage_levels']:
                        elements.append(Spacer(1, 10))
                        elements.append(Paragraph("<b>Damage Level Classifications:</b>", normal_style))
                        
                        damage_table_data = [["Level", "Criteria Summary", "λK", "λQ"]]
                        for dl in desc_data['damage_levels'][:4]:  # Limit to first 4 levels
                            level = dl.get('level', 'Unknown')
                            criteria = dl.get('criteria', [])
                            criteria_text = criteria[0][:80] + "..." if criteria and len(criteria[0]) > 80 else (criteria[0] if criteria else "N/A")
                            pf = dl.get('performance_factors', {})
                            lambda_k = str(pf.get('lambda_K', 'N/A'))
                            lambda_q = str(pf.get('lambda_Q', 'N/A'))
                            damage_table_data.append([level, criteria_text, lambda_k, lambda_q])
                        
                        damage_table = Table(damage_table_data, colWidths=[2.5*cm, 9*cm, 2*cm, 2*cm])
                        damage_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#95a5a6')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
                            ('PADDING', (0, 0), (-1, -1), 4),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
                        ]))
                        elements.append(damage_table)
                else:
                    # Plain text description
                    elements.append(Paragraph(description, normal_style))
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, just display as text (truncated if too long)
                if len(description) > 500:
                    description = description[:500] + "..."
                elements.append(Paragraph(description, normal_style))
        
        elements.append(Spacer(1, 15))
        
        # Confidence scores
        elements.append(Paragraph("Hybrid Confidence Score Breakdown", subheader_style))
        conf_data = [
            ["Score Component", "Value", "Weight"],
            ["LLM Confidence", f"{(fema_diagnosis.llm_confidence or 0):.1%}", "50%"],
            ["Retrieval Confidence", f"{(fema_diagnosis.retrieval_confidence or 0):.1%}", "30%"],
            ["Cross-Encoder Score", f"{(fema_diagnosis.cross_encoder_confidence or 0):.1%}", "20%"],
            ["HYBRID SCORE (Final)", f"{(fema_diagnosis.hybrid_score or 0):.1%}", "100%"]
        ]
        
        conf_table = Table(conf_data, colWidths=[6*cm, 4*cm, 3*cm])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c3e50')),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
            ('TEXTCOLOR', (0, -1), (-1, -1), HexColor('#ffffff')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(conf_table)
        
        elements.append(Spacer(1, 15))
        
        # Reasoning
        if fema_diagnosis.reasoning:
            elements.append(Paragraph("Analysis Reasoning", subheader_style))
            elements.append(Paragraph(fema_diagnosis.reasoning, normal_style))
        
        elements.append(PageBreak())
        
        # =========================================================================
        # CITATIONS & REFERENCES
        # =========================================================================
        elements.append(Paragraph("📚 FEMA 306 CITATIONS & REFERENCES", header_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#3498db')))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph(
            "The following sections from FEMA 306 were referenced in this diagnosis:", 
            normal_style
        ))
        elements.append(Spacer(1, 10))
        
        if fema_diagnosis.citations:
            for i, citation in enumerate(fema_diagnosis.citations, 1):
                elements.append(Paragraph(f"<b>Reference {i}</b>", subheader_style))
                elements.append(Paragraph(f"<b>Section:</b> {citation.section_id}", normal_style))
                elements.append(Paragraph(f"<b>Title:</b> {citation.title}", normal_style))
                if citation.relevance_score:
                    elements.append(Paragraph(f"<b>Relevance Score:</b> {citation.relevance_score:.4f}", normal_style))
                elements.append(Spacer(1, 5))
                elements.append(Paragraph(f"<i>Excerpt:</i>", normal_style))
                # Clean up citation text
                citation_text = citation.text_snippet.replace('\n', ' ').strip()
                if len(citation_text) > 800:
                    citation_text = citation_text[:800] + "..."
                elements.append(Paragraph(citation_text, citation_style))
                elements.append(Spacer(1, 10))
        else:
            elements.append(Paragraph("<i>No specific citations available.</i>", normal_style))
    
    # =========================================================================
    # FOOTER / DISCLAIMER
    # =========================================================================
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=gray))
    elements.append(Spacer(1, 10))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=gray,
        alignment=TA_JUSTIFY
    )
    
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated using automated crack analysis and FEMA 306 "
        "Retrieval-Augmented Generation (RAG) system. The analysis is intended to assist qualified "
        "structural engineers in their assessment and should not be used as the sole basis for "
        "structural decisions. All findings should be verified by a licensed professional engineer "
        "through on-site inspection before any remediation work is undertaken.",
        disclaimer_style
    ))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"<b>Reference Standard:</b> FEMA 306 - Evaluation of Earthquake Damaged Concrete and Masonry Wall Buildings",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Optionally save to file
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes


def get_severity_assessment(measurement_results: dict, fema_diagnosis) -> dict:
    """
    Generate a comprehensive severity assessment based on measurements and FEMA diagnosis.
    
    Returns dict with severity level, recommendations, and urgency.
    """
    severity_info = {
        "level": "Unknown",
        "urgency": "Unknown", 
        "color": "#7f8c8d",
        "recommendations": [],
        "immediate_actions": [],
        "monitoring_required": False
    }
    
    # Get width from measurements
    max_width_mm = 0
    if measurement_results:
        width_stats = measurement_results.get('width_stats', {})
        max_width_mm = width_stats.get('max_mm', 0)
    
    # Width-based severity (per FEMA 306 guidelines)
    if max_width_mm < 0.5:
        width_severity = "Minor"
    elif max_width_mm < 2.0:
        width_severity = "Moderate"
    elif max_width_mm < 5.0:
        width_severity = "Significant"
    else:
        width_severity = "Severe"
    
    # Combine with FEMA diagnosis if available
    if fema_diagnosis:
        fema_severity = fema_diagnosis.severity if fema_diagnosis.severity else "Unknown"
        damage_level = fema_diagnosis.damage_level if fema_diagnosis.damage_level else "Unknown"
        
        # Use the more severe assessment
        severity_order = ["Minor", "Moderate", "Significant", "Severe", "Critical"]
        width_idx = severity_order.index(width_severity) if width_severity in severity_order else 0
        fema_idx = -1
        for i, s in enumerate(severity_order):
            if s.lower() in fema_severity.lower():
                fema_idx = i
                break
        
        final_severity = severity_order[max(width_idx, fema_idx)] if fema_idx >= 0 else width_severity
        severity_info["level"] = final_severity
    else:
        severity_info["level"] = width_severity
    
    # Set urgency and color based on severity
    if severity_info["level"] in ["Minor"]:
        severity_info["urgency"] = "Low - Routine Monitoring"
        severity_info["color"] = "#27ae60"  # Green
        severity_info["recommendations"] = [
            "Monitor crack progression every 6 months",
            "Document with photographs for future comparison",
            "No immediate structural intervention required"
        ]
        severity_info["monitoring_required"] = True
        
    elif severity_info["level"] in ["Moderate"]:
        severity_info["urgency"] = "Medium - Schedule Assessment"
        severity_info["color"] = "#f39c12"  # Orange
        severity_info["recommendations"] = [
            "Schedule detailed structural assessment within 30 days",
            "Install crack monitors to track progression",
            "Identify and address potential water infiltration",
            "Consider preventive sealing of cracks"
        ]
        severity_info["immediate_actions"] = [
            "Install crack monitoring gauges"
        ]
        severity_info["monitoring_required"] = True
        
    elif severity_info["level"] in ["Significant"]:
        severity_info["urgency"] = "High - Prompt Action Required"
        severity_info["color"] = "#e67e22"  # Dark orange
        severity_info["recommendations"] = [
            "Immediate structural engineering assessment required",
            "Consider temporary shoring if load-bearing",
            "Restrict access to affected areas if safety concern",
            "Develop repair/strengthening plan"
        ]
        severity_info["immediate_actions"] = [
            "Contact structural engineer immediately",
            "Assess occupancy safety"
        ]
        severity_info["monitoring_required"] = True
        
    elif severity_info["level"] in ["Severe", "Critical"]:
        severity_info["urgency"] = "Critical - Immediate Action Required"
        severity_info["color"] = "#e74c3c"  # Red
        severity_info["recommendations"] = [
            "URGENT: Immediate structural engineering assessment",
            "Consider evacuation of affected areas",
            "Install emergency shoring as needed",
            "Document damage thoroughly for insurance/records",
            "Engage qualified contractor for emergency repairs"
        ]
        severity_info["immediate_actions"] = [
            "Evacuate if structural stability is questionable",
            "Contact structural engineer immediately",
            "Consider temporary shoring"
        ]
        severity_info["monitoring_required"] = False  # Beyond monitoring - needs repair
    
    return severity_info


@st.cache_resource
def load_fema_rag_agent(chroma_db_path: str = None):
    """
    Initialize the FEMA 306 RAG agent with caching.
    
    Args:
        chroma_db_path: Path to ChromaDB vector store
    
    Returns:
        Tuple of (StructuralAgent, error_message)
    """
    if not FEMA_RAG_AVAILABLE:
        return None, "FEMA RAG components not available. Please check the Independent study/RAG_updated folder."
    
    try:
        # Load environment variables from .env file
        env_path = os.path.join(
            os.path.dirname(__file__), 
            'Independent study', 'RAG_updated', '.env'
        )
        if os.path.exists(env_path):
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        # Default path
        if chroma_db_path is None:
            chroma_db_path = os.path.join(
                os.path.dirname(__file__), 
                'Independent study', 'RAG_updated', 'data', 'chroma_db'
            )
        
        # Initialize retriever
        retriever = SectionRetriever(db_path=chroma_db_path)
        
        # Initialize LLM service (uses GOOGLE_API_KEY env var)
        llm_service = GeminiLLMService()
        
        # Create agent
        agent = StructuralAgent(retriever=retriever, llm_service=llm_service)
        
        return agent, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


@st.cache_resource
def load_pipeline(yolo_path, seg_path, class_path, rag_dir=None):
    """Load pipeline with caching"""
    try:
        pipeline = CrackAnalysisPipeline(
            yolo_model_path=yolo_path,
            segmentation_model_path=seg_path,
            classification_model_path=class_path,
            rag_data_dir=rag_dir,
            device='cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu'
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Structural Crack Analysis System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model paths
        st.subheader("Model Paths")
        yolo_path = st.text_input(
            "YOLO Model Path",
            value="Crack_Detection_YOLO/crack_yolo_train/weights/best.pt",
            help="Path to YOLO detection model"
        )
        
        seg_path = st.text_input(
            "Segmentation Model Path",
            value="Masking_and_Classification_model/pretrained_net_G.pth",
            help="Path to segmentation model"
        )
        
        class_path = st.text_input(
            "Classification Model Path",
            value="Masking_and_Classification_model/crack_orientation_classifier.h5",
            help="Path to classification model"
        )
        
        rag_dir = st.text_input(
            "RAG Data Directory (Optional)",
            value="Rag_and_Reasoning/crack_analysis_rag/data",
            help="Directory for RAG system data"
        )
        
        use_rag = st.checkbox("Enable RAG Analysis", value=True)
        
        # FEMA 306 RAG Option
        if use_rag:
            use_fema_rag = st.checkbox(
                "🏛️ Use FEMA 306 Structural Diagnosis",
                value=True,
                help="Use FEMA 306 compliant RAG with Hybrid Confidence Scoring for structural failure mode diagnosis"
            )
            if use_fema_rag and FEMA_RAG_AVAILABLE:
                st.success("✅ FEMA 306 RAG Available")
            elif use_fema_rag and not FEMA_RAG_AVAILABLE:
                st.warning("⚠️ FEMA 306 RAG not available")
                use_fema_rag = False
        else:
            use_fema_rag = False
        
        st.markdown("---")
        st.subheader("📐 Brick Calibration")
        st.caption(f"📐 All images are standardized to **{STANDARD_IMAGE_WIDTH}px** width for consistent measurements.")
        
        # Brick type selection
        brick_type = st.selectbox(
            "Brick Type",
            options=["Standard Indian (90mm)", "Modular (90mm)", "Custom"],
            help="Select brick type to determine standard height for calibration"
        )
        
        if brick_type == "Custom":
            brick_height_mm = st.number_input(
                "Custom Brick Height (mm)",
                min_value=50.0,
                max_value=200.0,
                value=90.0,
                step=1.0,
                help="Enter the actual brick height in millimeters"
            )
        else:
            brick_height_mm = 90.0
            st.info(f"Using standard brick height: {brick_height_mm} mm")
        
        # Store in session state
        if 'brick_height_mm' not in st.session_state:
            st.session_state.brick_height_mm = brick_height_mm
        else:
            st.session_state.brick_height_mm = brick_height_mm
        
        # Calibration mode - updated with Interactive option
        calibration_mode = st.radio(
            "Calibration Mode",
            options=["Automatic", "Interactive 🎯 (Higher Accuracy)", "Manual Entry"],
            help="Automatic: Detects brick height from image. Interactive: Draw line on brick. Manual: Enter pixel height."
        )
        
        # Store calibration mode in session state
        st.session_state.calibration_mode = calibration_mode
        
        if calibration_mode == "Manual Entry":
            manual_brick_px = st.number_input(
                "Brick Height in Pixels",
                min_value=10,
                max_value=1000,
                value=100,
                step=5,
                help="Enter the brick height as measured in the image (pixels)"
            )
            if st.button("Apply Manual Calibration"):
                st.session_state.brick_height_px = manual_brick_px
                st.session_state.scale_mm_per_px = brick_height_mm / manual_brick_px
                st.session_state.brick_calibration_done = True
                st.success(f"✅ Scale set: {st.session_state.scale_mm_per_px:.4f} mm/pixel")
        
        elif calibration_mode == "Interactive 🎯 (Higher Accuracy)":
            st.info("👆 Click on TOP and BOTTOM edges of a brick in Step 5 to calibrate.")
            if st.button("Reset Interactive Calibration"):
                st.session_state.brick_calibration_done = False
                st.session_state.brick_height_px = None
                st.session_state.scale_mm_per_px = None
                st.session_state.click_points = []
                st.rerun()
        
        # Show current calibration
        if st.session_state.brick_calibration_done:
            st.success(f"📏 Scale: {st.session_state.scale_mm_per_px:.4f} mm/px")
        
        st.markdown("---")
        
        # Load pipeline button
        if st.button("🔄 Load Pipeline", type="primary"):
            with st.spinner("Loading models... This may take a few minutes..."):
                pipeline, error = load_pipeline(yolo_path, seg_path, class_path, rag_dir if use_rag else None)
                if error:
                    st.error(f"Error loading pipeline: {error}")
                    st.session_state.pipeline = None
                else:
                    st.session_state.pipeline = pipeline
                    st.success("✅ Pipeline loaded successfully!")
        
        # Pipeline status
        if st.session_state.pipeline:
            st.success("✅ Pipeline Ready")
        else:
            st.warning("⚠️ Pipeline not loaded")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Configure model paths in the sidebar
        2. Click "Load Pipeline" to initialize models
        3. Upload an image using the file uploader
        4. Click "Analyze Image" to run the complete pipeline
        5. View results in the main panel
        """)
    
    # Main content area
    if not st.session_state.pipeline:
        st.info("👈 Please load the pipeline using the sidebar before analyzing images.")
        return
    
    # File uploader
    st.markdown('<h2 class="step-header">📤 Upload Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing potential cracks",
        on_change=reset_image_state,
        key="crack_image_uploader"
    )
    
    if uploaded_file is not None:
        # Open and resize to standard size for consistency
        raw_image = Image.open(uploaded_file)
        image = resize_to_standard(raw_image, STANDARD_IMAGE_WIDTH)
        
        if raw_image.size != image.size:
            st.caption(f"📐 Image resized from {raw_image.size[0]}×{raw_image.size[1]} → {image.size[0]}×{image.size[1]} (standard width: {STANDARD_IMAGE_WIDTH}px)")
        
        st.image(image, caption="Uploaded Image")
        
        # Store standardized image in session state for later use
        st.session_state.uploaded_image = image
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            # Reset calibration for new image analysis
            st.session_state.brick_calibration_done = False
            st.session_state.scale_mm_per_px = None
            st.session_state.click_points = []
            # Save uploaded file temporarily
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_path = tmp_file.name
                # Convert to RGB if necessary (e.g. for PNG with transparency)
                if image.mode in ('RGBA', 'P', 'LA'):
                    rgb_image = image.convert('RGB')
                    rgb_image.save(tmp_path)
                else:
                    image.save(tmp_path)
            
            # Store path in session state
            st.session_state.uploaded_image_path = tmp_path
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run pipeline
                status_text.text("Running analysis pipeline...")
                progress_bar.progress(10)
                
                results = st.session_state.pipeline.process_image(
                    tmp_path,
                    use_rag=use_rag,
                    save_intermediate=True,
                    output_dir=tempfile.mkdtemp()
                )
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.results = results
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        if not results.get('success', False):
            st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        st.markdown("---")
        st.markdown('<h2 class="step-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Step 1: Detection Results
        st.markdown("### Step 1: Crack Detection (YOLO)")
        detection = results.get('detection', {})
        num_cracks = detection.get('num_cracks', 0)
        crack_detected = detection.get('crack_detected', False)
        
        if crack_detected:
            st.success(f"✅ **{num_cracks} crack(s) detected**")
            
            # Load original image and draw bounding boxes
            # Use stored image path or try to get from results
            image_path = st.session_state.uploaded_image_path or results.get('image_path')
            if image_path and os.path.exists(image_path):
                original_img = cv2.imread(image_path)
                # Convert BGR to RGB for display
                img_with_boxes = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)
            elif st.session_state.uploaded_image is not None:
                # Fallback: convert PIL image to numpy array (already RGB)
                img_with_boxes = np.array(st.session_state.uploaded_image.copy())
                if img_with_boxes.ndim == 2:
                    # Grayscale, convert to RGB
                    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)
                elif img_with_boxes.shape[2] == 4:
                    # RGBA, convert to RGB
                    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGBA2RGB)
                original_img = None
            else:
                original_img = None
                img_with_boxes = None
            
            if img_with_boxes is not None:
                
                # Draw bounding boxes
                detections = detection.get('detections', [])
                for i, det in enumerate(detections, 1):
                    x1, y1, x2, y2 = det['bbox']
                    confidence = det['confidence']
                    class_name = det['class']
                    
                    # Draw rectangle
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    # Draw label background
                    label = f"{class_name}: {confidence:.2%}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        img_with_boxes,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        (255, 0, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img_with_boxes,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                
                # Display image with bounding boxes
                st.image(img_with_boxes, caption=f"Detected Cracks with Bounding Boxes ({num_cracks} found)")
            
            # Display detection details
            detections = detection.get('detections', [])
            for i, det in enumerate(detections, 1):
                with st.expander(f"Crack {i} Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{det['confidence']:.2%}")
                    with col2:
                        st.metric("Class", det['class'])
                    st.write(f"Bounding Box: {det['bbox']}")
        else:
            st.info("ℹ️ **No cracks detected** in the image.")
            st.stop()
        
        # Step 2: Segmentation Results
        st.markdown("### Step 2: Image Segmentation (Cropped Regions)")
        segmentation = results.get('segmentation', {})
        if segmentation:
            masks = segmentation.get('masks', [])
            num_regions = segmentation.get('num_regions', 0)
            combined_mask_available = segmentation.get('combined_mask_available', False)
            
            if masks:
                st.success(f"✅ Segmentation complete for {num_regions} region(s)")
                
                # Show combined mask if available
                if combined_mask_available and 'combined_mask' in segmentation:
                    st.markdown("#### Combined Mask")
                    combined_mask = segmentation['combined_mask']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(combined_mask, caption="Combined Segmented Crack Mask")
                    with col2:
                        # Create overlay with original image
                        overlay_img = None
                        image_path = st.session_state.uploaded_image_path or results.get('image_path')
                        if image_path and os.path.exists(image_path):
                            overlay_img = cv2.imread(image_path)
                        elif st.session_state.uploaded_image is not None:
                            pil_img = np.array(st.session_state.uploaded_image)
                            if pil_img.ndim == 2:
                                overlay_img = cv2.cvtColor(pil_img, cv2.COLOR_GRAY2BGR)
                            elif pil_img.shape[2] == 3:
                                overlay_img = cv2.cvtColor(pil_img, cv2.COLOR_RGB2BGR)
                            elif pil_img.shape[2] == 4:
                                overlay_img = cv2.cvtColor(pil_img, cv2.COLOR_RGBA2BGR)
                        
                        if overlay_img is not None:
                            mask_resized = cv2.resize(combined_mask, (overlay_img.shape[1], overlay_img.shape[0]))
                            mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
                            overlay = cv2.addWeighted(overlay_img, 0.6, mask_colored, 0.4, 0)
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                    caption="Combined Overlay Visualization")
                
                # Show individual region masks
                if num_regions > 1:
                    st.markdown("#### Individual Region Masks")
                    cols = st.columns(min(3, num_regions))
                    
                    for i, mask_info in enumerate(masks):
                        with cols[i % 3]:
                            region_id = mask_info['region_id']
                            mask = mask_info['mask']
                            bbox = mask_info['bbox']
                            
                            st.image(mask, caption=f"Region {region_id} Mask")
                            st.caption(f"Bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            else:
                st.warning("⚠️ No segmentation masks generated")
        
        # Step 3: Classification Results
        st.markdown("### Step 3: Crack Classification (Per Region)")
        classification = results.get('classification', {})
        if classification:
            all_classifications = classification.get('all_classifications', [])
            best_classification = classification.get('best_classification')
            num_regions = classification.get('num_regions', 0)
            
            if best_classification:
                st.success(f"✅ **Best Classification: {best_classification['crack_type'].upper()}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Crack Type", best_classification['crack_type'].upper())
                with col2:
                    st.metric("Confidence", f"{best_classification['confidence']:.2%}")
                with col3:
                    st.metric("Best Region", f"Region {best_classification['region_id']}")
                
                # Type description
                type_descriptions = {
                    'vertical': 'Vertical cracks run up and down, often caused by thermal expansion or shrinkage.',
                    'horizontal': 'Horizontal cracks run along the length, often indicating flexural stress or overloading.',
                    'diagonal': 'Diagonal cracks appear at angles, typically caused by shear stress or foundation issues.',
                    'step': 'Step cracks follow mortar joints in masonry, often due to settlement or thermal movement.'
                }
                crack_type = best_classification['crack_type']
                st.info(f"💡 **{crack_type.upper()}**: {type_descriptions.get(crack_type, 'Unknown crack type')}")
                
                # Show all region classifications if multiple regions
                if num_regions > 1 and all_classifications:
                    st.markdown("#### All Region Classifications")
                    for cls_info in all_classifications:
                        region_id = cls_info['region_id']
                        crack_type = cls_info['crack_type']
                        confidence = cls_info['confidence']
                        bbox = cls_info['bbox']
                        
                        with st.expander(f"Region {region_id} - {crack_type.upper()} ({confidence:.1%})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {crack_type.upper()}")
                                st.write(f"**Confidence:** {confidence:.2%}")
                            with col2:
                                st.write(f"**Region ID:** {region_id}")
                                st.write(f"**Bounding Box:** [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                            
                            is_best = cls_info == best_classification
                            if is_best:
                                st.success("🏆 This is the best classification (highest confidence)")
            else:
                st.warning("⚠️ No valid classifications generated")
        
        # Step 4: RAG Analysis Results (Legacy - shown only if FEMA RAG is disabled)
        if use_rag and not use_fema_rag:
            st.markdown("### Step 4: RAG Analysis & Cause Determination (Original Full Image)")
            rag_analysis = results.get('rag_analysis', {})
            
            if rag_analysis:
                if rag_analysis.get('success', False):
                    st.success("✅ RAG analysis complete")
                    
                    # RAG Info Section - Show retrieved chunks and context
                    rag_info = rag_analysis.get('rag_info', {})
                    if rag_info.get('used_rag', False):
                        st.markdown("#### 🔍 RAG Retrieval Details")
                        
                        # Display search query
                        query = rag_info.get('query', 'N/A')
                        st.markdown(f"**Search Query:** `{query}`")
                        
                        # Display retrieved documents
                        retrieved_docs = rag_info.get('retrieved_documents', [])
                        num_docs = len(retrieved_docs)
                        st.markdown(f"**Retrieved Documents:** {num_docs} chunks")
                        
                        if retrieved_docs:
                            st.markdown("##### 📄 Retrieved Document Chunks:")
                            
                            for i, doc_info in enumerate(retrieved_docs, 1):
                                with st.expander(f"📑 Chunk {i} (Score: {doc_info.get('score', 0):.4f})"):
                                    # Display content
                                    content = doc_info.get('content', '')
                                    st.markdown("**Content:**")
                                    st.text_area(
                                        f"Chunk {i} Content",
                                        value=content,
                                        height=150,
                                        key=f"chunk_content_{i}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    # Display metadata
                                    metadata = doc_info.get('metadata', {})
                                    if metadata:
                                        st.markdown("**Metadata:**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if 'title' in metadata:
                                                st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                                            if 'source' in metadata:
                                                st.write(f"**Source:** {metadata.get('source', 'N/A')}")
                                        with col2:
                                            if 'crack_type' in metadata:
                                                st.write(f"**Crack Type:** {metadata.get('crack_type', 'N/A')}")
                                            if 'score' in doc_info:
                                                st.write(f"**Relevance Score:** {doc_info.get('score', 0):.4f}")
                                    
                                    # Show full metadata if available
                                    if metadata and len(metadata) > 3:
                                        with st.expander("View Full Metadata"):
                                            st.json(metadata)
                        
                        st.markdown("---")
                    
                    # Display comprehensive report
                    summary = rag_analysis.get('summary', {})
                    report = summary.get('comprehensive_report', '')
                    
                    if report:
                        st.markdown("#### 📋 Comprehensive Analysis Report")
                        st.markdown(f'<div class="info-box">{report}</div>', unsafe_allow_html=True)
                    
                else:
                    st.warning(f"⚠️ RAG analysis failed: {rag_analysis.get('error', 'Unknown error')}")
            else:
                st.info("ℹ️ RAG analysis not performed")
        
        # Step 4: Crack Dimensions (Length & Width Measurement)
        st.markdown("### Step 4: Crack Dimensions (Calibrated Measurement)")
        
        # Check if we have segmentation mask
        segmentation_for_measurement = results.get('segmentation', {})
        combined_mask = segmentation_for_measurement.get('combined_mask')
        
        if combined_mask is not None:
            # Get calibration mode from session state
            calibration_mode = st.session_state.get('calibration_mode', 'Automatic')
            
            # Load original image
            image_path = st.session_state.uploaded_image_path or results.get('image_path')
            if image_path and os.path.exists(image_path):
                original_img = cv2.imread(image_path)
                original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                original_img_rgb = np.array(st.session_state.uploaded_image)
                if len(original_img_rgb.shape) == 2:
                    original_img_rgb = cv2.cvtColor(original_img_rgb, cv2.COLOR_GRAY2RGB)
                original_img = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
            
            # Interactive mode calibration
            if calibration_mode == "Interactive 🎯 (Higher Accuracy)" and not st.session_state.brick_calibration_done:
                st.markdown("#### 🎯 Interactive Brick Height Calibration")
                
                if IMAGE_COORDINATES_AVAILABLE:
                    st.info("**Instructions:** Click on the **TOP edge** of a brick, then click on the **BOTTOM edge** of the same brick. The vertical distance will be used for calibration.")
                    
                    # Image is already standardized to STANDARD_IMAGE_WIDTH,
                    # so clicks are directly in the working pixel space.
                    display_img = original_img_rgb.copy()
                    
                    # Draw existing click points on the image
                    display_img_with_points = display_img.copy()
                    for i, pt in enumerate(st.session_state.click_points):
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, Red for second
                        label = "TOP" if i == 0 else "BOTTOM"
                        cv2.circle(display_img_with_points, (int(pt[0]), int(pt[1])), 8, color, -1)
                        cv2.circle(display_img_with_points, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), 2)
                        cv2.putText(display_img_with_points, label, (int(pt[0]) + 15, int(pt[1]) + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw line between points if we have 2
                    if len(st.session_state.click_points) == 2:
                        pt1 = tuple(map(int, st.session_state.click_points[0]))
                        pt2 = tuple(map(int, st.session_state.click_points[1]))
                        cv2.line(display_img_with_points, pt1, pt2, (255, 255, 0), 2)
                    
                    # Use streamlit_image_coordinates for click detection
                    coords = streamlit_image_coordinates(
                        Image.fromarray(display_img_with_points),
                        key="brick_calibration_image"
                    )
                    
                    # Process click
                    if coords is not None:
                        x, y = coords["x"], coords["y"]
                        
                        # Only add if it's a new click (not the same as last)
                        if len(st.session_state.click_points) < 2:
                            # Check if this is a genuinely new click
                            is_new = True
                            if st.session_state.click_points:
                                last_pt = st.session_state.click_points[-1]
                                if abs(last_pt[0] - x) < 5 and abs(last_pt[1] - y) < 5:
                                    is_new = False
                            
                            if is_new:
                                st.session_state.click_points.append((x, y))
                                st.rerun()
                    
                    # Show current status
                    if len(st.session_state.click_points) == 0:
                        st.warning("👆 Click on the **TOP edge** of a brick")
                    elif len(st.session_state.click_points) == 1:
                        st.warning("👆 Now click on the **BOTTOM edge** of the same brick")
                    elif len(st.session_state.click_points) >= 2:
                        # Calculate height — clicks are already in standardized image coords
                        y1 = st.session_state.click_points[0][1]
                        y2 = st.session_state.click_points[1][1]
                        brick_height_px = abs(y2 - y1)
                        
                        st.success(f"📏 **Measured brick height:** {brick_height_px:.1f} pixels")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("✅ Apply Calibration", type="primary"):
                                st.session_state.brick_height_px = brick_height_px
                                st.session_state.scale_mm_per_px = st.session_state.brick_height_mm / brick_height_px
                                st.session_state.brick_calibration_done = True
                                st.session_state.click_points = []
                                st.success(f"✅ Scale: {st.session_state.scale_mm_per_px:.4f} mm/pixel")
                                st.rerun()
                        with col2:
                            if st.button("🔄 Clear & Retry"):
                                st.session_state.click_points = []
                                st.rerun()
                        with col3:
                            st.write(f"Scale: {st.session_state.brick_height_mm / brick_height_px:.4f} mm/px")
                
                else:
                    # Fallback to manual entry if streamlit-image-coordinates not available
                    st.warning("⚠️ For the best experience, install: `pip install streamlit-image-coordinates`")
                    st.info("**Fallback Mode:** Enter brick height manually below.")
                    
                    # Display the image for reference
                    st.image(original_img_rgb, caption="Reference Image - Measure brick height in pixels", use_container_width=True)
                    
                    # Get image dimensions
                    h, w = original_img_rgb.shape[:2]
                    st.caption(f"Image dimensions: {w} x {h} pixels")
                    
                    # Interactive measurement using Y coordinates
                    st.markdown("##### Method 1: Enter Y coordinates of brick top and bottom")
                    col1, col2 = st.columns(2)
                    with col1:
                        y_top = st.number_input(
                            "Y coordinate of brick TOP edge (pixels from top)",
                            min_value=0,
                            max_value=h-1,
                            value=100,
                            step=1,
                            help="Enter the Y pixel position of the top edge of a brick"
                        )
                    with col2:
                        y_bottom = st.number_input(
                            "Y coordinate of brick BOTTOM edge (pixels from top)",
                            min_value=0,
                            max_value=h-1,
                            value=190,
                            step=1,
                            help="Enter the Y pixel position of the bottom edge of the same brick"
                        )
                    
                    calculated_height = abs(y_bottom - y_top)
                    st.write(f"📏 **Calculated brick height:** {calculated_height} pixels")
                    
                    if calculated_height > 10:
                        if st.button("✅ Apply This Measurement", type="primary", key="apply_interactive"):
                            st.session_state.brick_height_px = calculated_height
                            st.session_state.scale_mm_per_px = st.session_state.brick_height_mm / calculated_height
                            st.session_state.brick_calibration_done = True
                            st.success(f"✅ Calibration complete! Scale: {st.session_state.scale_mm_per_px:.4f} mm/pixel")
                            st.rerun()
                    else:
                        st.warning("⚠️ Brick height must be greater than 10 pixels")
                    
                    st.markdown("---")
                    st.markdown("##### Method 2: Direct pixel entry")
                    direct_height = st.number_input(
                        "Or directly enter brick height in pixels",
                        min_value=10,
                        max_value=500,
                        value=90,
                        step=5,
                        help="If you've measured the brick height using an external tool"
                    )
                    if st.button("✅ Apply Direct Entry", key="apply_direct"):
                        st.session_state.brick_height_px = direct_height
                        st.session_state.scale_mm_per_px = st.session_state.brick_height_mm / direct_height
                        st.session_state.brick_calibration_done = True
                        st.success(f"✅ Calibration complete! Scale: {st.session_state.scale_mm_per_px:.4f} mm/pixel")
                        st.rerun()
            
            # Automatic mode calibration
            elif calibration_mode == "Automatic" and not st.session_state.brick_calibration_done:
                st.info("📐 Attempting automatic brick height detection...")
                
                # Use BrickDetector for automatic calibration
                try:
                    detector = BrickDetector(
                        brick_height_mm=st.session_state.brick_height_mm
                    )
                    detection_result = detector.detect(original_img)
                    
                    if detection_result.success and detection_result.brick_height_px > 20:
                        st.session_state.brick_height_px = detection_result.brick_height_px
                        st.session_state.scale_mm_per_px = detection_result.scale_y_mm_per_px
                        st.session_state.brick_calibration_done = True
                        st.success(f"✅ Auto-detected brick height: {detection_result.brick_height_px:.1f} px → Scale: {detection_result.scale_y_mm_per_px:.4f} mm/px (Method: {detection_result.method})")
                    else:
                        st.warning(f"⚠️ Could not auto-detect brick height: {detection_result.error}")
                        st.info("💡 Try using **Interactive mode** for higher accuracy, or **Manual Entry** in the sidebar.")
                except Exception as e:
                    st.warning(f"⚠️ Brick detection failed: {str(e)}")
                    st.info("💡 Try using **Interactive mode** for higher accuracy, or **Manual Entry** in the sidebar.")
            
            # Perform measurement if calibrated
            if st.session_state.brick_calibration_done:
                try:
                    # Initialize measurement system
                    measurer = CrackMeasurerWithBrickCalibration(seg_model_path = "./Masking_and_Crack_Detection_Model/pretrained_net_G.pth")
                    
                    # Prepare mask for measurement
                    if isinstance(combined_mask, np.ndarray):
                        mask_for_measurement = combined_mask
                    else:
                        mask_for_measurement = np.array(combined_mask)
                    
                    # Convert to grayscale if needed
                    if len(mask_for_measurement.shape) == 3:
                        mask_for_measurement = cv2.cvtColor(mask_for_measurement, cv2.COLOR_BGR2GRAY)
                    
                    # Load original image for visualization
                    image_path = st.session_state.uploaded_image_path or results.get('image_path')
                    if image_path and os.path.exists(image_path):
                        original_img = cv2.imread(image_path)
                    else:
                        original_img = np.array(st.session_state.uploaded_image)
                        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                    
                    # Run measurement
                    measurement_results = measurer.measure(
                        image=original_img,
                        mask=mask_for_measurement,
                        scale_mm_per_px=st.session_state.scale_mm_per_px
                    )
                    st.session_state.measurement_results = measurement_results
                    
                    st.success("✅ Crack dimensions calculated!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        length_mm = measurement_results.get('length_mm', 0)
                        st.metric("Crack Length", f"{length_mm:.2f} mm")
                    with col2:
                        width_stats = measurement_results.get('width_stats', {})
                        avg_width_mm = width_stats.get('mean_mm', 0)
                        st.metric("Average Width", f"{avg_width_mm:.2f} mm")
                    with col3:
                        max_width_mm = width_stats.get('max_mm', 0)
                        st.metric("Maximum Width", f"{max_width_mm:.2f} mm")
                    
                    # Display skeleton visualization
                    skeleton = measurement_results.get('skeleton')
                    binary_mask = measurement_results.get('binary_mask')
                    
                    if skeleton is not None and binary_mask is not None:
                        st.markdown("#### 🦴 Skeleton Analysis Visualization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create skeleton overlay on mask
                            h, w = mask_for_measurement.shape[:2]
                            skeleton_viz = np.zeros((h, w, 3), dtype=np.uint8)
                            skeleton_viz[:, :, 0] = (binary_mask * 255).astype(np.uint8)  # Mask in blue
                            # Make skeleton thicker for visibility
                            kernel = np.ones((3, 3), np.uint8)
                            skeleton_thick = cv2.dilate(skeleton.astype(np.uint8) * 255, kernel, iterations=1)
                            skeleton_viz[skeleton_thick > 0] = [0, 0, 255]  # Red skeleton
                            st.image(skeleton_viz, caption="Crack Skeleton (Red) on Segmentation Mask")
                        
                        with col2:
                            # Skeleton only
                            skeleton_only = (skeleton.astype(np.uint8) * 255)
                            st.image(skeleton_only, caption="Skeleton Path (used for length calculation)")
                    
                    # Additional details
                    with st.expander("📏 Detailed Measurements"):
                        st.write(f"**Length (pixels):** {measurement_results.get('skeleton_pixels', 0)} px")
                        st.write(f"**Average Width (mm):** {width_stats.get('mean_mm', 0):.3f} mm")
                        st.write(f"**Median Width (mm):** {width_stats.get('median_mm', 0):.3f} mm")
                        st.write(f"**Maximum Width (mm):** {width_stats.get('max_mm', 0):.3f} mm")
                        st.write(f"**Minimum Width (mm):** {width_stats.get('min_mm', 0):.3f} mm")
                        st.write(f"**95th Percentile Width (mm):** {width_stats.get('p95_mm', 0):.3f} mm")
                        st.write(f"**Crack Area (mm²):** {measurement_results.get('area_mm2', 0):.2f} mm²")
                        st.write(f"**Scale Factor:** {st.session_state.scale_mm_per_px:.4f} mm/pixel")
                        st.write(f"**Brick Height (reference):** {st.session_state.brick_height_mm} mm = {st.session_state.brick_height_px:.1f} px")
                
                except Exception as e:
                    st.error(f"Error during measurement: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Brick calibration required. Please configure in sidebar (Manual Entry) or ensure bricks are visible in image.")
        else:
            st.warning("⚠️ No segmentation mask available for measurement.")
        
        # Step 5: Crack Location & Propagation Analysis
        st.markdown("### Step 5: Crack Location & Propagation Direction")
        
        if combined_mask is not None:
            try:
                # Initialize location analyzer
                analyzer = CrackLocationAnalyzer()
                
                # Prepare mask
                if isinstance(combined_mask, np.ndarray):
                    mask_for_location = combined_mask
                else:
                    mask_for_location = np.array(combined_mask)
                
                # Convert to grayscale if needed
                if len(mask_for_location.shape) == 3:
                    mask_for_location = cv2.cvtColor(mask_for_location, cv2.COLOR_BGR2GRAY)
                
                # Load original image for visualization
                image_path = st.session_state.uploaded_image_path or results.get('image_path')
                if image_path and os.path.exists(image_path):
                    original_img_bgr = cv2.imread(image_path)
                else:
                    original_img_pil = np.array(st.session_state.uploaded_image)
                    if len(original_img_pil.shape) == 2:
                        original_img_bgr = cv2.cvtColor(original_img_pil, cv2.COLOR_GRAY2BGR)
                    elif original_img_pil.shape[2] == 3:
                        original_img_bgr = cv2.cvtColor(original_img_pil, cv2.COLOR_RGB2BGR)
                    else:
                        original_img_bgr = cv2.cvtColor(original_img_pil, cv2.COLOR_RGBA2BGR)
                
                # Resize mask to match original image if needed
                if mask_for_location.shape[:2] != original_img_bgr.shape[:2]:
                    mask_for_location = cv2.resize(mask_for_location, 
                                                    (original_img_bgr.shape[1], original_img_bgr.shape[0]))
                
                # Run location analysis
                location_result = analyzer.analyze(mask_for_location, original_img_bgr)
                st.session_state.location_results = location_result
                
                if location_result.success:
                    st.success("✅ Location analysis complete!")
                    
                    # Display main results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        dominant = location_result.dominant_location.replace('_', ' ').title() if location_result.dominant_location else "Unknown"
                        st.metric("Dominant Location (Origin)", dominant)
                    with col2:
                        secondary = location_result.secondary_location.replace('_', ' ').title() if location_result.secondary_location else "Unknown"
                        st.metric("Secondary Location (Terminus)", secondary)
                    with col3:
                        propagation = location_result.propagation_direction if location_result.propagation_direction else "Unknown"
                        st.metric("Propagation Direction", propagation)
                    with col4:
                        orientation = location_result.orientation.title() if location_result.orientation else "Unknown"
                        st.metric("Crack Orientation", orientation)
                    
                    # Intensity information
                    if location_result.origin_intensity and location_result.terminus_intensity:
                        st.info(f"💡 **Crack Depth Indicator:** Origin intensity: {location_result.origin_intensity:.4f} (deeper) → Terminus intensity: {location_result.terminus_intensity:.4f} (shallower)")
                    
                    # Visualizations
                    st.markdown("#### 📍 Location Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Grid overlay with START/END markers
                        grid_overlay = analyzer._draw_region_grid(
                            original_img_bgr, 
                            endpoints=location_result.crack_endpoints
                        )
                        st.image(cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2RGB), 
                                caption="Grid Overlay with Origin (🟢 START) and Terminus (🔴 END)")
                    
                    with col2:
                        # Heatmap overlay
                        heatmap_overlay = analyzer._create_heatmap_overlay_image(
                            original_img_bgr, mask_for_location, alpha=0.5, blur_sigma=15.0
                        )
                        st.image(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), 
                                caption="Crack Density Heatmap")
                    
                    # Propagation description
                    with st.expander("📖 Propagation Analysis Details"):
                        # Format intensity values safely
                        origin_int_str = f"{location_result.origin_intensity:.4f}" if location_result.origin_intensity else "N/A"
                        terminus_int_str = f"{location_result.terminus_intensity:.4f}" if location_result.terminus_intensity else "N/A"
                        
                        st.markdown(f"""
                        **Origin (Start Point):** {dominant}
                        - This is where the crack originated based on higher intensity (deeper crack damage)
                        - Origin Intensity: {origin_int_str}
                        
                        **Terminus (End Point):** {secondary}
                        - This is where the crack terminates based on lower intensity (shallower)
                        - Terminus Intensity: {terminus_int_str}
                        
                        **Propagation Direction:** {propagation}
                        - The crack propagated from the origin towards the terminus
                        
                        **Crack Orientation:** {location_result.orientation.title() if location_result.orientation else 'Unknown'}
                        - Aspect Ratio: {location_result.aspect_ratio:.2f} (width/height ratio of bounding box)
                        - Spread Type: {location_result.spread_type.replace('_', ' ').title() if location_result.spread_type else 'Unknown'}
                        
                        **Coverage & Position:**
                        - Coverage Ratio: {location_result.coverage_ratio*100:.2f}% of image area
                        - Centroid: ({location_result.centroid[0]:.1f}, {location_result.centroid[1]:.1f}) pixels
                        
                        **Physical Interpretation:**
                        - Cracks typically start deeper at the point of stress concentration
                        - As the crack propagates, it becomes shallower towards the terminus
                        - This helps identify the source of structural stress
                        """)
                        
                        # Region distribution
                        v_pos = location_result.vertical_position
                        h_pos = location_result.horizontal_position
                        
                        st.markdown("**Vertical Distribution:**")
                        for region, percentage in sorted(v_pos.items(), key=lambda x: x[1], reverse=True):
                            region_name = region.replace('_', ' ').title()
                            st.write(f"- {region_name}: {percentage*100:.1f}%")
                        
                        st.markdown("**Horizontal Distribution:**")
                        for region, percentage in sorted(h_pos.items(), key=lambda x: x[1], reverse=True):
                            region_name = region.replace('_', ' ').title()
                            st.write(f"- {region_name}: {percentage*100:.1f}%")
                else:
                    st.warning(f"⚠️ Location analysis failed: {getattr(location_result, 'error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Error during location analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ No segmentation mask available for location analysis.")
        
        # Step 6: FEMA 306 Structural Diagnosis (Using calculated metrics)
        if use_rag and use_fema_rag:
            st.markdown("---")
            st.markdown("### Step 6: 🏛️ FEMA 306 Structural Failure Mode Diagnosis")
            
            # Check if we have the required data
            measurement_results = st.session_state.get('measurement_results')
            location_results = st.session_state.get('location_results')
            classification_data = classification.get('best_classification') if classification else None
            
            # =====================================================================
            # PREREQUISITE CHECK - Ensure all inputs are available
            # =====================================================================
            has_classification = classification_data is not None
            has_measurement = measurement_results is not None
            has_location = location_results and hasattr(location_results, 'success') and location_results.success
            
            # Show prerequisites status
            st.markdown("#### Prerequisites Check")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_classification:
                    st.success("✅ Step 3: Classification Complete")
                else:
                    st.warning("⏳ Step 3: Classification Pending")
            
            with col2:
                if has_measurement:
                    st.success("✅ Step 5: Measurement Complete")
                else:
                    st.warning("⏳ Step 5: Measurement Pending")
            
            with col3:
                if has_location:
                    st.success("✅ Step 6: Location Analysis Complete")
                else:
                    st.warning("⏳ Step 6: Location Analysis Pending")
            
            # Check if minimum requirements are met
            min_requirements_met = has_classification and (has_measurement or has_location)
            
            if not min_requirements_met:
                st.error("""
                ⚠️ **Cannot run FEMA 306 Diagnosis** - Missing required inputs:
                
                **Minimum Requirements:**
                - ✅ Crack Classification (Step 3) - **Required** for orientation
                - ✅ Crack Measurement (Step 5) OR Location Analysis (Step 6) - **At least one required**
                
                Please complete the steps above before running FEMA diagnosis.
                """)
                st.stop()
            
            st.markdown("---")
            
            # Try to detect structural component using WALL_Model
            structural_component = None
            if WALL_MODEL_AVAILABLE and combined_mask is not None:
                try:
                    # Get crack centroid from location results
                    crack_centroid = None
                    if location_results and hasattr(location_results, 'centroid'):
                        # Normalize centroid to 0-1 range
                        h, w = combined_mask.shape[:2]
                        crack_centroid = (
                            location_results.centroid[0] / w if w > 0 else 0.5,
                            location_results.centroid[1] / h if h > 0 else 0.5
                        )
                    
                    # Get original image for WALL_Model
                    original_img_for_wall = None
                    if st.session_state.uploaded_image is not None:
                        original_img_for_wall = np.array(st.session_state.uploaded_image)
                        if len(original_img_for_wall.shape) == 3 and original_img_for_wall.shape[2] == 3:
                            original_img_for_wall = cv2.cvtColor(original_img_for_wall, cv2.COLOR_RGB2BGR)
                    
                    if original_img_for_wall is not None:
                        structural_component = detect_structural_component(original_img_for_wall, crack_centroid)
                except Exception as e:
                    print(f"WALL_Model detection error: {e}")
                    structural_component = None
            
            # Show input data summary with source tracking
            with st.expander("📊 Input Data for FEMA Diagnosis (All Sources)", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔍 Classification Model (Step 3):**")
                    if classification_data:
                        crack_type = classification_data.get('crack_type', 'N/A')
                        confidence = classification_data.get('confidence', 0)
                        st.write(f"- **Crack Type:** `{crack_type.upper()}`")
                        st.write(f"- Confidence: {confidence:.2%}")
                        st.write(f"- Region: {classification_data.get('region_id', 'N/A')}")
                        st.success("✅ Orientation extracted from classification")
                    else:
                        st.write("- ⚠️ Classification data not available")
                        st.warning("Will use location analyzer for orientation")
                
                with col2:
                    st.markdown("**📏 Brick-Calibrated Measurement (Step 5):**")
                    if measurement_results:
                        width_stats = measurement_results.get('width_stats', {})
                        st.write(f"- **Max Width:** `{width_stats.get('max_mm', 0):.2f} mm`")
                        st.write(f"- Mean Width: {width_stats.get('mean_mm', 0):.2f} mm")
                        st.write(f"- **Length:** `{measurement_results.get('length_mm', 0):.2f} mm`")
                        calibration = measurement_results.get('calibration', {})
                        if calibration:
                            st.write(f"- Scale: {calibration.get('avg_scale_mm_per_px', 0):.4f} mm/px")
                        st.success("✅ Dimensions from brick calibration")
                    else:
                        st.write("- ⚠️ Measurement data not available")
                        st.warning("Width will be marked as Unknown")
                
                with col3:
                    st.markdown("**📍 Location Analysis (Step 6):**")
                    if location_results and hasattr(location_results, 'success') and location_results.success:
                        st.write(f"- Grid Location: `{location_results.dominant_location.replace('-', ' ').title() if location_results.dominant_location else 'N/A'}`")
                        st.write(f"- Orientation: {location_results.orientation.title() if location_results.orientation else 'N/A'}")
                        st.write(f"- Propagation: {location_results.propagation_direction if location_results.propagation_direction else 'N/A'}")
                    else:
                        st.write("- ⚠️ Location analysis not available")
                    
                    # WALL_Model structural detection
                    st.markdown("**🏗️ WALL_Model (Structural):**")
                    if structural_component and structural_component.get('component_type') != 'Unknown':
                        comp_type = structural_component.get('component_type', 'Unknown')
                        comp_conf = structural_component.get('confidence', 0)
                        st.write(f"- **Component:** `{comp_type}`")
                        st.write(f"- Confidence: {comp_conf:.0%}")
                        st.success("✅ Structural component detected")
                    elif WALL_MODEL_AVAILABLE:
                        st.write("- Component: Wall Panel (default)")
                        st.info("No specific component detected")
                    else:
                        st.write("- ⚠️ WALL_Model not available")
            
            # Build FEMA input from all calculated metrics
            fema_input = build_fema_input(
                classification_result=classification_data,
                measurement_result=measurement_results,
                location_result=location_results if location_results and hasattr(location_results, 'success') and location_results.success else None,
                structural_component=structural_component
            )
            
            # Display the FEMA input with metadata
            st.markdown("#### 📋 FEMA 306 Query Input (Final)")
            
            # Get metadata for display
            metadata = fema_input.get('_metadata', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Material:** {fema_input['material']}
                **Orientation:** {fema_input['orientation']} _(from: {metadata.get('orientation_source', 'unknown')})_
                **Width:** {fema_input['width']}
                """)
            with col2:
                st.info(f"""
                **Location:** {fema_input['location']} _(from: {metadata.get('location_source', 'unknown')})_
                **Description:** {fema_input['description']}
                """)
            
            # Run FEMA diagnosis
            if st.button("🔍 Run FEMA 306 Diagnosis", type="primary", key="run_fema_diagnosis"):
                with st.spinner("Running FEMA 306 structural diagnosis..."):
                    try:
                        # Load FEMA RAG agent
                        if st.session_state.fema_rag_agent is None:
                            agent, error = load_fema_rag_agent()
                            if error:
                                st.error(f"Failed to load FEMA RAG agent: {error}")
                            else:
                                st.session_state.fema_rag_agent = agent
                        
                        if st.session_state.fema_rag_agent:
                            # Prepare input for diagnosis (remove metadata, keep only FEMA fields)
                            diagnosis_input = {
                                "material": fema_input["material"],
                                "orientation": fema_input["orientation"],
                                "width": fema_input["width"],
                                "location": fema_input["location"],
                                "description": fema_input["description"]
                            }
                            
                            # Run diagnosis
                            diagnosis = st.session_state.fema_rag_agent.diagnose(diagnosis_input)
                            st.session_state.fema_diagnosis = diagnosis
                            
                            if diagnosis:
                                st.success("✅ FEMA 306 Diagnosis Complete!")
                            else:
                                st.warning("⚠️ Could not determine failure mode from the given observations.")
                        else:
                            st.error("FEMA RAG agent not available.")
                    
                    except Exception as e:
                        st.error(f"Error during FEMA diagnosis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display diagnosis results
            if st.session_state.fema_diagnosis:
                diagnosis = st.session_state.fema_diagnosis
                
                st.markdown("---")
                st.markdown("#### 🎯 Diagnosis Results")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    failure_mode_name = diagnosis.failure_mode.name if diagnosis.failure_mode else "Unknown"
                    # Extract short name
                    short_name = failure_mode_name.split()[-1] if failure_mode_name else "Unknown"
                    st.metric("Failure Mode", short_name)
                
                with col2:
                    hybrid_score = diagnosis.hybrid_score if diagnosis.hybrid_score else 0.0
                    st.metric("Hybrid Score", f"{hybrid_score:.2%}")
                
                with col3:
                    damage_level = diagnosis.damage_level if diagnosis.damage_level else "Unknown"
                    st.metric("Damage Level", damage_level)
                
                with col4:
                    severity = diagnosis.severity if diagnosis.severity else "Unknown"
                    st.metric("Severity", severity)
                
                # Hybrid Confidence Score Breakdown
                st.markdown("#### 📊 Hybrid Confidence Score Breakdown")
                st.markdown("""
                The hybrid score combines three confidence measures for reliable diagnosis:
                
                **Formula:** `Score = 0.5 × LLM + 0.3 × Retrieval + 0.2 × CrossEncoder`
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    llm_conf = diagnosis.llm_confidence if diagnosis.llm_confidence else 0.0
                    st.metric("LLM Confidence (50%)", f"{llm_conf:.2%}")
                    st.caption("Self-reported confidence from Gemini")
                with col2:
                    rc = diagnosis.retrieval_confidence if diagnosis.retrieval_confidence else 0.0
                    st.metric("Retrieval Confidence (30%)", f"{rc:.2%}")
                    st.caption("Softmax of embedding similarities")
                with col3:
                    rec = diagnosis.cross_encoder_confidence if diagnosis.cross_encoder_confidence else 0.0
                    st.metric("Cross-Encoder (20%)", f"{rec:.2%}")
                    st.caption("BERT re-ranking verification")
                
                # Progress bar for hybrid score
                st.progress(min(hybrid_score, 1.0))
                
                # Failure Mode Details
                st.markdown("#### 📖 Failure Mode Details")
                
                if diagnosis.failure_mode:
                    with st.expander("🔍 Identified Failure Mode", expanded=True):
                        st.markdown(f"**Name:** {diagnosis.failure_mode.name}")
                        st.markdown(f"**ID:** {diagnosis.failure_mode.id}")
                        st.markdown(f"**Material:** {diagnosis.failure_mode.material}")
                        st.markdown(f"**Type:** {diagnosis.failure_mode.type}")
                        
                        st.markdown("**Description:**")
                        st.text_area(
                            "Failure Mode Description",
                            value=diagnosis.failure_mode.description,
                            height=200,
                            key="fema_failure_mode_desc",
                            label_visibility="collapsed"
                        )
                
                # Reasoning
                if diagnosis.reasoning:
                    with st.expander("💭 Scoped Reasoning & Analysis", expanded=True):
                        st.markdown(diagnosis.reasoning)
                
                # Citations
                if diagnosis.citations:
                    with st.expander("📚 FEMA 306 Citations"):
                        for i, citation in enumerate(diagnosis.citations, 1):
                            st.markdown(f"**Citation {i}:**")
                            st.markdown(f"- **Section:** {citation.section_id}")
                            st.markdown(f"- **Title:** {citation.title}")
                            if citation.relevance_score:
                                st.markdown(f"- **Relevance:** {citation.relevance_score:.4f}")
                            st.text_area(
                                f"Citation {i} Snippet",
                                value=citation.text_snippet,
                                height=150,
                                key=f"citation_{i}",
                                label_visibility="collapsed"
                            )
                
                # Export diagnosis as JSON
                st.markdown("---")
                diagnosis_json = {
                    "failure_mode": {
                        "id": diagnosis.failure_mode.id if diagnosis.failure_mode else None,
                        "name": diagnosis.failure_mode.name if diagnosis.failure_mode else None,
                        "material": diagnosis.failure_mode.material if diagnosis.failure_mode else None,
                        "type": diagnosis.failure_mode.type if diagnosis.failure_mode else None,
                    },
                    "confidence_score": diagnosis.confidence_score,
                    "hybrid_score": diagnosis.hybrid_score,
                    "llm_confidence": diagnosis.llm_confidence,
                    "retrieval_confidence": diagnosis.retrieval_confidence,
                    "cross_encoder_confidence": diagnosis.cross_encoder_confidence,
                    "damage_level": diagnosis.damage_level,
                    "severity": diagnosis.severity,
                    "reasoning": diagnosis.reasoning,
                    "input_observations": {
                        "material": fema_input.get("material"),
                        "orientation": fema_input.get("orientation"),
                        "width": fema_input.get("width"),
                        "location": fema_input.get("location"),
                        "description": fema_input.get("description")
                    },
                    "data_sources": fema_input.get("_metadata", {})
                }
                
                st.download_button(
                    "📥 Download FEMA Diagnosis (JSON)",
                    data=json.dumps(diagnosis_json, indent=2),
                    file_name="fema_306_diagnosis.json",
                    mime="application/json"
                )
        
        # =====================================================================
        # FINAL REPORT SECTION
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📝 Final Analysis Summary")
        
        # Pipeline summary
        summary_text = st.session_state.pipeline.get_summary(results)
        
        with st.expander("View Pipeline Summary", expanded=False):
            st.code(summary_text, language='text')
        
        # =====================================================================
        # EXPORT SECTION - PDF AND OTHER FORMATS
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📤 Export Complete Report")
        
        st.markdown("""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border: 1px solid #c3e6cb;">
            <h4 style="color: #155724; margin: 0;">📄 Professional PDF Report</h4>
            <p style="color: #155724; margin: 5px 0;">
                Generate a comprehensive PDF report suitable for on-site use by structural engineers.
                Includes all measurements, FEMA 306 diagnosis, citations, and recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Check if all data is ready for PDF
        can_generate_pdf = PDF_AVAILABLE and (
            st.session_state.get('measurement_results') or 
            st.session_state.get('fema_diagnosis')
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if can_generate_pdf:
                if st.button("📄 Generate PDF Report", type="primary", key="generate_pdf"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            # Get FEMA input if available
                            fema_input_for_pdf = None
                            if use_rag and use_fema_rag:
                                classification_data = classification.get('best_classification') if classification else None
                                location_results_for_pdf = st.session_state.get('location_results')
                                fema_input_for_pdf = build_fema_input(
                                    classification_result=classification_data,
                                    measurement_result=st.session_state.get('measurement_results'),
                                    location_result=location_results_for_pdf if location_results_for_pdf and hasattr(location_results_for_pdf, 'success') and location_results_for_pdf.success else None,
                                    structural_component=None
                                )
                            
                            # Generate PDF
                            pdf_bytes = generate_analysis_pdf(
                                image_path=st.session_state.uploaded_image_path,
                                detection_results=results.get('detection', {}),
                                classification_results=results.get('classification', {}),
                                measurement_results=st.session_state.get('measurement_results'),
                                location_results=st.session_state.get('location_results'),
                                fema_input=fema_input_for_pdf,
                                fema_diagnosis=st.session_state.get('fema_diagnosis')
                            )
                            
                            # Store in session state for download
                            st.session_state['pdf_report'] = pdf_bytes
                            st.success("✅ PDF Report Generated!")
                            
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Download button if PDF is ready
                if st.session_state.get('pdf_report'):
                    report_date = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=st.session_state['pdf_report'],
                        file_name=f"crack_analysis_report_{report_date}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            else:
                if not PDF_AVAILABLE:
                    st.warning("📦 ReportLab not installed. Run: `pip install reportlab`")
                else:
                    st.info("Complete crack measurement or FEMA diagnosis to generate PDF report.")
        
        with col2:
            # Download FEMA diagnosis JSON if available
            if st.session_state.get('fema_diagnosis'):
                diagnosis = st.session_state.fema_diagnosis
                diagnosis_json = {
                    "report_generated": datetime.now().isoformat(),
                    "failure_mode": {
                        "id": diagnosis.failure_mode.id if diagnosis.failure_mode else None,
                        "name": diagnosis.failure_mode.name if diagnosis.failure_mode else None,
                        "material": diagnosis.failure_mode.material if diagnosis.failure_mode else None,
                        "type": diagnosis.failure_mode.type if diagnosis.failure_mode else None,
                        "description": diagnosis.failure_mode.description if diagnosis.failure_mode else None,
                    },
                    "confidence_scores": {
                        "hybrid_score": diagnosis.hybrid_score,
                        "llm_confidence": diagnosis.llm_confidence,
                        "retrieval_confidence": diagnosis.retrieval_confidence,
                        "cross_encoder_confidence": diagnosis.cross_encoder_confidence,
                    },
                    "assessment": {
                        "damage_level": diagnosis.damage_level,
                        "severity": diagnosis.severity,
                    },
                    "reasoning": diagnosis.reasoning,
                    "citations": [
                        {
                            "section_id": c.section_id,
                            "title": c.title,
                            "relevance_score": c.relevance_score,
                            "text_snippet": c.text_snippet
                        } for c in (diagnosis.citations or [])
                    ]
                }
                
                st.download_button(
                    "📋 Download JSON Report",
                    data=json.dumps(diagnosis_json, indent=2),
                    file_name="fema_306_diagnosis_full.json",
                    mime="application/json"
                )
            else:
                st.info("Run FEMA diagnosis to export JSON.")
        
        with col3:
            # Download combined mask if available
            if segmentation.get('combined_mask_available') and 'combined_mask_path' in segmentation:
                combined_mask_path = segmentation['combined_mask_path']
                if os.path.exists(combined_mask_path):
                    with open(combined_mask_path, 'rb') as f:
                        st.download_button(
                            "🖼️ Download Segmentation Mask",
                            f.read(),
                            file_name="segmentation_mask.jpg",
                            mime="image/jpeg"
                        )
            elif segmentation.get('masks'):
                masks = segmentation.get('masks', [])
                if masks and 'mask_path' in masks[0]:
                    mask_path = masks[0]['mask_path']
                    if os.path.exists(mask_path):
                        with open(mask_path, 'rb') as f:
                            st.download_button(
                                "🖼️ Download Segmentation Mask",
                                f.read(),
                                file_name="segmentation_mask.jpg",
                                mime="image/jpeg"
                            )
            else:
                st.info("No segmentation mask available.")
        
        # Additional export - Text summary
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📝 Download Text Summary",
                summary_text,
                file_name="crack_analysis_summary.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export all measurements as CSV-like format
            if st.session_state.get('measurement_results'):
                meas = st.session_state.measurement_results
                width_stats = meas.get('width_stats', {})
                meas_csv = f"""Crack Analysis Measurements
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MEASUREMENTS
============
Crack Length (mm),{meas.get('length_mm', 0) or 0:.2f}
Maximum Width (mm),{width_stats.get('max_mm', 0) or 0:.2f}
Average Width (mm),{width_stats.get('mean_mm', 0) or 0:.2f}
Median Width (mm),{width_stats.get('median_mm', 0) or 0:.2f}
Minimum Width (mm),{width_stats.get('min_mm', 0) or 0:.2f}
95th Percentile Width (mm),{width_stats.get('p95_mm', 0) or 0:.2f}
Crack Area (mm²),{meas.get('area_mm2', 0) or 0:.2f}
Skeleton Pixels,{meas.get('skeleton_pixels', 0)}
Scale (mm/px),{st.session_state.get('scale_mm_per_px', 0):.4f}
"""
                st.download_button(
                    "📊 Download Measurements (CSV)",
                    meas_csv,
                    file_name="crack_measurements.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()

