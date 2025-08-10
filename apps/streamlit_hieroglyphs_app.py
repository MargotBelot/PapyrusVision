#!/usr/bin/env python3
"""
PapyrusVision Hieroglyphs Detection and Analysis - Complete Web Application

Interactive web interface with Digital Paleography Tool for complete
hieroglyph detection, analysis, cropping, and catalog generation.

Features:
- Main detection interface with drag-and-drop upload
- Digital Paleography Tool for creating sign catalogs
- Batch processing capabilities
- Interactive HTML catalog generation with downloads
- ZIP file export of all crops and catalogs
- Unicode mapping and descriptions integration
- Multi-page application with navigation

Usage:
    streamlit run streamlit_hieroglyphs_app.py
"""

import streamlit as st
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import glob
from datetime import datetime
from pathlib import Path
import io
import base64
import tempfile
import zipfile
import shutil
from collections import defaultdict

def resize_image_for_model(image, min_size=800, max_size=1333):
    """
    Resize image to match training data dimensions used by the Detectron2 model.
    
    Args:
        image: PIL Image object
        min_size: Minimum size for the shorter side (default: 800)
        max_size: Maximum size for the longer side (default: 1333)
    
    Returns:
        PIL Image: Resized image maintaining aspect ratio, converted to RGB if needed
    """
    # Convert RGBA to RGB if needed (for JPEG compatibility)
    if image.mode == 'RGBA':
        # Create a white background
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = rgb_image
    elif image.mode not in ['RGB', 'L']:
        # Convert other modes to RGB
        image = image.convert('RGB')
    
    original_width, original_height = image.size
    
    # Determine the scaling factor
    shorter_side = min(original_width, original_height)
    longer_side = max(original_width, original_height)
    
    # Scale based on shorter side to meet min_size requirement
    scale_factor = min_size / shorter_side
    
    # Check if scaling would make longer side exceed max_size
    if longer_side * scale_factor > max_size:
        scale_factor = max_size / longer_side
    
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

sys.path.append(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)

# Import the Digital Paleography Tool
try:
    from digital_paleography_tool import DigitalPaleographyTool
except ImportError:
    st.error("Could not import DigitalPaleographyTool. Please ensure it exists.")

# Page configuration
st.set_page_config(
    page_title="PapyrusVision - Enhanced Hieroglyph Analysis",
    page_icon=":amphora:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #CD853F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F5DEB3, #DEB887);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .paleography-card {
        background: linear-gradient(135deg, #8B4513, #D2B48C);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
        text-align: center;
    }
    .feature-box {
        background: #FFF8DC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #DAA520;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #DAA520, #B8860B);
    }
    .nav-button {
        background: linear-gradient(135deg, #8B4513, #A0522D);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background: linear-gradient(135deg, #A0522D, #8B4513);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detectron_model():
    """Load the trained Detectron2 model"""
    try:
        import detectron2
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer, ColorMode
        
        # Find latest model
        model_dirs = glob.glob(os.path.join(MODELS_DIR, 'hieroglyph_model_*'))
        if not model_dirs:
            st.error("No trained models found!")
            return None, None, None
        
        latest_model_dir = sorted(model_dirs)[-1]
        model_name = os.path.basename(latest_model_dir)
        
        # Load model info
        model_info_file = os.path.join(latest_model_dir, 'model_info.json')
        if not os.path.exists(model_info_file):
            st.error(f"Model info not found: {model_info_file}")
            return None, None, None
        
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)
        
        # Load Unicode mapping
        unicode_mapping = {}
        unicode_file = os.path.join(DATA_DIR, 'annotations', 'gardiner_unicode_mapping.json')
        if os.path.exists(unicode_file):
            with open(unicode_file, 'r') as f:
                unicode_data = json.load(f)
            
            for gardiner_code, info in unicode_data.items():
                unicode_mapping[gardiner_code] = {
                    'unicode_codes': info.get('unicode_codes', []),
                    'description': info.get('description', 'Unknown'),
                    'unicode_symbol': get_unicode_symbol(info.get('unicode_codes', []))
                }
        
        # Set up model configuration
        cfg = get_cfg()
        config_file = os.path.join(latest_model_dir, 'config.yaml')
        if os.path.exists(config_file):
            cfg.merge_from_file(config_file)
        
        cfg.MODEL.WEIGHTS = os.path.join(latest_model_dir, 'model_final.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_info['detection_threshold']
        cfg.MODEL.DEVICE = 'cpu'  # Force CPU usage
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        return predictor, model_info, unicode_mapping
        
    except ImportError as e:
        st.error(f"Error importing Detectron2: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def get_unicode_symbol(unicode_codes):
    """Convert Unicode codes to actual symbols"""
    if not unicode_codes:
        return ""
    
    for code in unicode_codes:
        if code.startswith('U+'):
            try:
                unicode_int = int(code[2:], 16)
                return chr(unicode_int)
            except (ValueError, OverflowError):
                continue
    return ""

def predict_hieroglyphs(predictor, model_info, unicode_mapping, image, confidence_threshold):
    """Run hieroglyph detection on an image"""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
    
    # Update confidence threshold
    predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    
    # Make prediction
    outputs = predictor(img_cv)
    instances = outputs['instances'].to('cpu')
    
    # Extract predictions
    detections = []
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        for i in range(len(instances)):
            x1, y1, x2, y2 = boxes[i]
            width = x2 - x1
            height = y2 - y1
            
            # Get class information
            class_idx = int(classes[i])
            gardiner_code = model_info['category_names'][class_idx] if class_idx < len(model_info['category_names']) else f'Unknown_{class_idx}'
            
            # Get Unicode information
            unicode_info = unicode_mapping.get(gardiner_code, {
                'unicode_codes': [],
                'description': 'No description available',
                'unicode_symbol': ''
            })
            
            if gardiner_code == "X1":
                continue
            
            detection = {
                'id': i + 1,
                'gardiner_code': gardiner_code,
                'confidence': float(scores[i]),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(width),
                    'height': float(height),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2)
                },
                'unicode_codes': unicode_info['unicode_codes'],
                'unicode_symbol': unicode_info['unicode_symbol'],
                'description': unicode_info['description'],
                'area': float(width * height)
            }
            
            detections.append(detection)
    
    result = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_info': {
            'confidence_threshold': float(predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST),
            'total_classes': len(model_info['category_names'])
        },
        'detections': sorted(detections, key=lambda x: x['confidence'], reverse=True),
        'summary': {
            'total_detections': len(detections),
            'unique_classes': len(set([d['gardiner_code'] for d in detections])),
            'confidence_stats': {
                'mean': float(np.mean(scores)) if len(scores) > 0 else 0.0,
                'max': float(np.max(scores)) if len(scores) > 0 else 0.0,
                'min': float(np.min(scores)) if len(scores) > 0 else 0.0,
                'std': float(np.std(scores)) if len(scores) > 0 else 0.0
            },
            'high_confidence_count': sum(1 for d in detections if d['confidence'] >= 0.8),
            'medium_confidence_count': sum(1 for d in detections if 0.6 <= d['confidence'] < 0.8),
            'low_confidence_count': sum(1 for d in detections if d['confidence'] < 0.6)
        }
    }
    
    return result

def create_visualization(image, results):
    """Create visualization with bounding boxes"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)
    
    # Color mapping for confidence levels
    def get_confidence_color(confidence):
        if confidence >= 0.8:
            return 'lime'
        elif confidence >= 0.6:
            return 'cyan'
        else:
            return 'orange'
    
    # Draw detections
    for detection in results['detections']:
        bbox = detection['bbox']
        confidence = detection['confidence']
        gardiner_code = detection['gardiner_code']
        
        # Draw bounding box
        color = get_confidence_color(confidence)
        rect = Rectangle((bbox['x1'], bbox['y1']), bbox['width'], bbox['height'],
                       linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
        ax.add_patch(rect)
        
        # Create label with Gardiner code and confidence
        label = f"{gardiner_code} {confidence:.3f}"
        
        # Add text label with visibility
        ax.text(bbox['x1'], bbox['y1'] - 15, label, 
               color='white', fontsize=11, weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='white'))
    
    # Add legend
    if results['detections']:
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='lime', alpha=0.7, label=f"High confidence (≥0.8): {results['summary']['high_confidence_count']}"),
            plt.Rectangle((0,0),1,1, facecolor='cyan', alpha=0.7, label=f"Medium confidence (0.6-0.8): {results['summary']['medium_confidence_count']}"),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label=f"Low confidence (<0.6): {results['summary']['low_confidence_count']}")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set title
    title = f"Hieroglyph Detection Results\n"
    title += f"{results['summary']['total_detections']} detections • "
    title += f"{results['summary']['unique_classes']} unique classes • "
    title += f"threshold: {results['model_info']['confidence_threshold']:.2f}"
    
    ax.set_title(title, fontsize=16, pad=20, color='darkblue', weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_paleography_zip(crops_data, catalog_file):
    """Create a ZIP file containing all crops and the HTML catalog"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the HTML catalog
            zipf.write(catalog_file, 'digital_paleography.html')
            
            # Group crops by Gardiner code and add to ZIP
            for crop in crops_data:
                crop_path = crop['crop_path']
                if os.path.exists(crop_path):
                    # Create archive path with Gardiner code folder structure
                    archive_path = f"crops/{crop['gardiner_code']}/{crop['crop_filename']}"
                    zipf.write(crop_path, archive_path)
            
            # Add a README
            readme_content = f"""# Digital Hieroglyph Paleography

This archive contains:

## Files:
- `digital_paleography.html` - Interactive HTML catalog
- `crops/` - Directory containing cropped hieroglyph images organized by Gardiner codes

## Statistics:
- Total signs: {len(crops_data)}
- Unique Gardiner codes: {len(set(crop['gardiner_code'] for crop in crops_data))}
- Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}

## Filename Format:
Each crop is named with the format:
`[SOURCE]_[GARDINER]_[UNICODE]_[INDEX]_conf[CONFIDENCE].png`

Where:
- SOURCE: Original image filename
- GARDINER: Gardiner code (e.g., A1, G17, M17)
- UNICODE: Unicode code (e.g., U13000, U1317F) when available
- INDEX: Sequential number for multiple instances
- CONFIDENCE: Detection confidence score (0.00-1.00)

## Usage:
1. Open `digital_paleography.html` in a web browser to view the interactive catalog
2. Browse the `crops/` directory to access individual hieroglyph images organized by Gardiner code
3. Use Unicode codes to identify specific hieroglyphic variants

Generated by PapyrusVision Digital Paleography Tool
"""
            zipf.writestr('README.md', readme_content)
        
        return tmp_zip.name

def main():
    # Navigation
    st.sidebar.title("PapyrusVision Navigation")
    page = st.sidebar.radio(
        "Choose a tool:",
        ["Hieroglyph Detection", "Digital Paleography", "About"]
    )
    
    if page == "Hieroglyph Detection":
        show_detection_page()
    elif page == "Digital Paleography":
        show_paleography_page()
    elif page == "About":
        show_about_page()

def show_detection_page():
    """Main hieroglyph detection interface"""
    # Header
    st.markdown('<h1 class="main-header">PapyrusVision Hieroglyph Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive AI-powered analysis of ancient Egyptian hieroglyphs</p>', unsafe_allow_html=True)
    
    # Sidebar for model loading and settings
    with st.sidebar:
        st.header("Detection Settings")
        
        # Load model
        with st.spinner("Loading Detectron2 model..."):
            predictor, model_info, unicode_mapping = load_detectron_model()
        
        if predictor is None:
            st.error("Failed to load model. Please check your setup.")
            st.stop()
        
        st.success("Model loaded successfully!")
        
        # Model information
        with st.expander("Model Information"):
            st.write(f"**Model**: {os.path.basename(sorted(glob.glob(os.path.join(MODELS_DIR, 'hieroglyph_model_*')))[-1])}")
            st.write(f"**Classes**: {len(model_info['category_names'])}")
            st.write(f"**Default threshold**: {model_info['detection_threshold']}")
            st.write("**Image Processing**: Uploaded images are automatically resized to match training data dimensions (min: 800px, max: 1333px) for optimal detection performance.")
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=model_info['detection_threshold'],
            step=0.05,
            help="Adjust the minimum confidence score for detections"
        )
        
        st.markdown("---")
        st.markdown("### Color Legend")
        st.markdown("**🟢 High confidence** (≥0.8)")
        st.markdown("**🔵 Medium confidence** (0.6-0.8)")
        st.markdown("**🟡 Low confidence** (<0.6)")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Drag and drop your papyrus image here",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            help="Upload an image of papyrus with hieroglyphs for analysis"
        )
        
        # Default image option
        if not uploaded_file:
            use_default = st.checkbox("Use default test image", value=True)
            if use_default:
                default_image_path = os.path.join(DATA_DIR, 'images', '145_upscaled_bright.jpg')
                if os.path.exists(default_image_path):
                    uploaded_file = default_image_path
                    st.info("Using default test image: 145_upscaled_bright.jpg")
    
    with col2:
        st.header("Analysis Results")
        
        if uploaded_file:
            # Load and display image
            if isinstance(uploaded_file, str):  # Default image path
                image = Image.open(uploaded_file)
                image_name = os.path.basename(uploaded_file)
            else:  # Uploaded file
                image = Image.open(uploaded_file)
                image_name = uploaded_file.name
            
            # Get original dimensions
            original_width, original_height = image.size
            st.info(f"Original image size: {original_width}x{original_height}")
            
            # Resize image for consistent model performance
            resized_image = resize_image_for_model(image)
            new_width, new_height = resized_image.size
            
            if (new_width, new_height) != (original_width, original_height):
                st.info(f"Resized for model consistency: {new_width}x{new_height}")
            
            # Run prediction on resized image
            with st.spinner("Analyzing hieroglyphs..."):
                results = predict_hieroglyphs(predictor, model_info, unicode_mapping, resized_image, confidence_threshold)
            
            # Display results summary
            if results['summary']['total_detections'] > 0:
                st.success(f"Found {results['summary']['total_detections']} hieroglyphs!")
                
                # Metrics in columns
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("Total Detections", results['summary']['total_detections'])
                with met_col2:
                    st.metric("Unique Classes", results['summary']['unique_classes'])
                with met_col3:
                    st.metric("Max Confidence", f"{results['summary']['confidence_stats']['max']:.3f}")
                with met_col4:
                    st.metric("Mean Confidence", f"{results['summary']['confidence_stats']['mean']:.3f}")
                
            else:
                st.warning("No hieroglyphs detected. Try lowering the confidence threshold.")
    
    # Visualization section
    if uploaded_file and 'results' in locals() and results['summary']['total_detections'] > 0:
        st.header("Detection Visualization")
        
        # Create and display visualization using the resized image
        fig = create_visualization(resized_image, results)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        # Detection details
        st.header("Detection Details")
        
        # Create DataFrame for display
        detection_data = []
        for detection in results['detections']:
            # Get only the first valid Unicode code
            valid_unicode_codes = [code for code in detection['unicode_codes'] if code.startswith('U+')]
            primary_unicode_code = valid_unicode_codes[0] if valid_unicode_codes else '—'
            
            detection_data.append({
                'ID': detection['id'],
                'Gardiner Code': detection['gardiner_code'],
                'Unicode Code': primary_unicode_code if primary_unicode_code != '—' else 'N/A',
                'Confidence': f"{detection['confidence']:.3f}",
                'Center (x, y)': f"({detection['bbox']['center_x']:.0f}, {detection['bbox']['center_y']:.0f})",
                'Size (W×H)': f"{detection['bbox']['width']:.0f}×{detection['bbox']['height']:.0f}"
            })
        
        df = pd.DataFrame(detection_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export options
        st.header("Export Results")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # JSON export
            json_data = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"hieroglyph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with export_col2:
            # CSV export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"hieroglyph_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_paleography_page():
    """Digital Paleography Tool interface"""
    # Header
    st.markdown('<h1 class="main-header">Digital Hieroglyph Paleography</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create comprehensive catalogs of hieroglyphic signs with cropping and Unicode mapping</p>', unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("""
    <div class="paleography-card">
        <h2>Create Your Digital Paleography</h2>
        <p>Transform your hieroglyph images into an interactive catalog with individual sign crops, Unicode mappings, and detailed descriptions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>Automatic Cropping</h4>
            <p>Individual hieroglyph signs are automatically cropped from your images with proper padding and organization by Gardiner codes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>Interactive Catalog</h4>
            <p>HTML catalog with Unicode symbols, descriptions, confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>Complete Export</h4>
            <p>Download everything as a ZIP file including crops, HTML catalog, and detailed reports for offline use.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Settings sidebar
    with st.sidebar:
        st.header("Paleography Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for including signs in the paleography"
        )
        
        include_low_confidence = st.checkbox(
            "Include Low Confidence Signs",
            value=False,
            help="Include signs with confidence below 0.6 (may include false positives)"
        )
        
        min_size = st.slider(
            "Minimum Sign Size (pixels)",
            min_value=10,
            max_value=100,
            value=20,
            help="Minimum width or height for including signs"
        )
        
        st.info("📏 **Image Processing**: Uploaded images are automatically resized to match training data dimensions for consistent detection performance.")
    
    # Main interface
    st.header("Upload Images for Paleography")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload multiple hieroglyph images",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select multiple images to create a comprehensive paleography catalog"
    )
    
    # Default image option for paleography
    if not uploaded_files:
        use_default_paleo = st.checkbox("Use default test image for paleography demonstration", value=True)
        if use_default_paleo:
            default_image_path = os.path.join(DATA_DIR, 'images', '145_upscaled_bright.jpg')
            if os.path.exists(default_image_path):
                # Create a mock uploaded file list with the default image
                class MockFile:
                    def __init__(self, path):
                        self.name = '145_upscaled_bright.jpg'
                        self.path = path
                    
                    def read(self):
                        with open(self.path, 'rb') as f:
                            return f.read()
                
                uploaded_files = [MockFile(default_image_path)]
                st.info("Using default test image: 145_upscaled_bright.jpg")
                st.info("This will demonstrate the paleography tool with a sample hieroglyphic text.")
    
    if uploaded_files:
        if hasattr(uploaded_files[0], 'name') and uploaded_files[0].name == '145_upscaled_bright.jpg':
            st.success(f"Using default test image for demonstration")
        else:
            st.success(f"Loaded {len(uploaded_files)} images")
        
        # Process button
        if st.button("Create Digital Paleography", type="primary"):
            # Initialize the paleography tool
            with st.spinner("Initializing Digital Paleography Tool..."):
                try:
                    paleography_tool = DigitalPaleographyTool()
                    
                    if not paleography_tool.analyzer:
                        st.error("Could not load the hieroglyph analyzer. Please check your model setup.")
                        st.stop()
                    
                    st.success("Paleography tool initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Error initializing paleography tool: {e}")
                    st.stop()
            
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files with resizing for consistent processing
                file_paths = []
                for uploaded_file in uploaded_files:
                    # Load the original image
                    original_image = Image.open(uploaded_file)
                    original_width, original_height = original_image.size
                    
                    # Resize for model consistency
                    resized_image = resize_image_for_model(original_image)
                    new_width, new_height = resized_image.size
                    
                    # Save the resized image with proper file extension
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    file_path = os.path.join(temp_dir, f"{base_name}.jpg")
                    resized_image.save(file_path, format='JPEG', quality=95)
                    file_paths.append(file_path)
                    
                    # Show resize info if dimensions changed
                    if (new_width, new_height) != (original_width, original_height):
                        st.info(f"{uploaded_file.name}: Resized from {original_width}x{original_height} to {new_width}x{new_height} for consistent processing")
                
                # Process all images
                all_crops_data = []
                progress_bar = st.progress(0)
                
                for i, file_path in enumerate(file_paths):
                    st.info(f"Processing {os.path.basename(file_path)}...")
                    
                    try:
                        crops_data = paleography_tool.process_image(file_path, confidence_threshold)
                        
                        # Filter by settings
                        filtered_crops = []
                        for crop in crops_data:
                            if not include_low_confidence and crop['confidence'] < 0.6:
                                continue
                            
                            if crop['dimensions'][0] < min_size or crop['dimensions'][1] < min_size:
                                continue
                                
                            filtered_crops.append(crop)
                        
                        all_crops_data.extend(filtered_crops)
                        progress_bar.progress((i + 1) / len(file_paths))
                        
                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(file_path)}: {e}")
                        continue
                
                if all_crops_data:
                    st.success(f"Generated {len(all_crops_data)} sign crops from {len(file_paths)} images")
                    
                    # Create HTML catalog
                    with st.spinner("Creating interactive HTML catalog..."):
                        catalog_file = paleography_tool.create_html_catalog(all_crops_data)
                    
                    # Generate report
                    with st.spinner("Generating detailed report..."):
                        report_file = paleography_tool.generate_report(all_crops_data)
                    
                    # Show statistics
                    st.header("Paleography Statistics")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("Total Signs", len(all_crops_data))
                    
                    with stat_col2:
                        unique_codes = len(set(crop['gardiner_code'] for crop in all_crops_data))
                        st.metric("Unique Gardiner Codes", unique_codes)
                    
                    with stat_col3:
                        avg_confidence = np.mean([crop['confidence'] for crop in all_crops_data])
                        st.metric("Average Confidence", f"{avg_confidence:.3f}")
                    
                    with stat_col4:
                        source_images = len(set(crop['source_image'] for crop in all_crops_data))
                        st.metric("Source Images", source_images)
                    
                    # Show top Gardiner codes
                    st.header("Most Common Signs")
                    
                    # Count occurrences
                    code_counts = defaultdict(int)
                    for crop in all_crops_data:
                        code_counts[crop['gardiner_code']] += 1
                    
                    # Create bar chart
                    top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    chart_data = pd.DataFrame(top_codes, columns=['Gardiner Code', 'Count'])
                    st.bar_chart(chart_data.set_index('Gardiner Code'))
                    
                    # Preview some crops
                    st.header("Sample Crops Preview")
                    
                    # Show first few crops as preview
                    preview_cols = st.columns(5)
                    for i, crop in enumerate(all_crops_data[:5]):
                        with preview_cols[i]:
                            try:
                                crop_image = Image.open(crop['crop_path'])
                                st.image(crop_image, caption=f"{crop['gardiner_code']} ({crop['confidence']:.2f})")
                            except Exception as e:
                                st.error(f"Could not display crop: {e}")
                    
                    # Download options
                    st.header("Download Your Digital Paleography")
                    
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        # HTML catalog download
                        with open(catalog_file, 'rb') as f:
                            catalog_data = f.read()
                        
                        st.download_button(
                            label="Download HTML Catalog",
                            data=catalog_data,
                            file_name="digital_paleography.html",
                            mime="text/html",
                            help="Interactive HTML catalog with embedded images"
                        )
                    
                    with download_col2:
                        # ZIP package download
                        with st.spinner("Creating ZIP package..."):
                            zip_file = create_paleography_zip(all_crops_data, catalog_file)
                        
                        with open(zip_file, 'rb') as f:
                            zip_data = f.read()
                        
                        st.download_button(
                            label="Download Complete Package (ZIP)",
                            data=zip_data,
                            file_name=f"digital_paleography_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            help="Complete package with HTML catalog, all crop images, and documentation"
                        )
                    
                    st.success("Digital Paleography creation complete!")
                    
                else:
                    st.warning("No sign crops were generated. Try lowering the confidence threshold or adjusting the minimum size.")
    
    else:
        st.info("Please upload one or more images to create your digital paleography catalog.")

def show_about_page():
    """About page with information and instructions"""
    st.markdown('<h1 class="main-header">About PapyrusVision</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## About This Tool
    
    **PapyrusVision** is an AI-powered system for analyzing ancient Egyptian hieroglyphs in papyrus documents. 
    It combines computer vision with Egyptological knowledge to provide papyrus analysis.
    
    **Training Data**: The model was trained on 2,431 manually annotated hieroglyphs from the Book of the Dead of Nu (British Museum EA 10477), 
    covering 177 distinct Gardiner sign categories. This 18th Dynasty papyrus provides examples of classical Egyptian hieroglyphic writing, 
    ensuring the model learned from high-quality sources.
    
    ### Key Features
    
    #### Hieroglyph Detection
    - Real-time AI-powered detection using Detectron2
    - Confidence-based filtering and visualization
    - Interactive bounding boxes with color-coded confidence levels
    - Export capabilities for further analysis
    
    #### Digital Paleography
    - Automatic cropping of individual hieroglyphic signs
    - Organization by Gardiner codes with full descriptions
    - Interactive HTML catalog generation
    - Unicode mapping integration (594 official mappings)
    - Batch processing of multiple images
    - Complete ZIP package export for offline use
    
    ### Technology Stack
    
    - **AI Model**: Detectron2 (Facebook AI Research)
    - **Web Interface**: Streamlit
    - **Computer Vision**: OpenCV, PIL
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Matplotlib
    - **Unicode Standards**: Official Egyptian Hieroglyphs Unicode block
    
    ### Data Sources
    
    - **Gardiner Sign List**: Complete classification system
    - **Unicode Mappings**: Official Unicode Consortium mappings
    
    ### Getting Started
    
    1. **Single Image Analysis**: Use the "Hieroglyph Detection" page to analyze individual images
    2. **Batch Processing**: Use the "Digital Paleography" page to process multiple images and create catalogs
    3. **Adjust Settings**: Use the sidebar controls to fine-tune detection parameters
    4. **Export Results**: Download your results in various formats (JSON, CSV, HTML, ZIP)
    
    ### Tips for Best Results
    
    - Use high-quality, well-lit images
    - Adjust confidence threshold based on your needs
    - Lower confidence thresholds may include more signs but also more false positives
    """)

if __name__ == "__main__":
    main()
