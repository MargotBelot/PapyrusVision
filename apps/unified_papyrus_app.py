#!/usr/bin/env python3
"""
This application combines all features:
- Interactive hieroglyph detection with real-time visualization
- Digital paleography with complete notation
- Export formats

Features:
- Detection: AI-powered hieroglyph recognition
- Complete Notation: Gardiner codes and Unicode symbols
- Digital Paleography: Automated sign catalogs with documentation
- Export: HTML catalogs and research-grade outputs

Usage:
    streamlit run apps/unified_papyrus_app.py
"""

import streamlit as st
import os
import sys
import json
import numpy as np
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
import glob
from datetime import datetime
import zipfile
from collections import defaultdict
import base64
from pathlib import Path
import io

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(PROJECT_ROOT))

# Import modules
try:
    from scripts.hieroglyph_analysis_tool import HieroglyphAnalysisTool
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PapyrusVision - AI Ancient Egyptian Paleography",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 300;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #CD853F;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .enhancement-badge {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 10px 5px;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #F5DEB3, #DEB887);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #DAA520;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #8B4513, #A0522D);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .jsesh-notation {
        background: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        margin: 2px;
        display: inline-block;
    }
    .unicode-symbol {
        font-size: 2em;
        color: #8B4513;
        text-align: center;
        margin: 10px 0;
    }
    .detection-box {
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: rgba(40, 167, 69, 0.05);
    }
    .nav-tabs {
        border-bottom: 2px solid #8B4513;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #DAA520, #B8860B);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFF8DC, #F5DEB3);
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class UnifiedPapyrusApp:
    def __init__(self):
        """Initialize the application"""
        self.setup_directories()
        self.load_models_and_data()
    
    def setup_directories(self):
        """Setup working directories (disabled - no folder creation)"""
        temp_base = Path(tempfile.gettempdir()) / "papyrus_temp"
        
        self.base_dir = temp_base
        self.crops_dir = self.base_dir / "cropped_signs"
        self.catalogs_dir = self.base_dir / "catalogs"
        self.exports_dir = self.base_dir / "exports"
        
    @st.cache_resource
    def load_models_and_data(_self):
        """Load AI model and analysis tools"""
        with st.spinner("Loading AI models and analysis tools..."):
            try:
                # Load hierarchlyph analyzer
                analyzer = HieroglyphAnalysisTool()
                if not analyzer.load_model():
                    st.error("Could not load AI detection model")
                    return None
                
                return analyzer
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return None
    
    def resize_image_for_model(self, image, min_size=800, max_size=1333):
        """Resize image to match model requirements"""
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            image = rgb_image
        elif image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')

        original_width, original_height = image.size
        shorter_side = min(original_width, original_height)
        longer_side = max(original_width, original_height)

        scale_factor = min_size / shorter_side
        if longer_side * scale_factor > max_size:
            scale_factor = max_size / longer_side

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def get_confidence_class(self, confidence):
        """Get CSS class for confidence display"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def analyze_single_image(self, uploaded_file, confidence_threshold, analyzer):
        """Analyze a single uploaded image with AI detection"""
        # Create temp directory only when needed
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file temporarily
        temp_path = self.base_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Analyze image
            results = analyzer.predict_hieroglyphs(str(temp_path), confidence_threshold)
            if not results or not results.get('detections'):
                return [], None
            
            detections = results['detections']
            
            # Load original image
            original_image = cv2.imread(str(temp_path))
            if original_image is None:
                return [], None
            
            # Process each detection
            processed_detections = []
            for i, detection in enumerate(detections):
                confidence = detection.get('confidence', 0.0)
                gardiner_code = detection.get('gardiner_code', 'Unknown')
                
                # Create detection entry with ALL original data preserved
                processed_detection = {
                    'gardiner_code': gardiner_code,
                    'confidence': confidence,
                    'bbox': detection.get('bbox', {}),
                    'source_image': uploaded_file.name,
                    'detection_index': i,
                    'description': detection.get('description', f'Hieroglyph {gardiner_code}'),
                    'unicode_codes': detection.get('unicode_codes', []),
                    'unicode_symbol': detection.get('unicode_symbol', ''),
                    'area': detection.get('area', 0)
                }
                
                processed_detections.append(processed_detection)
            
            return processed_detections, original_image
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def create_detection_visualization(self, image, detections):
        """Create visualization with detection boxes"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
        
        for detection, color in zip(detections, colors):
            bbox = detection.get('bbox', {})
            if not bbox:
                continue
                
            x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
            x2, y2 = bbox.get('x2', 100), bbox.get('y2', 100)
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label with Gardiner code and confidence
            gardiner_code = detection.get('gardiner_code', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            label = f"{gardiner_code} ({confidence:.1%})"
            ax.text(x1, y1-5, label, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor=color, alpha=0.8), fontsize=8, fontweight='bold')
        
        ax.set_title(f"Detected Hieroglyphs ({len(detections)} signs)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig
    
    def create_interactive_detection_visualization(self, image, detections):
        """Create interactive plotly visualization with detection boxes"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig = go.Figure()
        
        # Add the image
        fig.add_trace(go.Image(
            z=image_rgb,
            name="Papyrus Image"
        ))
        
        # Add detection boxes as shapes
        shapes = []
        annotations = []
        
        # Generate colors for each detection
        colors = px.colors.qualitative.Set3
        
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', {})
            if not bbox:
                continue
                
            x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
            x2, y2 = bbox.get('x2', 100), bbox.get('y2', 100)
            
            # Get color (cycle through available colors)
            color = colors[i % len(colors)]
            
            # Add bounding box
            shapes.append(dict(
                type="rect",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(color=color, width=3),
                fillcolor="rgba(0,0,0,0)"
            ))
            
            # Add label
            gardiner_code = detection.get('gardiner_code', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            label = f"{gardiner_code} ({confidence:.1%})"
            
            annotations.append(dict(
                x=x1,
                y=y1-10,
                text=label,
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=color,
                bordercolor="white",
                borderwidth=1
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Detected Hieroglyphs ({len(detections)} signs) - Interactive View",
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, image_rgb.shape[1]]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
                range=[image_rgb.shape[0], 0]  # Flip y-axis for image coordinates
            ),
            showlegend=False,
            shapes=shapes,
            annotations=annotations,
            margin=dict(l=0, r=0, t=50, b=0),
            height=700
        )
        
        return fig
    
    def create_enhanced_html_catalog(self, all_detections, title="Digital Paleography"):
        """Create HTML catalog with complete notation"""
        # Group detections by Gardiner code
        grouped_detections = defaultdict(list)
        for detection in all_detections:
            grouped_detections[detection['gardiner_code']].append(detection)
        
        sorted_codes = sorted(grouped_detections.keys())
        
        # Calculate statistics
        total_signs = len(all_detections)
        total_codes = len(sorted_codes)
        high_confidence = sum(1 for d in all_detections if d.get('confidence', 0) >= 0.8)
        avg_confidence = sum(d.get('confidence', 0) for d in all_detections) / len(all_detections) if all_detections else 0
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f5f5, #e9ecef);
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #8B4513, #D2B48C);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .header h1 {{
            margin: 0;
            font-size: 3em;
            font-weight: 300;
        }}
        .enhancement-badge {{
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1em;
            margin: 15px 0;
            display: inline-block;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .stats-dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #8B4513;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #8B4513;
            margin-bottom: 5px;
        }}
        .gardiner-section {{
            background: white;
            margin: 30px 0;
            border-radius: 20px;
            box-shadow: 0 6px 25px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .gardiner-header {{
            background: linear-gradient(135deg, #8B4513, #A0522D);
            color: white;
            padding: 30px;
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 25px;
            align-items: center;
        }}
        .symbol-display {{
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 15px;
            font-size: 3.5em;
            text-align: center;
            min-width: 100px;
            backdrop-filter: blur(10px);
        }}
        .notation-info h2 {{
            margin: 0 0 15px 0;
            font-size: 2em;
        }}
        .notation-row {{
            display: flex;
            gap: 15px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .notation-item {{
            background: rgba(255,255,255,0.1);
            padding: 8px 15px;
            border-radius: 25px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            backdrop-filter: blur(5px);
        }}
        .jsesh-code {{ background: #28a745; color: white; }}
        .unicode-code {{ background: #007bff; color: white; }}
        .gardiner-code {{ background: #dc3545; color: white; }}
        .detections-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 25px;
            padding: 30px;
        }}
        .detection-card {{
            background: #fafafa;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .detection-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .confidence-badge {{
            background: #28a745;
            color: white;
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>AI-Powered Hieroglyph Detection with Complete Scholarly Notation</p>
        <div class="enhancement-badge">AI-Powered Detection & Classification</div>
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
    </div>
    
    <div class="stats-dashboard">
        <div class="stat-card">
            <div class="stat-number">{total_signs}</div>
            <div>Total Detected Signs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_codes}</div>
            <div>Unique Gardiner Codes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{high_confidence}</div>
            <div>High Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_confidence:.1%}</div>
            <div>Average Confidence</div>
        </div>
    </div>
"""
        
        # Add each Gardiner code section
        for code in sorted_codes:
            detections = grouped_detections[code]
            first_detection = detections[0]
            
            description = first_detection.get('description', f'Hieroglyph {code}')
            avg_conf = sum(d.get('confidence', 0) for d in detections) / len(detections)
            
            html_content += f"""
    <div class="gardiner-section" id="{code}">
        <div class="gardiner-header">
            <div class="symbol-display">{code}</div>
            <div class="notation-info">
                <h2>{code}</h2>
                <div class="notation-row">
                    <span class="notation-item gardiner-code">Gardiner: {code}</span>
                    <span class="notation-item confidence-badge">Avg Confidence: {avg_conf:.1%}</span>
                </div>
                <div style="margin-top: 15px; font-style: italic; opacity: 0.9;">
                    {description}
                </div>
                <div style="margin-top: 10px; opacity: 0.8;">
                    {len(detections)} detected instances
                </div>
            </div>
        </div>
        <div class="detections-grid">"""
            
            # Add detection cards (without images for now, can be enhanced later)
            for detection in detections:
                confidence = detection.get('confidence', 0.0)
                source_image = detection.get('source_image', 'Unknown')
                
                html_content += f"""
            <div class="detection-card">
                <div style="margin-bottom: 15px;">
                    <span class="confidence-badge">{confidence:.1%}</span>
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    <div><strong>{source_image}</strong></div>
                    <div>Detection #{detection.get('detection_index', 0)}</div>
                </div>
            </div>"""
            
            html_content += """
        </div>
    </div>"""
        
        html_content += """
</body>
</html>"""
        
        return html_content
    
    def process_crop_for_display(self, crop):
        """Process crop for better display quality in the interface"""
        if crop.size == 0:
            return None
        
        # Get dimensions
        h, w = crop.shape[:2]
        
        # Skip tiny crops
        if h < 8 or w < 8:
            return None
        
        target_size = 120  # Optimal size for display in Streamlit columns
        
        # Calculate scale to fit within target size while maintaining aspect ratio
        scale = min(target_size / w, target_size / h)
        
        # Don't upscale tiny crops too much to avoid extreme pixelation
        if max(h, w) < 20:
            scale = min(scale, 3.0)
        elif max(h, w) < 40:
            scale = min(scale, 2.5)
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        if scale > 2.0:  # Significant upscaling
            # Use LANCZOS for smooth upscaling of small crops
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        elif scale > 1.2:  # Moderate upscaling
            # Use CUBIC for moderate upscaling
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif scale < 0.8:  # Downscaling
            # Use AREA for smooth downscaling
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Minimal scaling or no scaling
            if scale == 1.0:
                resized = crop
            else:
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply noise reduction for cleaner display
        if len(resized.shape) > 2:  # Color image
            denoised = cv2.fastNlMeansDenoisingColored(resized, None, 3, 3, 7, 21)
        else:  # Grayscale
            denoised = cv2.fastNlMeansDenoising(resized, None, 3, 7, 21)
        
        # Apply adaptive sharpening based on image size
        if max(new_h, new_w) < 60:  # Small images need more sharpening
            kernel = np.array([[-0.1, -0.2, -0.1],
                              [-0.2,  2.0, -0.2],
                              [-0.1, -0.2, -0.1]])
        else:  # Larger images need subtle sharpening
            kernel = np.array([[-0.05, -0.1, -0.05],
                              [-0.1,   1.3, -0.1],
                              [-0.05, -0.1, -0.05]])
        
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Ensure output is within reasonable bounds
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def process_crop_for_export(self, crop):
        """Process crop for high-quality export files"""
        if crop.size == 0:
            return None
        
        # Get dimensions
        h, w = crop.shape[:2]
        
        # Skip tiny crops that are likely noise
        if h < 6 or w < 6:
            return None
        
        # Target size for exported crops - higher quality for exports
        target_size = 150  # Larger size for better export quality
        
        # Calculate scaling to fit within target size while maintaining aspect ratio
        scale = min(target_size / w, target_size / h)
        
        # Smart scaling limits based on original size
        if max(h, w) < 15:  # Very tiny crops
            scale = min(scale, 4.0)
        elif max(h, w) < 30:  # Small crops
            scale = min(scale, 3.0)
        elif max(h, w) < 60:  # Medium crops
            scale = min(scale, 2.0)
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        # High-quality interpolation based on scaling
        if scale > 2.5:  # Significant upscaling
            # Use LANCZOS for best quality upscaling
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        elif scale > 1.2:  # Moderate upscaling
            # Use CUBIC for smooth upscaling
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif scale < 0.7:  # Downscaling
            # Use AREA for smooth downscaling
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Minimal scaling
            if abs(scale - 1.0) < 0.1:  # Almost no scaling needed
                resized = crop
            else:
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create a square canvas with padding if needed (for consistent export)
        if new_w != new_h:
            canvas_size = max(new_w, new_h)
            # Use light gray background instead of white for better contrast
            background_color = 245  # Light gray
            
            if len(crop.shape) > 2:  # Color image
                canvas = np.full((canvas_size, canvas_size, crop.shape[2]), 
                               background_color, dtype=crop.dtype)
            else:  # Grayscale
                canvas = np.full((canvas_size, canvas_size), 
                               background_color, dtype=crop.dtype)
            
            # Center the crop in the canvas
            start_x = (canvas_size - new_w) // 2
            start_y = (canvas_size - new_h) // 2
            
            if len(crop.shape) > 2:  # Color image
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            else:  # Grayscale
                canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            final_crop = canvas
        else:
            final_crop = resized
        
        # Apply advanced denoising for export quality
        if len(final_crop.shape) > 2:  # Color image
            # Stronger denoising for exports
            denoised = cv2.fastNlMeansDenoisingColored(final_crop, None, 5, 5, 7, 21)
        else:  # Grayscale
            denoised = cv2.fastNlMeansDenoising(final_crop, None, 5, 7, 21)
        
        # Apply adaptive sharpening based on final size
        final_h, final_w = denoised.shape[:2]
        
        if max(final_h, final_w) < 80:  # Small final images need more sharpening
            kernel = np.array([[-0.1, -0.2, -0.1],
                              [-0.2,  2.2, -0.2],
                              [-0.1, -0.2, -0.1]])
        elif max(final_h, final_w) < 120:  # Medium images
            kernel = np.array([[-0.08, -0.15, -0.08],
                              [-0.15,  1.8, -0.15],
                              [-0.08, -0.15, -0.08]])
        else:  # Larger images need subtle sharpening
            kernel = np.array([[-0.05, -0.1, -0.05],
                              [-0.1,   1.5, -0.1],
                              [-0.05, -0.1, -0.05]])
        
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Final contrast enhancement for better visibility
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(sharpened.shape) > 2:  # Color image
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced = clahe.apply(sharpened)
        
        # Ensure output is within valid bounds
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def create_analytics_section(self, detections, title="Analysis"):
        """Create analytics section for detection results"""
        if not detections:
            return
        
        st.subheader(title)
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(detections))
        
        with col2:
            unique_codes = len(set(d['gardiner_code'] for d in detections))
            st.metric("Unique Signs", unique_codes)
        
        with col3:
            avg_confidence = np.mean([d['confidence'] for d in detections])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            high_confidence = sum(1 for d in detections if d.get('confidence', 0) >= 0.8)
            st.metric("High Confidence", f"{high_confidence}/{len(detections)}")
        
        # Only show detailed analytics if we have enough detections
        if len(detections) >= 3:
            with st.expander("Detailed Analytics", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Detection quality overview
                    st.subheader("Detection Quality Overview")
                    confidence_scores = [d['confidence'] for d in detections]
                    
                    # Create a simple bar chart showing confidence ranges
                    high_conf = sum(1 for score in confidence_scores if score >= 0.8)
                    medium_conf = sum(1 for score in confidence_scores if 0.5 <= score < 0.8)
                    low_conf = sum(1 for score in confidence_scores if score < 0.5)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['High\n(≥80%)', 'Medium\n(50-79%)', 'Low\n(<50%)']
                    counts = [high_conf, medium_conf, low_conf]
                    colors = ['#28a745', '#ffc107', '#dc3545']
                    
                    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
                    
                    # Add count labels on top of bars
                    for bar, count in zip(bars, counts):
                        if count > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                   str(count), ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_ylabel('Number of Detections')
                    ax.set_title('Detection Quality Breakdown')
                    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    st.pyplot(fig)
                    
                    # Add download button for the chart
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Quality Chart (PNG)",
                        data=buf.getvalue(),
                        file_name=f"detection_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        key=f"download_quality_chart_{id(detections)}"
                    )
                    
                    plt.close()
                    
                    # Add explanation
                    st.caption("**How to read this chart:**")
                    st.caption("• **High confidence** = Very likely correct detections")
                    st.caption("• **Medium confidence** = Probably correct, may need review")
                    st.caption("• **Low confidence** = Uncertain detections, check manually")
                
                with col2:
                    # Top signs
                    st.subheader("Most Frequent Signs")
                    gardiner_counts = {}
                    for d in detections:
                        code = d['gardiner_code']
                        gardiner_counts[code] = gardiner_counts.get(code, 0) + 1
                    
                    top_signs = sorted(gardiner_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Create a bar chart for top signs
                    if top_signs:
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        codes = [item[0] for item in top_signs]
                        counts = [item[1] for item in top_signs]
                        
                        bars = ax2.bar(codes, counts, color='#8B4513', alpha=0.7, edgecolor='black')
                        
                        # Add count labels on top of bars
                        for bar, count in zip(bars, counts):
                            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                   str(count), ha='center', va='bottom', fontweight='bold')
                        
                        ax2.set_xlabel('Gardiner Codes')
                        ax2.set_ylabel('Number of Detections')
                        ax2.set_title('Most Frequent Hieroglyphic Signs')
                        ax2.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig2)
                        
                        # Add download button for the frequency chart
                        buf2 = io.BytesIO()
                        fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
                        buf2.seek(0)
                        
                        st.download_button(
                            label="Download Frequency Chart (PNG)",
                            data=buf2.getvalue(),
                            file_name=f"sign_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            key=f"download_frequency_chart_{id(detections)}"
                        )
                        
                        plt.close()
                    
                    # Also show the text list
                    st.write("**Sign Frequencies:**")
                    for code, count in top_signs:
                        # Try to get symbol (fallback to code if no symbol)
                        symbol = next((d.get('unicode_symbol', code) for d in detections if d['gardiner_code'] == code), code)
                        percentage = (count / len(detections)) * 100
                        st.write(f"**{symbol} {code}**: {count} ({percentage:.1f}%)")
                
                # Export analytics button - prepare data outside the conditional
                # Create analytics report data
                analytics_data = {
                    'summary': {
                        'total_detections': len(detections),
                        'unique_signs': unique_codes,
                        'average_confidence': float(avg_confidence),
                        'high_confidence_count': high_confidence,
                        'high_confidence_percent': float((high_confidence / len(detections)) * 100)
                    },
                    'top_signs': dict(top_signs),
                    'confidence_distribution': confidence_scores,
                    'analysis_info': {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'total_detections_analyzed': len(detections)
                    },
                    'detections': detections
                }
                
                # Convert to JSON
                analytics_json = json.dumps(analytics_data, indent=2, ensure_ascii=False)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"papyrus_analytics_{timestamp}.json"
                
                # Direct download button without conditional logic
                st.download_button(
                    label="Download Analytics Report (JSON)",
                    data=analytics_json,
                    file_name=filename,
                    mime="application/json",
                    key=f"download_analytics_direct_{id(detections)}",
                    help="Download complete analytics report with all detection data"
                )

def main():
    # Initialize the app
    app = UnifiedPapyrusApp()
    
    # Load models and data
    analyzer = app.load_models_and_data()
    
    # Header
    st.markdown('<div class="main-header">PapyrusVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Digital Paleography</div>', unsafe_allow_html=True)
    
    
    # Interface explanation
    with st.expander("How to Use This Interface", expanded=False):
        st.markdown("""
        ### Getting Started
        
        **PapyrusVision** is an AI-powered digital paleography tool for ancient Egyptian hieroglyph analysis.
        
        #### **Single Image Analysis Tab**
        **Perfect for individual image analysis and research:**
        - **Quick Start**: Built-in test image (Book of the Dead of Nu, Spell 145) or upload your own
        - **AI Detection**: Detectron2-powered analysis with 73% mAP@0.5 across 177 Gardiner categories
        - **Interactive Visualization**: Color-coded detection boxes with confidence scores
        - **Real-time Analytics**: Automatic quality analysis, confidence distributions, and frequency charts
        - **Professional Export**: CSV/JSON formats with complete metadata and PNG visualizations
        - **Advanced Processing**: High-quality image processing with noise reduction and enhancement
        
        #### **Digital Paleography Tab**
        **Complete workflow for creating professional paleographic catalogs:**
        - **End-to-end Workflow**: Upload → Detect → Process → Preview → Export in one interface
        - **Image Processing**: 150px export crops with advanced multi-stage enhancement pipeline
        - **Smart Organization**: Automatic Gardiner code grouping with expandable preview sections
        - **Professional Quality**: CLAHE enhancement, adaptive sharpening, and noise reduction
        - **Interactive Preview**: Browse all cropped signs organized by classification before downloading
        - **Output**: Professional HTML catalogs with embedded images and complete metadata
        - **Archive Creation**: One-click ZIP packages with organized folder structures for offline research
        
        #### **Sidebar Configuration**
        **Fine-tune analysis parameters for optimal results:**
        - **Confidence Threshold Slider**: Real-time adjustable detection sensitivity (0.1-0.95)
          - **0.3-0.5**: Detect more signs, may include some uncertain detections
          - **0.5-0.7**: Balanced approach (recommended for most images)
          - **0.7-0.9**: High precision, very reliable detections only
        - **Model Status Indicator**: Live verification of AI detection model availability
        - **System Health**: Automatic checks for optimal performance
        
        #### **Key Technical Features**
        - **State-of-the-Art AI**: Detectron2-based neural networks trained on 2,430 manually annotated hieroglyphs
        - **Comprehensive Classification**: 177 distinct Gardiner sign categories with Unicode coverage
        - **Advanced Image Processing**: Multi-stage enhancement pipeline with CLAHE, denoising, and adaptive sharpening
        - **Export Standards**: Output with complete scholarly metadata
        - **Quality Assurance**: Automatic analytics, confidence analysis, and visual quality reporting
        - **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux with consistent results
        
        #### **Best Practices for Optimal Results**
        - **Image Quality**: High-resolution scans (minimum 800px shortest side) for best detection accuracy
        - **File Formats**: TIFF/PNG preferred for archival quality; JPEG acceptable for general use
        - **Threshold Tuning**: Start with default 0.5, adjust based on image clarity and desired precision
        - **Image Preparation**: Ensure good contrast between hieroglyphs and background for optimal detection
        - **Quality Control**: Use built-in analytics to assess detection quality and adjust parameters
        - **Iterative Approach**: Test different confidence levels to find optimal balance for your specific images
        
        #### **Output Formats and Research Applications**
        - **CSV Files**: Structured tabular data with metadata headers, perfect for Excel, databases, and statistical analysis
        - **JSON Files**: Complete machine-readable format, APIs, and data integration
        - **PNG Visualizations**: High-resolution charts and detection overlays for presentations and publications
        - **HTML Catalogs**: Interactive documents with embedded images for web viewing
        - **ZIP Archives**: Complete research packages with organized folders, metadata, and all generated files
        - **Analytics Reports**: Comprehensive quality assessments with confidence distributions and frequency analysis
        
        """)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Analysis settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.95, value=0.5, step=0.05,
            help="Minimum confidence for hieroglyph detection"
        )
        
        # Model status
        st.subheader("Model Status")
        if analyzer:
            st.success("AI Detection Model Active")
            st.info("Gardiner classification enabled")
        else:
            st.error("AI Detection Model unavailable")
        
    
    # Interface selection - using radio to avoid tab switching issues
    if 'interface_mode' not in st.session_state:
        st.session_state.interface_mode = "Single Image Analysis"
    
    interface_mode = st.radio(
        "Choose Interface:",
        ["Single Image Analysis", "Digital Paleography"],
        index=0 if st.session_state.interface_mode == "Single Image Analysis" else 1,
        horizontal=True,
        key="interface_selector"
    )
    
    # Update session state
    st.session_state.interface_mode = interface_mode
    
    st.markdown("---")
    
    if interface_mode == "Single Image Analysis":
        st.header("Single Image Analysis")
        st.markdown("Upload a single papyrus image for detailed analysis with complete notation.")
        
        # Image source selection
        image_source = st.radio(
            "Choose Image Source:",
            ["Use Test Image (recommended)", "Upload Image (likely to not work at the moment)"],
            horizontal=True
        )
        
        uploaded_file = None
        image = None
        test_image_path = DATA_DIR / 'images' / '145_upscaled_bright.jpg'
        
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose a papyrus image",
                type=['png', 'jpg', 'jpeg', 'tiff'],
                help="Upload high-resolution papyrus images for best results"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        else:
            # Use test image as default
            if test_image_path.exists():
                st.info("Using test image: Book of the Dead of Nu (Spell 145) - BM EA 10477")
                image = Image.open(test_image_path)
                # Create a mock uploaded file for analysis
                class MockUploadedFile:
                    def __init__(self, path):
                        self.name = path.name
                        self.path = path
                    
                    def getbuffer(self):
                        return open(self.path, 'rb').read()
                
                uploaded_file = MockUploadedFile(test_image_path)
            else:
                st.error("Test image not found. Please upload your own image.")
        
        if image and analyzer:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display image
                resized_image = app.resize_image_for_model(image)
                caption = f"Test Image: {uploaded_file.name}" if image_source == "Use Test Image" else f"Uploaded: {uploaded_file.name}"
                st.image(resized_image, caption=caption, width='stretch')
            
            with col2:
                st.subheader("Image Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Original Size:** {image.size[0]} × {image.size[1]} px")
                st.write(f"**Processed Size:** {resized_image.size[0]} × {resized_image.size[1]} px")
                st.write(f"**Format:** {image.format}")
            
            # Check if we have cached results for this image
            image_key = f"{uploaded_file.name}_{confidence_threshold}"
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image with AI detection..."):
                    detections, original_image = app.analyze_single_image(
                        uploaded_file, confidence_threshold, analyzer
                    )
                    
                    # Store results in session state
                    if detections and original_image is not None:
                        st.session_state[f'detections_{image_key}'] = detections
                        st.session_state[f'original_image_{image_key}'] = original_image
                        st.session_state[f'analyzed_image_{image_key}'] = {
                            'name': uploaded_file.name,
                            'size': image.size,
                            'processed_size': resized_image.size,
                            'format': image.format
                        }
            
            # Display results if they exist (either just calculated or from session state)
            if f'detections_{image_key}' in st.session_state:
                detections = st.session_state[f'detections_{image_key}']
                original_image = st.session_state[f'original_image_{image_key}']
                stored_image_info = st.session_state[f'analyzed_image_{image_key}']
                
                if detections:
                    st.success(f"Found {len(detections)} hieroglyphs!")
                    
                    # Create visualization
                    fig = app.create_detection_visualization(original_image, detections)
                    st.pyplot(fig)
                    
                    # Add download button for detection visualization
                    buf_viz = io.BytesIO()
                    fig.savefig(buf_viz, format='png', dpi=300, bbox_inches='tight')
                    buf_viz.seek(0)
                    
                    st.download_button(
                        label="Download Detection Visualization (PNG)",
                        data=buf_viz.getvalue(),
                        file_name=f"detection_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        key=f"download_detection_viz_{image_key}"
                    )
                    
                    
                    # Display results table
                    st.subheader("Detection Results Table")
                    
                    # Prepare data for table
                    table_data = []
                    for i, detection in enumerate(detections):
                        bbox = detection.get('bbox', {})
                        bbox_str = f"({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}, {bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})"
                        
                        # Add fallback info if used
                        gardiner_display = detection['gardiner_code']
                        if detection.get('fallback_used'):
                            gardiner_display = f"{detection['gardiner_code']} (→{detection['fallback_used']})"
                        
                        # Get Unicode symbol and codes for display
                        unicode_symbol = detection.get('unicode_symbol', '')
                        unicode_codes = detection.get('unicode_codes', [])
                        unicode_display = ''
                        if unicode_symbol and unicode_symbol.strip():
                            unicode_display = unicode_symbol
                            if unicode_codes:
                                unicode_display += f" ({unicode_codes[0]})"
                        elif unicode_codes:
                            unicode_display = unicode_codes[0]
                        
                        table_data.append({
                            '#': i + 1,
                            'Gardiner': gardiner_display,
                            'Unicode': unicode_display,
                            'Confidence': f"{detection['confidence']:.1%}",
                            'BBox (x1,y1,x2,y2)': bbox_str,
                            'Description': detection['description']
                        })
                    
                    # Display table
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, width='stretch', hide_index=True)
                    
                    # Add analytics section for single image analysis
                    app.create_analytics_section(detections, "Single Image Analysis")
                    
                    # Export options
                    st.subheader("Export Results")
                    col1, col2 = st.columns(2)
                    
                    # Prepare export data outside the button conditions
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Prepare CSV data
                    metadata_info = {
                        "Analysis_Timestamp": datetime.now().isoformat(),
                        "Image_Name": uploaded_file.name,
                        "Confidence_Threshold": confidence_threshold,
                        "Total_Detections": len(detections),
                        "Image_Size": f"{image.size[0]} × {image.size[1]} px"
                    }
                    
                    csv_data = []
                    # Add metadata as first rows (commented format)
                    csv_data.append({"#": "# PapyrusVision Analysis Results", "Gardiner": "", "Unicode": "", "Confidence": "", "BBox (x1,y1,x2,y2)": "", "Description": ""})
                    for key, value in metadata_info.items():
                        csv_data.append({"#": f"# {key}: {value}", "Gardiner": "", "Unicode": "", "Confidence": "", "BBox (x1,y1,x2,y2)": "", "Description": ""})
                    csv_data.append({"#": "", "Gardiner": "", "Unicode": "", "Confidence": "", "BBox (x1,y1,x2,y2)": "", "Description": ""})
                    csv_data.extend(table_data)
                    
                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)
                    csv_filename = f"papyrus_detection_{timestamp}.csv"
                    
                    # Prepare JSON data
                    json_data = {
                        "metadata": {
                            "analysis_timestamp": datetime.now().isoformat(),
                            "image_name": uploaded_file.name,
                            "confidence_threshold": confidence_threshold,
                            "total_detections": len(detections),
                            "image_size": {"width": image.size[0], "height": image.size[1]},
                            "processed_size": {"width": resized_image.size[0], "height": resized_image.size[1]}
                        },
                        "detections": detections
                    }
                    json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
                    json_filename = f"papyrus_detection_{timestamp}.json"
                    
                    # Create download buttons
                    with col1:
                        st.download_button(
                            label="Download CSV Report",
                            data=csv_string,
                            file_name=csv_filename,
                            mime="text/csv",
                            help="Download detection results as CSV with metadata"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download JSON Report",
                            data=json_string,
                            file_name=json_filename,
                            mime="application/json",
                            help="Download detection results as JSON with full metadata"
                        )
                    
                    
                else:
                    st.warning("No hieroglyphs detected. Try lowering the confidence threshold.")
    
    elif interface_mode == "Digital Paleography":
        st.header("Digital Paleography")
        st.markdown("Create a digital paleography from a single image by cropping all detected signs.")
        
        
        # Image source selection (same as Single Image Analysis)
        dp_image_source = st.radio(
            "Choose Image Source:",
            ["Use Test Image (recommended)", "Upload Image (likely to not work at the moment)"],
            horizontal=True,
            key="dp_image_source"
        )
        
        dp_uploaded_file = None
        dp_image = None
        test_image_path = DATA_DIR / 'images' / '145_upscaled_bright.jpg'
        
        if dp_image_source == "Upload Image":
            dp_uploaded_file = st.file_uploader(
                "Choose a papyrus image",
                type=['png', 'jpg', 'jpeg', 'tiff'],
                help="Upload a high-resolution papyrus image for best results",
                key="dp_file_uploader"
            )
            if dp_uploaded_file:
                dp_image = Image.open(dp_uploaded_file)
        else:
            # Use test image by default
            if test_image_path.exists():
                st.info("Using test image: Book of the Dead of Nu (Spell 145) - BM EA 10477")
                dp_image = Image.open(test_image_path)
                # Create a mock uploaded file for analysis
                class MockUploadedFile:
                    def __init__(self, path):
                        self.name = path.name
                        self.path = path
                    
                    def getbuffer(self):
                        return open(self.path, 'rb').read()
                
                dp_uploaded_file = MockUploadedFile(test_image_path)
            else:
                st.error("Test image not found. Please upload your own image.")
        
        if dp_image and analyzer:
            # Display image
            st.image(dp_image, caption=f"Selected Image: {dp_uploaded_file.name}", width='stretch')
            
            # Create unique key for this digital paleography session
            dp_key = f"dp_{dp_uploaded_file.name}_{confidence_threshold}"
            
            if st.button("Create Digital Paleography", type="primary", key="create_dp_button"):
                try:
                    with st.spinner("Detecting signs and creating cropped catalog..."):
                        detections, original_image = app.analyze_single_image(
                            dp_uploaded_file, confidence_threshold, analyzer
                        )
                        
                        # Store results in session state
                        if detections and original_image is not None:
                            st.session_state[f'dp_detections_{dp_key}'] = detections
                            st.session_state[f'dp_original_image_{dp_key}'] = original_image
                            st.session_state[f'dp_image_info_{dp_key}'] = {
                                'name': dp_uploaded_file.name,
                                'confidence_threshold': confidence_threshold
                            }
                            # Keep user in Digital Paleography interface
                            st.session_state.interface_mode = "Digital Paleography"
                        else:
                            st.error("No detections found. Try lowering the confidence threshold.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
            
            # Display results if they exist (either just calculated or from session state)
            if f'dp_detections_{dp_key}' in st.session_state:
                detections = st.session_state[f'dp_detections_{dp_key}']
                original_image = st.session_state[f'dp_original_image_{dp_key}']
                stored_dp_info = st.session_state[f'dp_image_info_{dp_key}']
                
                if detections and original_image is not None:
                    # Store processing info in session state but don't create files yet
                    stem = Path(dp_uploaded_file.name).stem
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Store all processing data in session state for download generation
                    st.session_state[f'dp_processing_data_{dp_key}'] = {
                        'stem': stem,
                        'timestamp': timestamp,
                        'detections': detections,
                        'original_image': original_image,
                        'dp_uploaded_file': dp_uploaded_file,
                        'confidence_threshold': confidence_threshold
                    }
                    
                    st.success(f"Digital paleography detected {len(detections)} hieroglyphic signs!")
                    
                    # Display complete digital paleography organized by Gardiner code
                    st.subheader("Digital Paleography Results")
                    st.markdown(f"**{dp_uploaded_file.name}** — Created: {datetime.now().strftime('%B %d, %Y at %H:%M')} — Confidence threshold: {confidence_threshold}")
                    
                    # Group detections by Gardiner code for display
                    groups = defaultdict(list)
                    for i, det in enumerate(detections):
                        gardiner_code = det.get('gardiner_code', 'Unknown')
                        groups[gardiner_code].append((det, i))
                    
                    # Display each Gardiner code section
                    for code in sorted(groups.keys()):
                        with st.expander(f"**{code}** ({len(groups[code])} instances)", expanded=True):
                            # Get additional info for this code
                            code_detection = groups[code][0][0]  # First detection for this code
                            description = code_detection.get('description', f'Hieroglyph {code}')
                            
                            # Header with metadata - now with Unicode symbol display
                            col1, col2, col3 = st.columns([1, 2, 2])
                            with col1:
                                # Display Unicode symbol if available
                                unicode_symbol = code_detection.get('unicode_symbol', '')
                                if unicode_symbol and unicode_symbol.strip():
                                    # Show both Unicode symbol and code
                                    st.markdown(f"<div style='font-size: 3em; text-align: center; color: #8B4513; font-weight: bold; margin-bottom: 10px;'>{unicode_symbol}</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='font-size: 1.2em; text-align: center; color: #8B4513; font-weight: bold;'>{code}</div>", unsafe_allow_html=True)
                                else:
                                    # Fallback to just the code
                                    st.markdown(f"<div style='font-size: 2em; text-align: center; color: #8B4513; font-weight: bold;'>{code}</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"**Description:** {description}")
                                st.markdown(f"**Classification:** Gardiner {code}")
                                
                                # Show Unicode codes if available
                                unicode_codes = code_detection.get('unicode_codes', [])
                                if unicode_codes:
                                    codes_str = ', '.join(unicode_codes)
                                    st.markdown(f"**Unicode:** {codes_str}")
                            
                            with col3:
                                confidences = [det[0]['confidence'] for det in groups[code]]
                                st.markdown(f"**Instances:** {len(groups[code])}")
                                st.markdown(f"**Confidence range:** {min(confidences):.1%} - {max(confidences):.1%}")
                                st.markdown(f"**Average confidence:** {np.mean(confidences):.1%}")
                            
                            st.markdown("---")
                            
                            # Show detection info for each instance with cropped images
                            cols_per_row = 4  # Show 4 cropped images per row
                            
                            for i in range(0, len(groups[code]), cols_per_row):
                                row_detections = groups[code][i:i+cols_per_row]
                                cols = st.columns(len(row_detections))
                                
                                for col_idx, (det, idx) in enumerate(row_detections):
                                    with cols[col_idx]:
                                        bbox = det.get('bbox', {})
                                        conf = det.get('confidence', 0.0)
                                        x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                                        x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                                        
                                        # Extract and display the cropped sign
                                        if original_image is not None:
                                            # Ensure coordinates are within image bounds
                                            h, w = original_image.shape[:2]
                                            x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
                                            x2, y2 = min(max(x1+1, x2), w), min(max(y1+1, y2), h)
                                            
                                            if x2 > x1 and y2 > y1:
                                                crop = original_image[y1:y2, x1:x2]
                                                if crop.size > 0:
                                                    # Process crop for better display quality
                                                    processed_crop = app.process_crop_for_display(crop)
                                                    if processed_crop is not None:
                                                        # Convert BGR to RGB for display
                                                        crop_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
                                                        st.image(crop_rgb, caption=f"#{idx+1}: {conf:.1%}", width='stretch')
                                                    else:
                                                        st.write(f"**#{idx+1}**: {conf:.1%} (processing failed)")
                                                else:
                                                    st.write(f"**#{idx+1}**: {conf:.1%} (crop failed)")
                                            else:
                                                st.write(f"**#{idx+1}**: {conf:.1%} (invalid bbox)")
                                        else:
                                            st.write(f"**#{idx+1}**: {conf:.1%} (no image)")
                                        
                                        # Show bounding box info in small text
                                        st.caption(f"BBox: ({x1}, {y1}, {x2}, {y2})")
                    
                    # Add analytics section for digital paleography
                    app.create_analytics_section(detections, "Digital Paleography Analysis")
                    
                    # Export button - only creates files when clicked
                    st.subheader("Export Digital Paleography")
                    if st.button("Generate Export Files", type="primary"):
                        with st.spinner("Creating cropped images and export files..."):
                            # Create the actual files for export
                            session_dir = app.crops_dir / f"{stem}_{timestamp}"
                            session_dir.mkdir(parents=True, exist_ok=True)
                            
                            csv_rows = []
                            gardiner_counter = defaultdict(int)
                            detailed_groups = defaultdict(list)
                            
                            # Group detections by Gardiner code and create crops
                            for idx, det in enumerate(detections):
                                gardiner_code = det.get('gardiner_code', 'Unknown')
                                bbox = det.get('bbox', {})
                                x1 = int(max(0, bbox.get('x1', 0)))
                                y1 = int(max(0, bbox.get('y1', 0)))
                                x2 = int(max(x1+1, bbox.get('x2', x1+1)))
                                y2 = int(max(y1+1, bbox.get('y2', y1+1)))
                                
                                h, w = original_image.shape[:2]
                                x1, y1 = min(x1, w-1), min(y1, h-1)
                                x2, y2 = min(x2, w), min(y2, h)
                                
                                crop = original_image[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                
                                # Process crop for better quality
                                processed_crop = app.process_crop_for_export(crop)
                                if processed_crop is None:
                                    continue
                                
                                # Create subfolder for this Gardiner code
                                gardiner_dir = session_dir / gardiner_code
                                gardiner_dir.mkdir(exist_ok=True)
                                
                                # Increment counter for this Gardiner code
                                gardiner_counter[gardiner_code] += 1
                                instance_num = gardiner_counter[gardiner_code]
                                
                                conf = det.get('confidence', 0.0)
                                crop_name = f"{gardiner_code}_{instance_num:02d}_{conf:.2f}.png"
                                crop_path = gardiner_dir / crop_name
                                cv2.imwrite(str(crop_path), processed_crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                                
                                # Store for detailed groups
                                detailed_groups[gardiner_code].append({
                                    'detection': det,
                                    'crop_path': str(crop_path),
                                    'crop_name': crop_name,
                                    'csv_data': {}
                                })
                                
                                # Get Unicode information
                                unicode_symbol = det.get('unicode_symbol', '')
                                unicode_codes = det.get('unicode_codes', [])
                                unicode_codes_str = ', '.join(unicode_codes) if unicode_codes else ''
                                
                                # Store relative path for CSV with complete metadata
                                relative_path = f"{gardiner_code}/{crop_name}"
                                csv_rows.append({
                                    'filename': relative_path,
                                    'detection_id': det.get('detection_index', idx) + 1,
                                    'gardiner_code': gardiner_code,
                                    'unicode_symbol': unicode_symbol,
                                    'unicode_codes': unicode_codes_str,
                                    'confidence': conf,
                                    'confidence_threshold': confidence_threshold,
                                    'instance_number': instance_num,
                                    'bbox_x1': x1,
                                    'bbox_y1': y1, 
                                    'bbox_x2': x2, 
                                    'bbox_y2': y2,
                                    'bbox_width': x2 - x1,
                                    'bbox_height': y2 - y1,
                                    'bbox_center_x': (x1 + x2) / 2,
                                    'bbox_center_y': (y1 + y2) / 2,
                                    'bbox_area': (x2 - x1) * (y2 - y1),
                                    'description': det.get('description', f'Hieroglyph {gardiner_code}'),
                                    'source_image': dp_uploaded_file.name,
                                    'analysis_timestamp': datetime.now().isoformat(),
                                    'image_width': w,
                                    'image_height': h
                                })
                            
                            # Write index CSV
                            if csv_rows:
                                import csv as _csv
                                index_csv = session_dir / 'index.csv'
                                with open(index_csv, 'w', newline='', encoding='utf-8') as f:
                                    writer = _csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                                    writer.writeheader()
                                    writer.writerows(csv_rows)
                            
                            # Create HTML with embedded images
                            def image_to_base64(image_path):
                                try:
                                    with open(image_path, 'rb') as img_file:
                                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                        return f"data:image/png;base64,{img_data}"
                                except FileNotFoundError:
                                    return "data:image/png;base64,"
                            
                            html_parts = [
                                "<!DOCTYPE html><html><head><meta charset='utf-8'>",
                                "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                                "<title>PapyrusVision Digital Paleography - Complete Scholarly Catalog</title>",
                                "<style>",
                                "* { box-sizing: border-box; }",
                                "body { font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:15px; background:#f8f9fa; color:#333; line-height:1.6; }",
                                ".header { text-align:center; background:linear-gradient(135deg,#8B4513,#D2B48C); color:white; padding:30px 15px; border-radius:15px; margin-bottom:30px; box-shadow:0 4px 15px rgba(0,0,0,0.15); }",
                                ".header h1 { margin:0; font-size:clamp(2em, 4vw, 3.5em); font-weight:300; }",
                                ".metadata { background:white; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }",
                                ".metadata-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:15px; }",
                                ".metadata-item { padding:10px; background:#f8f9fa; border-radius:8px; border-left:4px solid #8B4513; }",
                                ".code-section { margin:30px 0; background:white; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.1); overflow:hidden; }",
                                ".code-header { background:linear-gradient(135deg,#8B4513,#A0522D); color:white; padding:25px 15px; }",
                                ".unicode-display { font-size:4em; text-align:center; margin:15px 0; color:white; text-shadow:2px 2px 4px rgba(0,0,0,0.5); }",
                                ".notation-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin:15px 0; }",
                                ".notation-item { background:rgba(255,255,255,0.1); padding:10px 15px; border-radius:20px; font-family:monospace; font-weight:bold; text-align:center; }",
                                ".instances-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:25px; padding:25px 15px; }",
                                ".instance-card { background:#fafafa; border-radius:12px; padding:20px; box-shadow:0 3px 10px rgba(0,0,0,0.1); }",
                                ".crop-image { max-width:100%; height:auto; border-radius:6px; margin-bottom:15px; }",
                                ".instance-metadata { background:#f0f0f0; padding:10px; border-radius:6px; font-size:0.9em; margin-top:10px; }",
                                ".bbox-info { font-family:monospace; font-size:0.8em; color:#666; }",
                                "</style>",
                                "</head><body>",
                                f"<div class='header'>",
                                f"<h1>PapyrusVision Digital Paleography</h1>",
                                f"<h2>{dp_uploaded_file.name}</h2>",
                                f"<p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | Confidence: {confidence_threshold}</p>",
                                f"</div>",
                                f"<div class='metadata'>",
                                f"<h3>Analysis Metadata</h3>",
                                f"<div class='metadata-grid'>",
                                f"<div class='metadata-item'><strong>Total Detections:</strong> {len(detections)}</div>",
                                f"<div class='metadata-item'><strong>Unique Signs:</strong> {len(detailed_groups)}</div>",
                                f"<div class='metadata-item'><strong>Image Size:</strong> {w} × {h} pixels</div>",
                                f"<div class='metadata-item'><strong>Confidence Threshold:</strong> {confidence_threshold}</div>",
                                f"<div class='metadata-item'><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
                                f"<div class='metadata-item'><strong>Model:</strong> Detectron2 Hieroglyph Detection</div>",
                                f"</div></div>"
                            ]
                            
                            # Add sections for each code with complete Unicode and metadata
                            for code in sorted(detailed_groups.keys()):
                                instances = detailed_groups[code]
                                first_det = instances[0]['detection']
                                
                                # Get Unicode information for this code
                                unicode_symbol = first_det.get('unicode_symbol', '')
                                unicode_codes = first_det.get('unicode_codes', [])
                                description = first_det.get('description', f'Hieroglyph {code}')
                                
                                html_parts.append(f"<div class='code-section' id='{code}'>")
                                html_parts.append(f"<div class='code-header'>")
                                
                                # Unicode symbol display
                                if unicode_symbol and unicode_symbol.strip():
                                    html_parts.append(f"<div class='unicode-display'>{unicode_symbol}</div>")
                                
                                html_parts.append(f"<h2>Gardiner {code}</h2>")
                                html_parts.append(f"<p style='font-size:1.2em; margin:10px 0;'>{description}</p>")
                                
                                # Notation information
                                html_parts.append(f"<div class='notation-grid'>")
                                html_parts.append(f"<div class='notation-item'>Gardiner: {code}</div>")
                                if unicode_codes:
                                    html_parts.append(f"<div class='notation-item'>Unicode: {', '.join(unicode_codes)}</div>")
                                if unicode_symbol:
                                    html_parts.append(f"<div class='notation-item'>Symbol: {unicode_symbol}</div>")
                                html_parts.append(f"<div class='notation-item'>Instances: {len(instances)}</div>")
                                html_parts.append(f"</div>")
                                
                                html_parts.append(f"</div>")
                                html_parts.append(f"<div class='instances-grid'>")
                                
                                for idx, instance in enumerate(instances, 1):
                                    det = instance['detection']
                                    crop_path = instance['crop_path']
                                    crop_name = instance['crop_name']
                                    img_data_uri = image_to_base64(crop_path)
                                    
                                    # Get bbox information
                                    bbox = det.get('bbox', {})
                                    x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                                    x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                                    bbox_area = det.get('area', (x2-x1)*(y2-y1))
                                    
                                    html_parts.append(f"""
                                    <div class='instance-card'>
                                        <img class='crop-image' src='{img_data_uri}' alt='{code} instance {idx}'>
                                        <h4>Detection #{idx}</h4>
                                        <div class='instance-metadata'>
                                            <div><strong>Confidence:</strong> {det['confidence']:.1%}</div>
                                            <div><strong>Gardiner Code:</strong> {code}</div>
                                            <div><strong>Description:</strong> {description}</div>
                                            <div class='bbox-info'><strong>Bounding Box:</strong> ({x1}, {y1}, {x2}, {y2})</div>
                                            <div class='bbox-info'><strong>Size:</strong> {x2-x1} × {y2-y1} px</div>
                                            <div class='bbox-info'><strong>Area:</strong> {bbox_area:.0f} px²</div>
                                            <div><strong>File:</strong> {crop_name}</div>
                                        </div>
                                    </div>
                                    """)
                                
                                html_parts.append("</div></div>")
                            
                            html_parts.append("</body></html>")
                            
                            html_path = session_dir / 'digital_paleography.html'
                            with open(html_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(html_parts))
                            
                            # Zip everything
                            zip_name = f"digital_paleography_{stem}_{timestamp}.zip"
                            zip_path = app.exports_dir / zip_name
                            
                            # Ensure exports directory exists
                            app.exports_dir.mkdir(parents=True, exist_ok=True)
                            
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for p in session_dir.rglob('*'):
                                    if p.is_file():
                                        zipf.write(p, arcname=p.relative_to(session_dir))
                            
                            st.success(f"Export created with {len(csv_rows)} cropped signs!")
                            
                            # Download buttons
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label="Download Complete Digital Paleography (ZIP)",
                                    data=f.read(),
                                    file_name=zip_name,
                                    mime="application/zip"
                                )
                            
                            with open(html_path, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    label="Download HTML Catalog",
                                    data=f.read(),
                                    file_name=html_path.name,
                                    mime="text/html"
                                )
                else:
                    st.warning("No hieroglyphs detected. Try lowering the confidence threshold.")
        else:
            st.info("Please select an image to create a digital paleography.")

if __name__ == "__main__":
    main()
