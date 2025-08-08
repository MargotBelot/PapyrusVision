#!/usr/bin/env python3
"""
PapyrusNU Hieroglyph Detection - Interactive Analysis Tool

This tool provides an interactive visualization of hieroglyph detections with
comprehensive data export capabilities including Gardiner codes, coordinates,
and Unicode information.

Features:
- Load and visualize papyrus images
- Run hieroglyph detection with confidence scoring
- Interactive visualization with bounding boxes and labels
- Export results to JSON and CSV formats
- Include Gardiner codes, coordinates, Unicode codes and symbols
- Confidence-based filtering and analysis

Usage:
    python scripts/hieroglyph_analysis_tool.py [--image_path IMAGE_PATH] [--confidence_threshold 0.5]
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import torch
from PIL import Image, ImageFont, ImageDraw
import glob
from datetime import datetime
from pathlib import Path

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

sys.path.append(SCRIPTS_DIR)

class HieroglyphAnalysisTool:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.model = None
        self.predictor = None
        self.model_info = None
        self.unicode_mapping = {}
        self.cfg = None
        
    def load_model(self):
        """Load the trained Detectron2 model"""
        print("🔄 Loading Detectron2 model...")
        
        try:
            import detectron2
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.utils.visualizer import Visualizer, ColorMode
            
            print(f"   ✅ Detectron2 version: {detectron2.__version__}")
        except ImportError as e:
            print(f"   ❌ Error importing Detectron2: {e}")
            return False
        
        # Find latest model
        model_dirs = glob.glob(os.path.join(MODELS_DIR, 'hieroglyph_model_*'))
        if not model_dirs:
            print("   ❌ No trained models found!")
            return False
        
        latest_model_dir = sorted(model_dirs)[-1]
        model_name = os.path.basename(latest_model_dir)
        print(f"   Using model: {model_name}")
        
        # Load model info
        model_info_file = os.path.join(latest_model_dir, 'model_info.json')
        if not os.path.exists(model_info_file):
            print(f"   Model info not found: {model_info_file}")
            return False
            
        with open(model_info_file, 'r') as f:
            self.model_info = json.load(f)
        
        # Load Unicode mapping
        unicode_file = os.path.join(DATA_DIR, 'annotations', 'gardiner_unicode_mapping.json')
        if os.path.exists(unicode_file):
            with open(unicode_file, 'r') as f:
                unicode_data = json.load(f)
            
            # Create mapping from Gardiner code to Unicode info
            for gardiner_code, info in unicode_data.items():
                self.unicode_mapping[gardiner_code] = {
                    'unicode_codes': info.get('unicode_codes', []),
                    'description': info.get('description', 'Unknown'),
                    'unicode_symbol': self.get_unicode_symbol(info.get('unicode_codes', []))
                }
            print(f"   Loaded Unicode mappings for {len(self.unicode_mapping)} Gardiner codes")
        else:
            print("   Unicode mapping file not found")
        
        # Set up model configuration
        self.cfg = get_cfg()
        config_file = os.path.join(latest_model_dir, 'config.yaml')
        if os.path.exists(config_file):
            self.cfg.merge_from_file(config_file)
        
        self.cfg.MODEL.WEIGHTS = os.path.join(latest_model_dir, 'model_final.pth')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.model_info['detection_threshold']
        self.cfg.MODEL.DEVICE = 'cpu'  # Force CPU usage
        
        # Create predictor
        try:
            self.predictor = DefaultPredictor(self.cfg)
            print("   Model loaded successfully!")
            print(f"   Default confidence threshold: {self.model_info['detection_threshold']}")
            return True
        except Exception as e:
            print(f"   Error loading model: {e}")
            return False
    
    def get_unicode_symbol(self, unicode_codes):
        """Convert Unicode codes to actual symbols - DISABLED for prediction display"""
        # Always return empty string to disable Unicode symbol display
        return ""
    
    def predict_hieroglyphs(self, image_path, confidence_threshold=None):
        """Run hieroglyph detection on an image"""
        if self.predictor is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None
        
        if confidence_threshold is not None:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            from detectron2.engine import DefaultPredictor
            self.predictor = DefaultPredictor(self.cfg)  # Recreate with new threshold
        
        print(f"Analyzing image: {os.path.basename(image_path)}")
        print(f"   Confidence threshold: {self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"   Image size: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Make prediction
        outputs = self.predictor(img)
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
                gardiner_code = self.model_info['category_names'][class_idx] if class_idx < len(self.model_info['category_names']) else f'Unknown_{class_idx}'
                
                # Get Unicode information
                unicode_info = self.unicode_mapping.get(gardiner_code, {
                    'unicode_codes': [],
                    'description': 'No description available',
                    'unicode_symbol': ''
                })
                
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
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'image_size': {
                'width': img.shape[1],
                'height': img.shape[0],
                'channels': img.shape[2]
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': os.path.basename(sorted(glob.glob(os.path.join(MODELS_DIR, 'hieroglyph_model_*')))[-1]),
                'confidence_threshold': float(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST),
                'total_classes': len(self.model_info['category_names'])
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
        
        print(f"   Analysis complete!")
        print(f"   Found {len(detections)} hieroglyphs")
        print(f"   Unique classes: {len(set([d['gardiner_code'] for d in detections]))}")
        print(f"   Confidence range: {result['summary']['confidence_stats']['min']:.3f} - {result['summary']['confidence_stats']['max']:.3f}")
        
        return result
    
    def visualize_results(self, image_path, results, save_path=None, show_plot=True):
        """Create interactive visualization of detection results"""
        print("Creating visualization...")
        
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.imshow(img_rgb)
        
        # Color mapping for confidence levels
        def get_confidence_color(confidence):
            if confidence >= 0.8:
                return 'green'
            elif confidence >= 0.6:
                return 'blue'
            else:
                return 'orange'
        
        # Draw detections
        for detection in results['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            gardiner_code = detection['gardiner_code']
            unicode_symbol = detection['unicode_symbol']
            
            # Draw bounding box
            color = get_confidence_color(confidence)
            rect = Rectangle((bbox['x1'], bbox['y1']), bbox['width'], bbox['height'],
                           linewidth=3, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Create label with Gardiner code and confidence (NO Unicode symbols)
            label_parts = [gardiner_code, f"{confidence:.3f}"]
            label = ' '.join(label_parts)
            
            # Add text label
            ax.text(bbox['x1'], bbox['y1'] - 10, label, 
                   color=color, fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.3, label='High confidence (≥0.8)'),
            plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.3, label='Medium confidence (0.6-0.8)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.3, label='Low confidence (<0.6)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Set title
        title = f"Hieroglyph Detection Results: {results['image_name']}\\n"
        title += f"{results['summary']['total_detections']} detections, "
        title += f"{results['summary']['unique_classes']} unique classes, "
        title += f"threshold: {results['model_info']['confidence_threshold']}"
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Visualization saved: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def export_to_json(self, results, output_path):
        """Export results to JSON format"""
        print(f"📄 Exporting results to JSON: {os.path.basename(output_path)}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ JSON export complete: {output_path}")
    
    def export_to_csv(self, results, output_path):
        """Export results to CSV format"""
        print(f"Exporting results to CSV: {os.path.basename(output_path)}")
        
        # Prepare CSV data with consistent Unicode handling
        csv_data = []
        for detection in results['detections']:
            # Get only the first valid Unicode code (filter out <g>CODE</g> entries)
            valid_unicode_codes = [code for code in detection['unicode_codes'] if code.startswith('U+')]
            primary_unicode_code = valid_unicode_codes[0] if valid_unicode_codes else ''
            
            row = {
                'detection_id': detection['id'],
                'gardiner_code': detection['gardiner_code'],
                'confidence': detection['confidence'],
                'x1': detection['bbox']['x1'],
                'y1': detection['bbox']['y1'],
                'x2': detection['bbox']['x2'],
                'y2': detection['bbox']['y2'],
                'width': detection['bbox']['width'],
                'height': detection['bbox']['height'],
                'center_x': detection['bbox']['center_x'],
                'center_y': detection['bbox']['center_y'],
                'area': detection['area'],
                'unicode_code': primary_unicode_code,  # Single primary Unicode code
                # 'unicode_symbol': detection['unicode_symbol'],  # Disabled for cleaner output
                'description': detection['description'],
                'image_name': results['image_name'],
                'analysis_timestamp': results['analysis_timestamp']
            }
            csv_data.append(row)
        
        # Write CSV
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"   CSV export complete: {output_path}")
    
    def print_summary(self, results):
        """Print a comprehensive summary of the analysis"""
        print("\\n" + "="*70)
        print("HIEROGLYPH DETECTION ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\\nIMAGE INFORMATION:")
        print(f"   File: {results['image_name']}")
        print(f"   Size: {results['image_size']['width']}x{results['image_size']['height']} pixels")
        print(f"   Analysis: {results['analysis_timestamp']}")
        
        print(f"\\nDETECTION RESULTS:")
        print(f"   Total Detections: {results['summary']['total_detections']}")
        print(f"   Unique Gardiner Codes: {results['summary']['unique_classes']}")
        print(f"   Confidence Threshold: {results['model_info']['confidence_threshold']}")
        
        if results['summary']['total_detections'] > 0:
            print(f"\\nCONFIDENCE DISTRIBUTION:")
            print(f"   High (≥0.8): {results['summary']['high_confidence_count']} detections")
            print(f"   Medium (0.6-0.8): {results['summary']['medium_confidence_count']} detections")
            print(f"   Low (<0.6): {results['summary']['low_confidence_count']} detections")
            
            print(f"\\nCONFIDENCE STATISTICS:")
            stats = results['summary']['confidence_stats']
            print(f"   Mean: {stats['mean']:.3f}")
            print(f"   Range: {stats['min']:.3f} - {stats['max']:.3f}")
            print(f"   Std Dev: {stats['std']:.3f}")
            
            print(f"\\nDETECTED HIEROGLYPHS:")
            for i, detection in enumerate(results['detections'][:10], 1):
                # No Unicode symbols displayed
                print(f"   {i:2d}. {detection['gardiner_code']} - {detection['confidence']:.3f}")
            
            if len(results['detections']) > 10:
                print(f"       ... and {len(results['detections']) - 10} more")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Interactive Hieroglyph Detection Analysis Tool')
    parser.add_argument('--image_path', type=str, help='Path to the image to analyze')
    parser.add_argument('--confidence_threshold', type=float, default=None, 
                       help='Detection confidence threshold (default: use model default)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as image directory)')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip interactive visualization')
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = HieroglyphAnalysisTool()
    
    # Load model
    if not tool.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Determine image path
    if args.image_path:
        image_path = args.image_path
    else:
        # Use test image by default
        test_image = os.path.join(DATA_DIR, 'images', '145_upscaled_bright.jpg')
        if os.path.exists(test_image):
            image_path = test_image
            print(f"Using default test image: {os.path.basename(test_image)}")
        else:
            print("❌ No image specified and default test image not found.")
            print("Usage: python hieroglyph_analysis_tool.py --image_path YOUR_IMAGE.jpg")
            return
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(image_path), 'hieroglyph_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    results = tool.predict_hieroglyphs(image_path, args.confidence_threshold)
    if results is None:
        return
    
    # Create output filenames
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_output = os.path.join(output_dir, f"{image_name}_hieroglyph_analysis_{timestamp}.json")
    csv_output = os.path.join(output_dir, f"{image_name}_hieroglyph_analysis_{timestamp}.csv")
    viz_output = os.path.join(output_dir, f"{image_name}_hieroglyph_visualization_{timestamp}.png")
    
    # Export results
    tool.export_to_json(results, json_output)
    tool.export_to_csv(results, csv_output)
    
    # Create visualization
    if not args.no_visualization:
        tool.visualize_results(image_path, results, viz_output, show_plot=True)
    else:
        tool.visualize_results(image_path, results, viz_output, show_plot=False)
    
    # Print summary
    tool.print_summary(results)
    
    print(f"\\nOUTPUT FILES:")
    print(f"   JSON: {json_output}")
    print(f"   CSV: {csv_output}")
    print(f"   Visualization: {viz_output}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
