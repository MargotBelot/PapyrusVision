#!/usr/bin/env python3
"""
Digital Paleography Tool - Batch Processing Version

This command-line tool is designed for RESEARCH WORKFLOWS and LARGE-SCALE PROCESSING.

Use this tool when you:
- Need to process entire directories of images automatically
- Want to integrate hieroglyph analysis into research pipelines
- Require fine-tuned control over confidence thresholds and filtering
- Are working with large datasets that need batch processing

For interactive analysis and single-image exploration, use the Streamlit web application:
    streamlit run streamlit_hieroglyphs_app.py

This tool creates a digital paleography by:
1. Processing images to detect hieroglyphs
2. Cropping individual signs from detections
3. Organizing crops by Gardiner codes
4. Creating an interactive HTML catalog with descriptions and Unicode info
5. Generating comprehensive reports and structured output
"""

import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import base64
from io import BytesIO
from PIL import Image
import shutil

# Import your existing detection modules
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

try:
    from scripts.hieroglyph_analysis_tool import HieroglyphAnalysisTool
except ImportError:
    print("Could not import HieroglyphAnalysisTool. Make sure the module exists.")

class DigitalPaleographyTool:
    def __init__(self):
        """Initialize the Digital Paleography Tool"""
        self.setup_directories()
        self.load_unicode_mappings()
        self.load_gardiner_descriptions()
        
        # Initialize the hieroglyph analyzer
        try:
            self.analyzer = HieroglyphAnalysisTool()
            if self.analyzer.load_model():
                print("Hieroglyph analyzer loaded successfully")
            else:
                print("Error loading model in analyzer")
                self.analyzer = None
        except Exception as e:
            print(f"Error loading analyzer: {e}")
            self.analyzer = None
    
    def setup_directories(self):
        """Create directory structure for the paleography"""
        self.base_dir = Path("/Users/margot/Desktop/PapyrusNU_Detectron/digital_paleography")
        self.crops_dir = self.base_dir / "cropped_signs"
        self.catalog_dir = self.base_dir / "catalog"
        self.reports_dir = self.base_dir / "reports"
        
        # Create directories
        for dir_path in [self.base_dir, self.crops_dir, self.catalog_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def load_unicode_mappings(self):
        """Load Unicode mappings for Gardiner codes"""
        mapping_file = "/Users/margot/Desktop/PapyrusNU_Detectron/data/annotations/gardiner_unicode_mapping.json"
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.unicode_mappings = json.load(f)
            print(f"Loaded {len(self.unicode_mappings)} Unicode mappings")
        except Exception as e:
            print(f"Error loading Unicode mappings: {e}")
            self.unicode_mappings = {}
    
    def load_gardiner_descriptions(self):
        """Load Gardiner code descriptions"""
        descriptions_file = "/Users/margot/Desktop/PapyrusNU_Detectron/data/annotations/gardiner_descriptions.json"
        try:
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                self.gardiner_descriptions = json.load(f)
            print(f"Loaded descriptions for {len(self.gardiner_descriptions)} Gardiner codes")
        except Exception as e:
            print(f"Error loading Gardiner descriptions: {e}")
            # Fallback to basic descriptions
            self.gardiner_descriptions = {
                'A1': 'Man sitting',
                'A2': 'Man with hand to mouth',
                'B1': 'Woman sitting',
                'C11': 'God with feather',
                'D21': 'Mouth',
                'D46': 'Hand',
                'F31': 'Three fox skins',
                'G1': 'Egyptian vulture',
                'G17': 'Owl',
                'G43': 'Quail chick',
                'M17': 'Reed',
                'N35': 'Ripple of water',
                'O1': 'House',
                'O4': 'Reed shelter',
                'P1': 'Boat on water',
                'R1': 'High table',
                'S28': 'Linen on pole',
                'T22': 'Arrowhead',
                'U23': 'Chisel',
                'V1': 'Coil of rope',
                'X1': 'Bread loaf',
                'Y1': 'Papyrus roll',
                'Z1': 'Stroke',
                'Z2': 'Three strokes'
            }
    
    def get_unicode_symbol(self, gardiner_code):
        """Get Unicode symbol for a Gardiner code"""
        if gardiner_code not in self.unicode_mappings:
            return None
        
        unicode_codes = self.unicode_mappings[gardiner_code].get('unicode_codes', [])
        if not unicode_codes:
            return None
        
        # Get the first valid Unicode code
        for code in unicode_codes:
            if code.startswith('U+'):
                try:
                    # Convert Unicode code to character
                    unicode_int = int(code[2:], 16)
                    return chr(unicode_int)
                except (ValueError, OverflowError):
                    continue
        
        return None
    
    def crop_detection(self, image, bbox, padding=10):
        """Crop a detection from the image with padding"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(w, int(x2 + padding))
        y2 = min(h, int(y2 + padding))
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def process_image(self, image_path, confidence_threshold=0.5):
        """Process a single image and extract hieroglyph crops"""
        if not self.analyzer:
            print("Analyzer not available")
            return []
        
        print(f"Processing image: {image_path}")
        
        # Load and analyze the image
        try:
            results = self.analyzer.predict_hieroglyphs(str(image_path), confidence_threshold)
            if not results or not results.get('detections'):
                print(f"   No detections found in {image_path}")
                return []
            detections = results['detections']
        except Exception as e:
            print(f"   Error analyzing {image_path}: {e}")
            return []
        
        # Load the original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   Could not load image: {image_path}")
            return []
        
        crops_data = []
        image_name = Path(image_path).stem
        
        # Process each detection
        for i, detection in enumerate(detections):
            confidence = detection.get('confidence', 0.0)
            if confidence < confidence_threshold:
                continue
            
            gardiner_code = detection.get('gardiner_code', 'Unknown')
            
            # Skip X1 (bread loaf) sign for testing purposes
            if gardiner_code == "X1":
                continue
            
            bbox_dict = detection.get('bbox', {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100})
            bbox = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]
            
            # Crop the detection
            cropped = self.crop_detection(image, bbox)
            
            if cropped.size == 0:
                continue
            
            # Get Unicode code for filename
            unicode_code_for_filename = ""
            if gardiner_code in self.unicode_mappings:
                codes = self.unicode_mappings[gardiner_code].get('unicode_codes', [])
                if codes:
                    # Use the first valid Unicode code, clean it for filename
                    first_code = codes[0]
                    if first_code.startswith('U+'):
                        unicode_code_for_filename = f"_{first_code.replace('+', '')}"
            
            # Create filename for the crop with Unicode code
            crop_filename = f"{image_name}_{gardiner_code}{unicode_code_for_filename}_{i:03d}_conf{confidence:.2f}.png"
            
            # Create Gardiner code directory if it doesn't exist
            gardiner_dir = self.crops_dir / gardiner_code
            gardiner_dir.mkdir(exist_ok=True)
            
            # Save the crop
            crop_path = gardiner_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            
            # Get additional information
            unicode_symbol = self.get_unicode_symbol(gardiner_code)
            unicode_code = None
            if gardiner_code in self.unicode_mappings:
                codes = self.unicode_mappings[gardiner_code].get('unicode_codes', [])
                unicode_code = codes[0] if codes else None
            
            description = self.gardiner_descriptions.get(gardiner_code, f"Hieroglyph {gardiner_code}")
            
            crop_data = {
                'source_image': image_name,
                'gardiner_code': gardiner_code,
                'confidence': confidence,
                'bbox': bbox,
                'crop_filename': crop_filename,
                'crop_path': str(crop_path),
                'unicode_symbol': unicode_symbol,
                'unicode_code': unicode_code,
                'description': description,
                'dimensions': cropped.shape[:2]  # height, width
            }
            
            crops_data.append(crop_data)
            print(f"   Cropped {gardiner_code} (conf: {confidence:.2f})")
        
        return crops_data
    
    def process_directory(self, input_dir, confidence_threshold=0.5):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        all_crops_data = []
        for image_file in image_files:
            crops_data = self.process_image(image_file, confidence_threshold)
            all_crops_data.extend(crops_data)
        
        return all_crops_data
    
    def create_html_catalog(self, crops_data):
        """Create an interactive HTML catalog of the paleography"""
        print("Creating HTML catalog...")
        
        # Group crops by Gardiner code
        grouped_crops = defaultdict(list)
        for crop in crops_data:
            grouped_crops[crop['gardiner_code']].append(crop)
        
        # Sort Gardiner codes
        sorted_codes = sorted(grouped_crops.keys())
        
        # Create HTML content
        html_content = self.generate_html_catalog(sorted_codes, grouped_crops)
        
        # Save HTML file
        catalog_file = self.catalog_dir / "digital_paleography.html"
        with open(catalog_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML catalog created: {catalog_file}")
        return catalog_file
    
    def generate_html_catalog(self, sorted_codes, grouped_crops):
        """Generate the HTML content for the catalog"""
        
        html_head = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Hieroglyph Paleography</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #8B4513, #D2B48C);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .gardiner-section {
            background: white;
            margin: 30px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .gardiner-header {
            background: #8B4513;
            color: white;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .unicode-symbol {
            font-size: 2em;
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            min-width: 60px;
            text-align: center;
        }
        .gardiner-info h2 {
            margin: 0;
            font-size: 1.5em;
        }
        .gardiner-info p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .crops-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .crop-item {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            background: #fafafa;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .crop-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-color: #8B4513;
        }
        .crop-image {
            max-width: 100%;
            max-height: 150px;
            object-fit: contain;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .crop-info {
            font-size: 0.9em;
            color: #666;
        }
        .confidence {
            background: #e8f5e8;
            color: #2d5a2d;
            padding: 3px 8px;
            border-radius: 15px;
            font-weight: bold;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #8B4513;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .toc {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .toc h3 {
            margin-top: 0;
            color: #8B4513;
        }
        .toc-list {
            column-count: 4;
            column-gap: 20px;
        }
        .toc-item {
            break-inside: avoid;
            margin-bottom: 5px;
        }
        .toc-link {
            color: #666;
            text-decoration: none;
            padding: 5px 10px;
            display: block;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .toc-link:hover {
            background-color: #f0f0f0;
            color: #8B4513;
        }
        @media (max-width: 768px) {
            .toc-list { column-count: 2; }
            .stats-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>"""
        
        # Calculate statistics
        total_signs = sum(len(crops) for crops in grouped_crops.values())
        total_codes = len(sorted_codes)
        avg_confidence = np.mean([crop['confidence'] for crops in grouped_crops.values() for crop in crops])
        
        html_body = f"""
    <div class="header">
        <h1>Digital Hieroglyph Paleography</h1>
        <p>Comprehensive catalog of detected hieroglyphic signs</p>
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
    </div>
    
    <div class="stats">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{total_signs}</div>
                <div class="stat-label">Total Signs</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{total_codes}</div>
                <div class="stat-label">Unique Gardiner Codes</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{avg_confidence:.1%}</div>
                <div class="stat-label">Average Confidence</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(self.unicode_mappings)}</div>
                <div class="stat-label">Unicode Mappings</div>
            </div>
        </div>
    </div>
    
    <div class="toc">
        <h3>Table of Contents</h3>
        <div class="toc-list">"""
        
        # Add table of contents
        for code in sorted_codes:
            count = len(grouped_crops[code])
            html_body += f'            <div class="toc-item"><a href="#{code}" class="toc-link">{code} ({count})</a></div>\n'
        
        html_body += """        </div>
    </div>"""
        
        # Add each Gardiner code section
        for code in sorted_codes:
            crops = grouped_crops[code]
            unicode_symbol = self.get_unicode_symbol(code)
            unicode_display = unicode_symbol if unicode_symbol else "—"
            
            unicode_code = ""
            if code in self.unicode_mappings:
                codes = self.unicode_mappings[code].get('unicode_codes', [])
                unicode_code = codes[0] if codes else ""
            
            description = self.gardiner_descriptions.get(code, f"Hieroglyph {code}")
            
            html_body += f"""
    <div class="gardiner-section" id="{code}">
        <div class="gardiner-header">
            <div class="unicode-symbol">{unicode_display}</div>
            <div class="gardiner-info">
                <h2>{code}</h2>
                <p>{description}</p>
                <p>Unicode: {unicode_code} • {len(crops)} instances</p>
            </div>
        </div>
        <div class="crops-grid">"""
            
            # Add crop images
            for crop in crops:
                # Convert image to base64 for embedding
                try:
                    img_path = crop['crop_path']
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    
                    html_body += f"""
            <div class="crop-item">
                <img src="data:image/png;base64,{img_data}" alt="{code}" class="crop-image">
                <div class="crop-info">
                    <div><strong>{crop['source_image']}</strong></div>
                    <div class="confidence">{crop['confidence']:.1%}</div>
                    <div>{crop['dimensions'][1]}×{crop['dimensions'][0]}px</div>
                </div>
            </div>"""
                except Exception as e:
                    print(f"   Could not embed image {crop['crop_path']}: {e}")
            
            html_body += """
        </div>
    </div>"""
        
        html_footer = """
</body>
</html>"""
        
        return html_head + html_body + html_footer
    
    def generate_report(self, crops_data):
        """Generate a comprehensive report of the paleography"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_signs": len(crops_data),
                "unique_gardiner_codes": len(set(crop['gardiner_code'] for crop in crops_data)),
                "average_confidence": np.mean([crop['confidence'] for crop in crops_data]),
                "source_images": len(set(crop['source_image'] for crop in crops_data))
            },
            "by_gardiner_code": {},
            "by_source_image": {}
        }
        
        # Group by Gardiner code
        by_code = defaultdict(list)
        for crop in crops_data:
            by_code[crop['gardiner_code']].append(crop)
        
        for code, crops in by_code.items():
            report["by_gardiner_code"][code] = {
                "count": len(crops),
                "average_confidence": np.mean([c['confidence'] for c in crops]),
                "unicode_code": crops[0]['unicode_code'],
                "description": crops[0]['description']
            }
        
        # Group by source image
        by_image = defaultdict(list)
        for crop in crops_data:
            by_image[crop['source_image']].append(crop)
        
        for image, crops in by_image.items():
            report["by_source_image"][image] = {
                "signs_detected": len(crops),
                "unique_codes": len(set(c['gardiner_code'] for c in crops)),
                "average_confidence": np.mean([c['confidence'] for c in crops])
            }
        
        # Save report
        report_file = self.reports_dir / f"paleography_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved: {report_file}")
        return report_file

def main():
    print("DIGITAL HIEROGLYPH PALEOGRAPHY TOOL")
    print("=" * 60)
    
    # Initialize the tool
    tool = DigitalPaleographyTool()
    
    # Get input directory from user
    input_dir = input("Enter path to directory containing hieroglyph images: ").strip()
    if not input_dir:
        input_dir = "/Users/margot/Desktop/sample_hieroglyphs"  # Default for testing
    
    # Get confidence threshold
    confidence_input = input("Enter confidence threshold (0.0-1.0, default 0.5): ").strip()
    try:
        confidence_threshold = float(confidence_input) if confidence_input else 0.5
    except ValueError:
        confidence_threshold = 0.5
    
    print(f"\nProcessing images from: {input_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Process all images
    crops_data = tool.process_directory(input_dir, confidence_threshold)
    
    if not crops_data:
        print("No crops were generated. Check your input directory and confidence threshold.")
        return
    
    print(f"\nGenerated {len(crops_data)} sign crops")
    
    # Create HTML catalog
    catalog_file = tool.create_html_catalog(crops_data)
    
    # Generate report
    report_file = tool.generate_report(crops_data)
    
    # Final summary
    print(f"\nDIGITAL PALEOGRAPHY COMPLETE!")
    print(f"   Crops saved in: {tool.crops_dir}")
    print(f"   HTML catalog: {catalog_file}")
    print(f"   Report: {report_file}")
    print(f"   Total signs: {len(crops_data)}")
    print(f"   Unique codes: {len(set(crop['gardiner_code'] for crop in crops_data))}")
    
    # Offer to open the catalog
    open_catalog = input(f"\nOpen HTML catalog in browser? (y/n): ").lower()
    if open_catalog in ['y', 'yes']:
        import webbrowser
        webbrowser.open(f"file://{catalog_file}")

if __name__ == "__main__":
    main()
