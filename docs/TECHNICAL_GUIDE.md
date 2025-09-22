# PapyrusVision: Technical Guide

**Advanced documentation for developers, researchers, and power users**

> **Quick Start**: For basic installation and usage, see [README.md](../README.md)

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Model Performance](#model-performance) 
3. [Advanced Usage](#advanced-usage)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Core Components
- **Detection Engine**: Detectron2 with Faster R-CNN (ResNet-50 + FPN backbone)
- **Training Data**: 2,430 annotated hieroglyphs across 177 Gardiner classes
- **Performance**: mAP@0.5: 0.73, mAP@0.5:0.95: 0.41
- **Unicode Support**: 594+ Egyptian Hieroglyphs (U+13000–U+1342F)
- **JSesh Integration**: 3,843 Unicode→JSesh mappings (99.8% coverage)

### Technology Stack
- **ML**: PyTorch, Detectron2, CUDA
- **Vision**: OpenCV, PIL, NumPy
- **Web**: Streamlit
- **Data**: Pandas, JSON
- **Standards**: Unicode, COCO Format

---

## Model Performance

### Training Setup
- **Platform**: Google Colab Pro (A100 GPU, 40GB VRAM)
- **Duration**: ~3 hours for 5,000 iterations
- **Architecture**: Faster R-CNN with ResNet-50 + FPN
- **Data**: 2,430 annotated hieroglyphs, 177 Gardiner classes
- **Splits**: 70% train, 20% validation, 10% test

### Performance Metrics
- **mAP@0.5**: 73%
- **mAP@0.5:0.95**: 41%  
- **Precision**: 78%
- **Recall**: 69%

### Key Challenges
- Class imbalance (60+ classes with <5 instances)
- Small sign detection (<20 pixels)
- Damaged papyrus regions

---

## Advanced Usage

### Command-Line Analysis Tool

```bash
# Basic usage
python scripts/hieroglyph_analysis_tool.py --image papyrus.jpg --confidence_threshold 0.5

# Batch processing
for image in *.jpg; do
    python scripts/hieroglyph_analysis_tool.py --image "$image" --output_dir "results"
done
```

### Web Application

```bash
# Launch Streamlit app
streamlit run apps/streamlit_app.py

# Enhanced paleography tool  
streamlit run apps/enhanced_paleography_tool.py
```

---

## API Reference

### Core Classes

#### HieroglyphAnalyzer

```python path=null start=null
from scripts.hieroglyph_analysis_tool import HieroglyphAnalyzer

# Initialize
analyzer = HieroglyphAnalyzer(
    model_path="models/hieroglyph_model_20250807_190054",
    confidence_threshold=0.5
)

# Analyze image
results = analyzer.analyze_image("papyrus.jpg")
```

#### JSesh Integration

```python path=null start=null
from scripts.jsesh_integration import JSeshIntegrator

# Complete notation
integrator = JSeshIntegrator()
notation = integrator.get_complete_notation('D4')
# Returns: gardiner_code, unicode_symbol, jsesh_code, description
```

### Configuration

```python path=null start=null
# Model settings
DEFAULT_CONFIDENCE = 0.5
MAX_IMAGE_SIZE = 4096
OUTPUT_FORMATS = ["json", "csv", "html"]
```

---

## Troubleshooting

### Installation Issues

**Detectron2 build failures:**
```bash path=null start=null
# Install dependencies first
pip install torch torchvision torchaudio

# Build from source
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**CUDA compatibility:**
```bash path=null start=null
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

**Memory errors:**
```python path=null start=null
# Use CPU inference
cfg.MODEL.DEVICE = "cpu"

# Or reduce input size
cfg.INPUT.MAX_SIZE_TEST = 800
```

**Unicode display problems:**
- Install Noto Sans Egyptian Hieroglyphs font
- Verify file encoding: `encoding='utf-8'`
- Check Unicode mappings exist for Gardiner codes

### Performance

**Slow detection:**
- Enable GPU acceleration: `cfg.MODEL.DEVICE = "cuda"`
- Optimize NMS threshold: `cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5`
- Reduce detections per image: `cfg.TEST.DETECTIONS_PER_IMAGE = 50`

**Web app timeouts:**
- Resize large images before processing
- Use progress indicators for user feedback

---

> For basic setup and usage, see [README.md](../README.md)
>
> For detailed examples and notebooks, see the project repository

---

For basic setup and usage instructions, see the main [README.md](../README.md).

For the latest updates and detailed changelog, see the [GitHub repository](https://github.com/margotbelot/PapyrusVision).
