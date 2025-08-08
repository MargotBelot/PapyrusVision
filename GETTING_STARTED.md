# 🚀 Quick Start Guide

## Installation (5 minutes)

1. **Install Python 3.8+** if not already installed

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Detectron2:**
   ```bash
   # For GPU (recommended)
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
   
   # For CPU only
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
   ```

## Run the Application (30 seconds)

```bash
streamlit run streamlit_hieroglyphs_app.py
```

Open your browser to `http://localhost:8501`

## Two Ways to Use PapyrusNU

### 🌐 Option 1: Web Application (Recommended for Most Users)

**Interactive interface - best for exploration and single-session work**

- **🔍 Single Image Analysis**: Upload and analyze individual papyrus images
- **📜 Interactive Paleography**: Create sign catalogs with real-time preview  
- **🎨 Visual Interface**: Drag-and-drop, live statistics, instant downloads
- **💡 Perfect for**: Demonstrations, education, exploratory analysis

### ⚡ Option 2: Batch Processing Tool (For Research)

**Command-line interface - best for large-scale research**

```bash
python digital_paleography_tool.py
```

- **📁 Bulk Processing**: Handle entire directories of images automatically
- **🔄 Research Workflows**: Integrate into automated analysis pipelines  
- **🛠️ Advanced Control**: Fine-tune confidence thresholds and filtering
- **💡 Perfect for**: Large datasets, batch analysis, research automation

### 📊 Data Analysis & Training
- Explore Jupyter notebooks in `notebooks/`
- View training performance plots  
- Analyze dataset statistics

## Key Files

- `streamlit_enhanced_app.py` - Main web application
- `digital_paleography_tool.py` - Paleography generator
- `scripts/hieroglyph_analysis_tool.py` - Core detection tool
- `models/` - Trained Detectron2 model
- `data/` - Dataset and analysis plots
- `notebooks/` - Complete training pipeline

## Quick Demo

The app includes a default test image for immediate demonstration. No additional setup required!

---

**Need help?** Check the full README.md for detailed documentation.
