# PapyrusVision - Quick Start Guide


## Installation (5 minutes)

1. **Install Conda and Set Up Python Environment (Recommended)**
    - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight) or [Anaconda](https://www.anaconda.com/products/distribution) (full distribution) if not already installed.
    - **Install Miniconda (macOS/Linux):**
       ```bash
       wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
       bash miniconda.sh
       # Follow the prompts, then restart your terminal
       ```
    - **Verify Conda installation:**
       ```bash
       conda --version
       ```
    - **Create and activate a Python 3.8+ environment:**
       ```bash
       conda create -n papyrusvision python=3.8
       conda activate papyrusvision
       ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Detectron2:**

   **If you are using Apple Silicon (M1/M2, macOS ARM):**
   Official Detectron2 wheels are not available for macOS ARM. You must build from source:
   ```bash
   pip install torch torchvision torchaudio
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

   **If you are using Linux or Windows (x86_64):**
   ```bash
   # For GPU (recommended)
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

   # For CPU only
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
   ```

## Run the Application (30 seconds)

```bash
streamlit run apps/streamlit_hieroglyphs_app.py
```

Open your browser to `http://localhost:8501`

## Two Ways to Use PapyrusVision

### Option 1: Web Application (Recommended for Most Users)

**Interactive interface - best for exploration and single-session work**

- **Single Image Analysis**: Upload and analyze individual papyrus images
- **Interactive Paleography**: Create sign catalogs with real-time preview  
- **Visual Interface**: Drag-and-drop, live statistics, instant downloads
- **Perfect for**: Demonstrations, education, exploratory analysis

### Option 2: Batch Processing Tool (For Research)

**Command-line interface - best for large-scale research**

```bash
python apps/digital_paleography_tool.py
```

- **Bulk Processing**: Handle entire directories of images automatically
- **Research Workflows**: Integrate into automated analysis pipelines  
- **Advanced Control**: Fine-tune confidence thresholds and filtering
- **Perfect for**: Large datasets, batch analysis, research automation

### Data Analysis & Training
- Explore Jupyter notebooks in `notebooks/`
- View training performance plots  
- Analyze dataset statistics

## Key Files

- `apps/streamlit_hieroglyphs_app.py` - Main web application
- `apps/digital_paleography_tool.py` - Paleography generator
- `scripts/hieroglyph_analysis_tool.py` - Core detection tool
- `models/` - Trained Detectron2 model
- `data/` - Dataset and analysis plots
- `notebooks/` - Complete training pipeline

## Quick Demo

The app includes a default test image for immediate demonstration. No additional setup required!

---

**Need help?** Check the full README.md for detailed documentation.
