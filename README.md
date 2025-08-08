# PapyrusNU: AI-Powered Hieroglyph Detection and Digital Paleography

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Detectron2](https://img.shields.io/badge/detectron2-latest-green.svg)](https://github.com/facebookresearch/detectron2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/margotbelot/PapyrusNU_Detectron.svg)](https://github.com/margotbelot/PapyrusNU_Detectron/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/margotbelot/PapyrusNU_Detectron.svg)](https://github.com/margotbelot/PapyrusNU_Detectron/network)

## Overview

PapyrusNU is an advanced computer vision system that uses deep learning to detect and analyze ancient Egyptian hieroglyphs in papyrus documents. The system combines state-of-the-art object detection with comprehensive Egyptological knowledge to provide researchers with powerful analysis capabilities.

## Key Features

- **🔍 Hieroglyph Detection**: AI-powered detection using Detectron2 framework
- **📜 Digital Paleography**: Automated cropping and cataloging of individual signs
- **🌐 Unicode Integration**: Official Unicode mappings for 594+ hieroglyphic signs
- **📊 Interactive Visualizations**: Comprehensive analysis and reporting tools
- **🎨 Web Interface**: User-friendly Streamlit application

## System Architecture

### Core Components

1. **Detection Model**: Detectron2-based object detection trained on hieroglyphic signs
2. **Unicode Mapping System**: Complete Gardiner sign list with Unicode integration
3. **Digital Paleography Tool**: Automated sign extraction and cataloging
4. **Web Interface**: Interactive Streamlit application for analysis

### Technology Stack

- **Deep Learning**: PyTorch, Detectron2
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Web Interface**: Streamlit
- **Standards**: Unicode Egyptian Hieroglyphs block

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/margotbelot/PapyrusNU_Detectron.git
cd PapyrusNU_Detectron

# Install dependencies
pip install -r requirements.txt

# Install Detectron2 (choose appropriate version for your system)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### Alternative Installation (CPU-only)

```bash
# For CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
pip install -r requirements.txt
```

## Usage

### 1. Interactive Web Application (Recommended)

**For most users - real-time analysis and exploration**

```bash
streamlit run streamlit_hieroglyphs_app.py
```

Open your browser to `http://localhost:8501` to access the interactive interface.

#### Web App Features:
- **🔍 Single Image Analysis**: Upload and analyze individual papyrus images
- **📜 Interactive Paleography**: Create sign catalogs with real-time preview
- **🎨 Visual Interface**: Drag-and-drop uploads, interactive visualizations
- **💾 Instant Downloads**: Export results as JSON, CSV, HTML, or ZIP packages
- **📊 Live Statistics**: Real-time analysis and progress tracking

**Best for**: Exploratory analysis, demonstrations, educational use, single-session work

### 2. Batch Processing Tool (For Research)

**For researchers and automation - large-scale processing**

```bash
# Process multiple images in a directory
python digital_paleography_tool.py
```

#### Command-Line Tool Features:
- **📁 Batch Processing**: Process entire directories of images automatically
- **🔄 Automation Ready**: Perfect for research pipelines and scripts
- **💾 Comprehensive Output**: Generates detailed reports and organized file structures
- **⚡ High Performance**: Optimized for processing many images efficiently
- **🛠️ Customizable**: Fine-tuned control over confidence thresholds and filtering

**Best for**: Large datasets, research workflows, automated processing, batch analysis

### 3. Individual Analysis Scripts

```bash
# Analyze a single image with detailed output
python scripts/hieroglyph_analysis_tool.py --image path/to/image.jpg
```

### 3. Jupyter Notebooks

Explore the analysis pipeline through interactive notebooks in the `notebooks/` directory:

1. `01_data_preparation.ipynb` - Dataset preparation and preprocessing
2. `02_data_analysis.ipynb` - Exploratory data analysis
3. `03_model_training.ipynb` - Model training and evaluation
4. `04_model_evaluation.ipynb` - Performance analysis
5. `05_model_predictions_visualization.ipynb` - Results visualization

## Project Structure

```
PapyrusNU_Detectron/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── GETTING_STARTED.md            # Quick start guide
├── PROJECT_REPORT.md             # Comprehensive technical report
├── streamlit_hieroglyphs_app.py  # Interactive web application (recommended)
├── digital_paleography_tool.py   # Batch processing tool (research workflows)
│
├── scripts/                       # Core analysis tools
│   ├── hieroglyph_analysis_tool.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   └── visualization.py
│
├── notebooks/                     # Jupyter analysis notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_predictions_visualization.ipynb
│
├── models/                        # Trained model files
│   └── hieroglyph_model_*/
│       ├── model_final.pth
│       ├── config.yaml
│       ├── model_info.json
│       └── *.png                  # Model analysis plots
│
├── data/                          # Dataset and annotations
│   ├── images/                    # Sample images
│   ├── annotations/               # COCO-format annotations
│   │   ├── train_annotations.json
│   │   ├── val_annotations.json
│   │   ├── gardiner_unicode_mapping.json
│   │   └── gardiner_descriptions.json
│   └── analysis_plots/            # Data analysis visualizations
│       ├── training_analysis.png      # Training performance analysis
│       ├── training_convergence.png   # Training convergence visualization
│       └── *.png, *.html, *.json      # Complete analysis suite
│
└── training_*.png                 # Training performance plots (root level)
```

## Model Performance

The trained model achieves:
- **mAP@0.5**: 0.73
- **mAP@0.5:0.95**: 0.41
- **Training Classes**: 50+ Gardiner sign categories
- **Unicode Coverage**: 594 official mappings

## Key Outputs

### 1. Detection Results
- Bounding boxes with confidence scores
- Gardiner code classification
- Unicode symbol mapping
- Exportable formats (JSON, CSV)

### 2. Digital Paleography
- Individual sign crops organized by Gardiner codes
- Interactive HTML catalog with Unicode symbols
- Comprehensive metadata and descriptions
- ZIP packages for offline use

### 3. Visualizations
- Training performance plots
- Data distribution analysis
- Detection confidence heatmaps
- Class distribution charts

## Academic Applications

- **Digital Humanities**: Automated analysis of historical texts
- **Egyptology**: Large-scale hieroglyphic corpus analysis
- **Computer Vision**: Object detection in historical documents
- **Cultural Heritage**: Digital preservation and accessibility

## Technical Details

### Model Architecture
- **Base**: Faster R-CNN with ResNet-50 backbone
- **Framework**: Detectron2
- **Training**: Transfer learning from COCO weights
- **Augmentation**: Rotation, scaling, color jittering

### Dataset
- **Format**: COCO annotation standard
- **Classes**: Gardiner sign list categories
- **Split**: 70% train, 20% validation, 10% test
- **Annotations**: Bounding boxes with class labels

## Contributing

This project was developed as part of academic research in digital humanities and computer vision. Contributions, bug reports, and improvements are welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Margot Belot**
- GitHub: [@margotbelot](https://github.com/margotbelot)
- Email: margot.belot@example.edu

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{papyrusnu2024,
  title={PapyrusNU: AI-Powered Hieroglyph Detection and Digital Paleography},
  author={Margot Belot},
  year={2024},
  url={https://github.com/margotbelot/PapyrusNU_Detectron}
}
```

## Bibliography & References

### Primary Sources
- **Digital Archive**: [Thesaurus Linguae Aegyptiae - Book of the Dead of Nu](https://tla.digital/object/7NOILVRXDVBPZBXA4S4FRIHTMA)
- **British Museum Collection**: [BM EA 10477](https://www.britishmuseum.org/collection/object/Y_EA10477-25)

### Scholarly Publications
- **Lapp, G.** (1997). *The Papyrus of Nu (BM EA 10477)*. Catalogue of Books of the Dead in the British Museum I. London: British Museum Press.
- **Taylor, J.H.** (2010). *Journey through the Afterlife: Ancient Egyptian Book of the Dead*. London: British Museum Press.

### Technical References
- **Detectron2**: He, K., et al. (2019). Detectron2. Facebook AI Research.
- **CVAT**: Computer Vision Annotation Tool. Intel Corporation.
- **Unicode Standard**: Unicode Egyptian Hieroglyphs Block (U+13000–U+1342F)

## Acknowledgments

- Detectron2 team at Facebook AI Research
- Unicode Consortium for Egyptian Hieroglyphs standard  
- Digital humanities and Egyptology research communities
- British Museum for digitization and open access to cultural heritage
- Thesaurus Linguae Aegyptiae project for digital resources

---

**For questions or support, please [open an issue](https://github.com/margotbelot/PapyrusNU_Detectron/issues) or contact margot.belot@example.edu**
