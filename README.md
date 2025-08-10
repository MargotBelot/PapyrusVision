# PapyrusVision: AI-Powered Hieroglyph Detection and Digital Paleography

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Detectron2](https://img.shields.io/badge/detectron2-latest-green.svg)](https://github.com/facebookresearch/detectron2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/margotbelot/PapyrusVision.svg)](https://github.com/margotbelot/PapyrusVision/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/margotbelot/PapyrusVision.svg)](https://github.com/margotbelot/PapyrusVision/network)

## Overview

PapyrusVision is a computer vision system that uses deep learning to detect and analyze ancient Egyptian hieroglyphs in papyrus documents. The system combines state-of-the-art object detection with comprehensive Egyptological knowledge to provide researchers with powerful analysis capabilities.

**Quick Demo**: The app includes a sample papyrus image for immediate demonstration - no setup required!

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Clone and navigate
git clone https://github.com/margotbelot/PapyrusVision.git
cd PapyrusVision

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Detectron2

**macOS Apple Silicon (M1/M2):**
```bash
pip install torch torchvision torchaudio
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Linux/Windows x86_64:**
```bash
# GPU (recommended)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# CPU only
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

### 3. Run the Web Application

```bash
streamlit run apps/streamlit_hieroglyphs_app.py
```

Open your browser to `http://localhost:8501` 🎉

## Features

- **🔍 Hieroglyph Detection**: AI-powered detection using Detectron2 framework
- **📜 Digital Paleography**: Automated cropping and cataloging of individual signs  
- **🏺 Unicode Integration**: Official Unicode mappings for 594+ hieroglyphic signs
- **📊 Interactive Visualizations**: Comprehensive analysis and reporting tools
- **🌐 Web Interface**: User-friendly drag-and-drop application
- **📚 Research Tools**: Batch processing for large datasets

## Two Usage Options

### 🌐 Web Application (Recommended)
**Perfect for**: Exploration, demonstrations, education, single-session work

- **Interactive Detection**: Upload and analyze individual images with real-time feedback
- **Visual Paleography**: Create sign catalogs with live preview and statistics  
- **User-Friendly**: Drag-and-drop interface, instant downloads, progress tracking

### ⚙️ Command-Line Tool
**Perfect for**: Large datasets, research workflows, automated processing

```bash
# Batch processing
python apps/digital_paleography_tool.py

# Individual analysis  
python scripts/hieroglyph_analysis_tool.py --image path/to/image.jpg
```

**Features**: Bulk processing, research pipelines, advanced control, customizable thresholds

## Model Performance

The trained Detectron2 model achieves:
- **mAP@0.5**: 0.73
- **mAP@0.5:0.95**: 0.41  
- **Training Classes**: 178 Gardiner sign categories
- **Unicode Coverage**: 594+ official mappings

## Project Structure

```
PapyrusVision/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── apps/                             # User applications  
│   ├── streamlit_hieroglyphs_app.py  # 🌐 Web interface (recommended)
│   └── digital_paleography_tool.py   # ⚙️ Batch processing
├── scripts/                          # Core analysis tools
├── notebooks/                        # 📓 Jupyter analysis pipeline
├── models/                           # 🤖 Trained Detectron2 model
├── data/                            # 📊 Dataset and annotations
└── docs/                            # 📚 Technical documentation
    └── TECHNICAL_GUIDE.md           # Complete technical details
```

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

## Technical Details

For complete technical documentation, training details, and advanced usage, see [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{papyrusvision2024,
  title={PapyrusVision: AI-Powered Hieroglyph Detection and Digital Paleography},
  author={Margot Belot},
  year={2024},
  url={https://github.com/margotbelot/PapyrusVision}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author & Support

**Margot Belot**
- GitHub: [@margotbelot](https://github.com/margotbelot)  
- Email: margotbelot@icloud.com

For questions or support, please [open an issue](https://github.com/margotbelot/PapyrusVision/issues) or contact margotbelot@icloud.com

---

## Bibliography & References

### Primary Sources
- **Digital Archive**: [Thesaurus Linguae Aegyptiae - Book of the Dead of Nu](https://tla.digital/object/7NOILVRXDVBPZBXA4S4FRIHTMA)
- **British Museum Collection**: [BM EA 10477](https://www.britishmuseum.org/collection/object/Y_EA10477-25)

### Technical References  
- **Detectron2**: He, K., et al. (2019). Detectron2. Facebook AI Research.
- **Unicode Standard**: Unicode Egyptian Hieroglyphs Block (U+13000–U+1342F)

### Acknowledgments
- Detectron2 team at Facebook AI Research
- Unicode Consortium for Egyptian Hieroglyphs standard
- Digital humanities and Egyptology research communities  
- British Museum for digitization and open access to cultural heritage
- Thesaurus Linguae Aegyptiae project for digital resources
- **High Performance**: Optimized for processing many images efficiently
- **Customizable**: Fine-tuned control over confidence thresholds and filtering

### 3. Individual Analysis Scripts

```bash
# Analyze a single image with detailed output
python scripts/hieroglyph_analysis_tool.py --image path/to/image.jpg
```

### 4. Jupyter Notebooks

Explore the analysis pipeline through interactive notebooks in the `notebooks/` directory:

1. `01_data_preparation.ipynb` - Dataset preparation and preprocessing
2. `02_data_analysis.ipynb` - Exploratory data analysis
3. `03_model_training.ipynb` - Model training and evaluation
4. `04_model_evaluation.ipynb` - Performance analysis
5. `05_model_predictions_visualization.ipynb` - Results visualization


## Project Structure

```
PapyrusVision/
 README.md                      # This file
 requirements.txt               # Python dependencies
 
 apps/                          # User applications
    streamlit_hieroglyphs_app.py  # Interactive web application (recommended)
    digital_paleography_tool.py   # Batch processing tool
 
 docs/                          # Documentation
    GETTING_STARTED.md          # Quick start guide
    PROJECT_REPORT.md           # Comprehensive technical report
    COMPLETE_PROJECT_DOCUMENTATION.md # Full project documentation
    STRUCTURE_OPTIMIZATION_PLAN.md    # Structure optimization plan

 scripts/                       # Core analysis tools
    hieroglyph_analysis_tool.py
    dataset_utils.py
    evaluation.py
    visualization.py

 notebooks/                     # Jupyter analysis notebooks
    01_data_preparation.ipynb
    02_data_analysis.ipynb
    03_model_training.ipynb
    04_model_evaluation.ipynb
    05_model_predictions_visualization.ipynb

 models/                        # Trained model files
    hieroglyph_model_*/
        model_final.pth
        config.yaml
        model_info.json
        *.png                  # Model analysis plots

 data/                          # Dataset and annotations
    images/                    # Sample images
    annotations/               # COCO-format annotations
       train_annotations.json
       val_annotations.json
       gardiner_unicode_mapping.json
       gardiner_descriptions.json
    analysis_plots/            # Data analysis visualizations
        training_analysis.png      # Training performance analysis
        training_convergence.png   # Training convergence visualization
        *.png, *.html, *.json      # Complete analysis suite

 training_*.png                 # Training performance plots
```

## Model Performance

The trained model achieves:
- **mAP@0.5**: 0.73
- **mAP@0.5:0.95**: 0.41
- **Training Classes**: 178 Gardiner sign categories
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
- Metadata and descriptions
- ZIP packages for offline use


## Documentation

All detailed documentation has been moved to the `docs/` folder for clarity:

- [Quick Start Guide](docs/GETTING_STARTED.md)
- [Project Report](docs/PROJECT_REPORT.md)
- [Complete Project Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md)
- [Structure Optimization Plan](docs/STRUCTURE_OPTIMIZATION_PLAN.md)

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
- Email: margotbelot@icloud.com

## Citation

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

---

**For questions or support, please [open an issue](https://github.com/margotbelot/PapyrusNU_Detectron/issues) or contact margotbelot@icloud.com**
