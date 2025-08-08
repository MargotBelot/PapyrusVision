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

# 🏺 Detectron2 Hieroglyphic Detection Training Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Detectron2](https://img.shields.io/badge/detectron2-latest-green.svg)](https://github.com/facebookresearch/detectron2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a complete, production-ready pipeline for training a Detectron2 model to perform instance segmentation and classification of Egyptian hieroglyphs from high-resolution papyrus images.

---

### 🏛️ Dataset Source: The Book of the Dead of Nu (BM EA 10477)

This pipeline is optimized for a high-resolution scan of a specific, historically significant artifact.

#### About the Papyrus

The **Book of the Dead of Nu** is a remarkable funerary papyrus from ancient Egypt's 18th Dynasty (c. 1550-1295 BCE). Nu was a royal scribe and steward who served under the pharaohs of the New Kingdom. This papyrus contains a collection of magical spells, prayers, and incantations designed to guide Nu safely through the afterlife and ensure his resurrection.

**Sheet 25** specifically features **Spell 145**, which describes the "Fields of Iaru" (the Egyptian version of paradise) and includes detailed vignettes showing the deceased in the afterlife. The text is written in hieroglyphic script using both black and red ink, with red used for emphasis and ritual instructions.

**Historical Significance:**
- **Artifact**: The Book of the Dead of Nu, sheet 25, featuring full-colour vignettes and text from Spell 145
- **Period**: 18th Dynasty, New Kingdom (c. 1550-1295 BCE)
- **Owner**: Nu, royal scribe and steward 
- **Current Location**: British Museum, London (EA 10477)
- **Image Reference**: [British Museum Collection](https://www.britishmuseum.org/collection/object/Y_EA10477-25)
- **Content**: 59 columns of hieroglyphic text with **2,431 manually annotated signs**

**Technical Details:**
- **Annotations**: 2,431 hieroglyphs labeled across **177 distinct Gardiner classes**
- **Platform**: Annotations created using [CVAT (Computer Vision Annotation Tool)](https://www.cvat.ai/)
- **Resolution**: High-resolution museum digitization suitable for detailed analysis

---

### ✨ Key Features

- **🏺 End-to-End Pipeline**: Four sequential notebooks guide you from data setup to final evaluation.
- **🛡️ Zero Data Leakage**: A **stratified spatial split** (70/20/10) is used on the single source image, ensuring annotations are not shared between train, validation, and test sets for reliable evaluation.
- **📊 Comprehensive Data Analysis**: Includes an entire notebook (`02_data_analysis.ipynb`) dedicated to deep exploratory data analysis with interactive plots to understand object size, aspect ratio, and spatial distribution.
- **🔄 Smart Augmentation**: The training pipeline uses a custom data mapper to apply augmentations like random flips, rotation, and color adjustments, which is critical for model generalization on a single-image dataset.
- **📈 Monitored Training**: The custom trainer logs detailed metrics (total loss, class loss, bbox loss, mask loss, learning rate) and performs periodic evaluation on the validation set.
- **🎨 Rich Reporting & Visualization**: Automatically generates and saves analysis plots, training curves, evaluation metrics, and visual comparisons of ground truth vs. predictions.

---

### 📂 Project Structure

```
PapyrusNU_Detectron/
├── data/
│   ├── images/              # Contains the 145_upscaled_bright.jpg image.
│   └── annotations/         # Stores all COCO JSON annotation files.
│       ├── annotations.json     # Original, raw annotation file.
│       ├── train_annotations.json # 70% of annotations for training.
│       ├── val_annotations.json   # 20% for validation during training.
│       └── test_annotations.json  # 10% for final, unbiased evaluation.
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Validates, splits, and prepares the dataset.
│   ├── 02_data_analysis.ipynb     # Performs deep exploratory data analysis (EDA).
│   ├── 03_model_training.ipynb      # Trains the Detectron2 model.
│   └── 04_model_evaluation.ipynb    # Evaluates the trained model on the test set.
├── scripts/
│   ├── dataset_utils.py         # Core logic for data splitting and preparation.
│   ├── visualization.py         # Utilities for creating all plots and visualizations.
│   └── evaluation.py            # Logic for computing performance metrics.
└── models/                      # Directory where trained models and outputs are saved.
```

---

### 🚀 Google Colab Tutorial

Follow these steps to run the entire pipeline in Google Colab.

#### Step 1: Upload Your Project to Google Drive

1.  **Zip the Project:** On your local machine, create a single ZIP file named `PapyrusNU_Detectron.zip` from the project folder.
2.  **Upload to Drive:** Upload this `PapyrusNU_Detectron.zip` file to the main (root) directory of your Google Drive.

#### Step 2: Set Up the Colab Environment

1.  **Open Colab:** Go to [colab.research.google.com](https://colab.research.google.com) and create a **New notebook**.
2.  **Set GPU Runtime:** In the Colab menu, navigate to `Runtime` -> `Change runtime type` and select **T4 GPU** from the dropdown menu. A GPU is essential for training.

#### Step 3: Mount Drive and Unzip the Project

Copy and paste the following code block into a cell in your Colab notebook and run it. This will connect to your Google Drive and extract the project files.

```python
from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Path to the ZIP file in your Google Drive
zip_path = '/content/drive/My Drive/PapyrusNU_Detectron.zip'
extract_path = '/content/'

if os.path.exists(zip_path):
    print(f"Found ZIP file at: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Project unzipped successfully to: {extract_path}PapyrusNU_Detectron")
else:
    print(f"❌ Error: ZIP file not found at {zip_path}.")
    print("Please ensure you have uploaded 'PapyrusNU_Detectron.zip' to the root of your Google Drive.")
```

#### Step 4: Run the Notebooks in Sequence

Use the file browser on the left side of Colab to navigate into the unzipped folder: `/content/PapyrusNU_Detectron/notebooks/`. Open and run each notebook in the following order.

1.  **Run `01_data_preparation.ipynb`**
    *   **What it does:** Loads the raw annotations, performs the 70/20/10 split, saves the new `train/val/test` JSON files, and registers the datasets with Detectron2.
    *   **Action:** Click `Runtime` -> `Run all`.

2.  **Run `02_data_analysis.ipynb`**
    *   **What it does:** Creates detailed, interactive plots to give you a deep understanding of the dataset's properties.
    *   **Action:** Click `Runtime` -> `Run all`.

3.  **Run `03_model_training.ipynb`**
    *   **What it does:** Configures the Mask R-CNN model, applies data augmentation, and starts the training process. This is the most time-consuming step.
    *   **Action:** Click `Runtime` -> `Run all`. Monitor the output to see the training loss decrease.

4.  **Run `04_model_evaluation.ipynb`**
    *   **What it does:** Loads your newly trained model and runs a full evaluation on the hold-out test set, generating final metrics and visualizations.
    *   **Action:** Click `Runtime` -> `Run all`.

#### Step 5: Review and Save Your Results

All outputs are saved within the `/content/PapyrusNU_Detectron/models/` directory. Each training run creates a new subfolder named with a timestamp (e.g., `hieroglyph_model_20250807_160100`). Inside, you will find:

-   `model_final.pth`: The trained model weights.
-   `config.yaml`: The model configuration file.
-   `training_history.json`: A log of all training metrics.
-   `*.png`: All plots and visualizations generated during the pipeline.
-   `comprehensive_evaluation_report.json`: The final report with all performance metrics.

To save these results permanently, copy the model output folder back to your Google Drive by running this code in a Colab cell:

```python
import shutil

# Find the latest model directory
latest_model_dir = sorted(os.listdir('/content/PapyrusNU_Detectron/models'))[-1]
full_path = f'/content/PapyrusNU_Detectron/models/{latest_model_dir}'

# Copy it to your Google Drive
shutil.copytree(full_path, f'/content/drive/My Drive/{latest_model_dir}')

print(f"✅ Successfully copied {latest_model_dir} to your Google Drive!")
```
