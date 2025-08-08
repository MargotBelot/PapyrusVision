"""
PapyrusNU: AI-Powered Hieroglyph Detection and Digital Paleography

This package provides comprehensive tools for analyzing ancient Egyptian hieroglyphs
using modern computer vision and deep learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Margot Belot"
__email__ = "margot.belot@example.edu"

# Core functionality imports
from .core.detector import HieroglyphDetector
from .core.paleography import DigitalPaleographyTool
from .core.unicode_mapping import UnicodeMapper

# Utility imports
from .utils.visualization import visualize_detections, create_analysis_plots
from .utils.dataset_utils import load_annotations, validate_dataset

__all__ = [
    "HieroglyphDetector",
    "DigitalPaleographyTool", 
    "UnicodeMapper",
    "visualize_detections",
    "create_analysis_plots",
    "load_annotations",
    "validate_dataset",
]
