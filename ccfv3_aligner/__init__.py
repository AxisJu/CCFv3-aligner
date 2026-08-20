# -*- coding: utf-8 -*-
"""
CCFv3-Aligner: High-Precision 3D DAPI Population Template & Deep Learning Brain Slice Studio
"""

__version__ = "3.5.0"
__author__ = "Axis Ju"

from .core.atlas import AtlasManager, AllenCCFAtlas
from .core.slide_io import SlideLoader, APX100Slide
from .models.predictor import BrainRegionPredictor
from .gui.studio_window import DAPIStudioWindow
from .gui.main_window import MainWindow

__all__ = [
    "AtlasManager",
    "AllenCCFAtlas",
    "SlideLoader",
    "APX100Slide",
    "BrainRegionPredictor",
    "DAPIStudioWindow",
    "MainWindow"
]
