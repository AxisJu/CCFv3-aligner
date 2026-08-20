# -*- coding: utf-8 -*-
"""
ccfv3_aligner.gui.main_window
Compatibility wrapper for MainWindow -> DAPIStudioWindow.
"""

from .studio_window import DAPIStudioWindow

class MainWindow(DAPIStudioWindow):
    """
    Standard Main Window for CCFv3-Aligner Studio (3D DAPI Population Edition).
    """
    pass

__all__ = ["MainWindow", "DAPIStudioWindow"]
