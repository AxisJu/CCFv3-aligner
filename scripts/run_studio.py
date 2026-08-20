# -*- coding: utf-8 -*-
"""
scripts/run_studio.py
Main GUI Entrypoint for APX100 3D DAPI Population Alignment Studio & AI Assistant.
"""

import os
import sys
import argparse
import tkinter as tk

# Ensure package root is in python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ccfv3_aligner.gui.studio_window import DAPIStudioWindow


def main():
    parser = argparse.ArgumentParser(description="Launch APX100 3D DAPI Interactive Alignment Studio")
    parser.add_argument("--manifest", type=str, default=None, help="Path to cohort_manifest.csv")
    parser.add_argument("--results", type=str, default=None, help="Output root directory for results, manual_alignment and QC")
    parser.add_argument("--model", type=str, default=None, help="Path to deep learning model checkpoint (.pth)")
    args = parser.parse_args()

    default_results = r"D:\2024_DBSeq\results\260814_Rsrc1-Sf3b1 IF"
    results_dir = args.results or (default_results if os.path.exists(default_results) else os.getcwd())

    default_manifest = os.path.join(results_dir, "slice_manifest", "cohort_manifest.csv")
    manifest_path = args.manifest or (default_manifest if os.path.exists(default_manifest) else None)

    root = tk.Tk()
    app = DAPIStudioWindow(root, results_root=results_dir, cohort_manifest_path=manifest_path, model_path=args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
