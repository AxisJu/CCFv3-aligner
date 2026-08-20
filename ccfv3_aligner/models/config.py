# -*- coding: utf-8 -*-
"""
ccfv3_aligner.models.config
Configuration, Class Definitions, and Hyperparameters for Deep Learning Segmentation.
"""

import os
import torch

# 16-Class Ontology (Background + 15 Target Core Nuclei, excluding Cortex)
CLASSES = [
    {"idx": 0,  "allen_id": 0,     "acronym": "Background", "name": "Background", "color": "#000000"},
    {"idx": 1,  "allen_id": 545,   "acronym": "RSPd",       "name": "Retrosplenial Area (Dorsal)", "color": "#ff1a71"},
    {"idx": 2,  "allen_id": 463,   "acronym": "CA3",        "name": "Hippocampal Field CA3", "color": "#860d4c"},
    {"idx": 3,  "allen_id": 382,   "acronym": "CA1",        "name": "Hippocampal Field CA1", "color": "#53082f"},
    {"idx": 4,  "allen_id": 10704, "acronym": "DG",         "name": "Dentate Gyrus", "color": "#16f2f2"},
    {"idx": 5,  "allen_id": 1022,  "acronym": "GPe",        "name": "Globus Pallidus External", "color": "#b199ff"},
    {"idx": 6,  "allen_id": 672,   "acronym": "CP",         "name": "Caudoputamen (Striatum)", "color": "#8d7acc"},
    {"idx": 7,  "allen_id": 250,   "acronym": "LSc",        "name": "Lateral Septal Nucleus (Caudal)", "color": "#3283fe"},
    {"idx": 8,  "allen_id": 262,   "acronym": "RT",         "name": "Reticular Nucleus of Thalamus", "color": "#ff6600"},
    {"idx": 9,  "allen_id": 581,   "acronym": "TRS",        "name": "Triangular Nucleus of Septum", "color": "#7609b1"},
    {"idx": 10, "allen_id": 483,   "acronym": "MH",         "name": "Medial Habenula", "color": "#faa307"},
    {"idx": 11, "allen_id": 186,   "acronym": "LH",         "name": "Lateral Habenula", "color": "#c68105"},
    {"idx": 12, "allen_id": 64,    "acronym": "ATN",        "name": "Anterior Thalamic Nuclei", "color": "#1460ff"},
    {"idx": 13, "allen_id": 181,   "acronym": "RE",         "name": "Nucleus of Reuniens", "color": "#1340ff"},
    {"idx": 14, "allen_id": 930,   "acronym": "PF",         "name": "Parafascicular Nucleus", "color": "#0a4093"},
    {"idx": 15, "allen_id": 599,   "acronym": "CM",         "name": "Central Medial Thalamic Nucleus", "color": "#08306d"},
]

NUM_CLASSES = len(CLASSES)  # 16 classes (0..15)
ALLEN_TO_IDX = {c["allen_id"]: c["idx"] for c in CLASSES if c["allen_id"] > 0}
IDX_TO_ALLEN = {c["idx"]: c["allen_id"] for c in CLASSES}
IDX_TO_ACRONYM = {c["idx"]: c["acronym"] for c in CLASSES}
IDX_TO_COLOR = {c["idx"]: c["color"] for c in CLASSES}
ACRONYM_TO_IDX = {c["acronym"]: c["idx"] for c in CLASSES}

# Hardware & Model Defaults
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 2          # DAPI (Ch0) + NeuN (Ch1)
TARGET_SIZE = (512, 384) # (Height, Width) for Canonical Half-Brain

DEFAULT_MODEL_FILENAME = "best_model.pth"
