# 🧠 CCFv3-Aligner: 3D DAPI Population Atlas & Deep Learning Alignment Studio

**High-Precision Interactive Brain Slice Registration & 15-Nuclei Deep Learning Segmentation Studio**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Atlas: 3D DAPI Population](https://img.shields.io/badge/Atlas-3D%20DAPI%20Population%20Template-green.svg)](https://github.com/AxisJu/CCFv3-aligner)
[![GUI: Tkinter + Matplotlib](https://img.shields.io/badge/GUI-Tkinter%20%2B%20Matplotlib-orange.svg)](https://matplotlib.org/)

`CCFv3-Aligner` is an advanced interactive brain registration and deep learning segmentation workstation designed for mouse whole-brain immunofluorescence (IF/IHC) slices and Whole Slide Imaging (WSI, such as Olympus `.vsi`, multi-channel `.tif`, and `.png`). 

It combines **3D DAPI Population Template matching** across the full anteroposterior (AP) axis with an automated **Dual-Channel Attention ResUNet** deep learning inference engine and **per-ROI directional edge-pinning**, resolving coronal tilt, asymmetric hemisphere stretch, and tissue shearing.

---

## 🌟 Key Features

### 1. 3D DAPI Population Template & Full AP Brain Extent
- **Full Coronal AP Coverage**: Continuous AP layer index from $Y = 0$ to $Y = 192$ (spanning Bregma $-4.00\text{ mm}$ to $+2.40\text{ mm}$), covering the anterior prefrontal cortex through posterior visual areas.
- **Bilateral Split-Hemisphere AP Selection**: Allows independent AP layer assignment for left ($Y_L$) and right ($Y_R$) hemispheres with a 1-click synchronization toggle (`🔗 Link L/R`).

### 2. 🤖 Dual-Channel Attention ResUNet Deep Learning Assistant
- **Automated 15-Nuclei Segmentation**: Integrated deep neural network featuring Squeeze-and-Excitation Channel Attention, Atrous Spatial Pyramid Pooling (ASPP), and Spatial Attention Gates.
- **Canonical Hemisphere Mirroring**: Splits brain slices into left and right hemispheres, performs canonical mirrored inference, and reassembles seamless full-slice multi-nuclei masks.
- **Dual-Channel Fluorescence Normalization**: Robust automated denoising and background subtraction for DAPI (Ch0) and NeuN (Ch1).
- **Live Interactive DL Overlay**: Instant visual feedback in the GUI studio for comparing AI predictions against template boundaries.

### 3. 🖥️ 3-Viewport High-Speed Comparative Studio
- **Viewport 1 (Top Left)**: Left hemisphere 3D DAPI population template at $Y_L$.
- **Viewport 2 (Bottom Left)**: Right hemisphere 3D DAPI population template at $Y_R$.
- **Viewport 3 (Right Column)**: Live experimental slice canvas with 4-view channel switcher (`Magma`, `Grayscale DAPI`, `Target GFP/IF`, `NeuN Cy5`).
- **60+ FPS Vectorized Transform**: Real-time smooth dragging, scaling, and midline shifting without rendering lag.

### 4. 🎯 Direct Point-and-Click ROI Pinning
- **Canvas Interaction**: Click directly on any nucleus or contour on the canvas to instantly focus and tune it.
- **Detached Move Handles**: Below-ROI label badges with drag handles for small or tightly clustered subcortical nuclei.
- **Bilateral Scale & Midline Adjustments**: Independent horizontal/vertical scaling anchored to dorsal surface peaks.
- **Region Exclusion / Restoral**: 1-click exclusion (`❌ Exclude ROI`) and full reset (`🔄 Restore All`).

### 5. 📊 4-in-1 Master Quality Control (QC) Export
- Generates publication-ready 2×2 Master QC diagnostic figures:
  - **Panel A**: 3D DAPI Population Template slice @ $Y_L / Y_R$.
  - **Panel B**: Dual Anatomical Overlay (Cyan: Experimental, Magenta: Template).
  - **Panel C**: 3D Spatial Warped Model Baseline.
  - **Panel D**: Final Transformed Target Subregions & Outlines.

---

## 🏗️ Repository Architecture

```text
CCFv3-aligner/
├── ccfv3_aligner/                   # Core Python package
│   ├── core/                        # Core registration & I/O
│   │   ├── atlas.py                 # 3D DAPI Population Template & CCFv3 Manager
│   │   ├── slide_io.py              # Universal WSI Reader (Olympus VSI, TIFF, PNG)
│   │   ├── preprocess.py            # Fluorescence normalization & tissue masking
│   │   └── landmark.py              # Midline & dorsal apex landmark estimation
│   ├── models/                      # Deep learning inference engine
│   │   ├── network.py               # Dual-Channel Attention ResUNet (ASPP + SE)
│   │   ├── predictor.py             # Hemisphere Mirroring & Assembly Predictor
│   │   ├── config.py                # 16-class ontology & Allen ID mappings
│   │   └── denoise.py               # Fluorescence channel filtering & smoothing
│   ├── gui/                         # Graphical user interface
│   │   ├── studio_window.py         # 3-viewport interactive studio (Tkinter+Matplotlib)
│   │   └── main_window.py           # Compatibility wrapper
│   └── utils/                       # Utility & export tools
│       ├── config.py                # Global color palette & region taxonomy
│       └── export_qc.py             # 4-in-1 Master QC figure generator
├── scripts/
│   ├── run_studio.py                # GUI Studio entrypoint
│   ├── predict_slice.py             # Deep learning CLI segmentation tool
│   ├── scan_manifest.py             # Cohort slide recursive scanner
│   └── batch_export.py              # Headless batch registration exporter
├── launch_gui_aligner-DAPI.bat      # Windows 1-Click Studio Launcher
├── launch_aligner.bat               # Windows Universal Launcher
├── run_studio.sh                    # Linux / macOS Bash Launcher
├── requirements.txt                 # Python dependency specifications
├── setup.py                         # Package installation setup
├── .gitignore                       # Git exclusion rules
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## ⚡ Installation

### 1. Conda Environment Setup

```bash
# Create and activate Python 3.10 environment
conda create -n ccfv3_env python=3.10 -y
conda activate ccfv3_env

# Clone repository
git clone https://github.com/AxisJu/CCFv3-aligner.git
cd CCFv3-aligner

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## 🚀 Quick Start

### 1. Launching the GUI Studio

- **Windows**: Double-click **`launch_gui_aligner-DAPI.bat`** (or `launch_aligner.bat`).
- **Command Line**:
  ```bash
  python scripts/run_studio.py --manifest path/to/cohort_manifest.csv --results path/to/output_dir
  ```

### 2. Running Deep Learning Segmentation (CLI)

To segment 15 core brain nuclei on a single image or an entire folder:

```bash
# Single slice segmentation
python scripts/predict_slice.py -i data/sample_slide.vsi -o results/dl_output

# Batch folder segmentation
python scripts/predict_slice.py -i /path/to/slides/ -o results/dl_batch
```

Output includes:
- `<slice>_pred_mask.png`: 16-class segmentation mask.
- `<slice>_pred_overlay.png`: 3-panel comparative diagnostic figure.
- `<slice>_region_stats.csv`: Quantification of pixel areas for all 15 nuclei.

---

## 🧠 Supported Brain Regions (16-Class Ontology)

| Index | Allen CCF ID | Acronym | Full Structure Name | Color Code |
| :---: | :---: | :---: | :--- | :---: |
| **0** | `0` | **Background** | Non-tissue / Background | `#000000` |
| **1** | `545` | **RSPd** | Retrosplenial Area (Dorsal) | `#ff1a71` |
| **2** | `463` | **CA3** | Hippocampal Field CA3 | `#860d4c` |
| **3** | `382` | **CA1** | Hippocampal Field CA1 | `#53082f` |
| **4** | `10704` | **DG** | Dentate Gyrus | `#16f2f2` |
| **5** | `1022` | **GPe** | Globus Pallidus External | `#b199ff` |
| **6** | `672` | **CP** | Caudoputamen (Striatum) | `#8d7acc` |
| **7** | `250` | **LSc** | Lateral Septal Nucleus (Caudal) | `#3283fe` |
| **8** | `262` | **RT** | Reticulated Nucleus of Thalamus | `#ff6600` |
| **9** | `581` | **TRS** | Triangular Nucleus of Septum | `#7609b1` |
| **10** | `483` | **MH** | Medial Habenula | `#faa307` |
| **11** | `186` | **LH** | Lateral Habenula | `#c68105` |
| **12** | `64` | **ATN** | Anterior Thalamic Nuclei (AD/AV/AM) | `#1460ff` |
| **13** | `181` | **RE** | Nucleus of Reuniens | `#1340ff` |
| **14** | `930` | **PF** | Parafascicular Nucleus | `#0a4093` |
| **15** | `599` | **CM** | Central Medial Thalamic Nucleus | `#08306d` |

---

## 🛠️ Configuration & Data Placement

### 3D DAPI Population Template
By default, the studio auto-detects `dapi_template.nii.gz` and `dapi_template_segmentation_full.nii.gz`. You can also configure a custom path via environment variable:
```bash
export CCFV3_DAPI_ATLAS_DIR="/path/to/dapi_atlas_paper"
```

### Deep Learning Model Checkpoints
Place your trained weights (`best_model.pth`) inside `models/` or set:
```bash
export CCFV3_MODEL_PATH="/path/to/best_model.pth"
```

---

## ⌨️ Keyboard & Mouse Shortcuts

- **`Enter`**: Save current alignment, export 4-in-1 Master QC, and advance to next slide.
- **`Left / Right Arrow`**: Jump to previous / next slide in cohort.
- **`Up / Down Arrow`**: Fine-tune position ($Y$-offset) of selected ROI.
- **`Delete`**: Exclude / hide selected ROI from annotation export.
- **`Left Click on Canvas`**: Focus on clicked nucleus and display transform handles.
- **`Mouse Drag`**: Translate active nucleus ROI across canvas.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments & Citation

If you use `CCFv3-Aligner` or the 3D DAPI Population Template in your research, please cite:
```bibtex
@software{ccfv3_aligner2026,
  author = {Axis Ju},
  title = {CCFv3-Aligner: High-Precision 3D DAPI Population Atlas & Deep Learning Brain Slice Registration Studio},
  year = {2026},
  url = {https://github.com/AxisJu/CCFv3-aligner}
}
```
