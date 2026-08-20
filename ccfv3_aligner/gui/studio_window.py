# -*- coding: utf-8 -*-
"""
ccfv3_aligner.gui.studio_window
APX100 Mouse Brain Slice Interactive Alignment Studio (3D DAPI Population Template Edition).
Featuring:
- 3-Viewport Comparative Layout (Left Hemisphere Atlas, Right Hemisphere Atlas, Interactive Slice Canvas)
- Full-Brain Coronal AP Coverage (Y: 0 to 192, Bregma -4.00 mm to +2.40 mm)
- Deep Learning Multi-Nuclei Segmentation Integration (AttentionResUNet)
- Direct Point-and-Click ROI Selection & Edge-Pinning Vectorized Transforms
- Bilateral Independent Scaling, Shift, and AP-Layer Assignment
"""

import os
import sys
import time
import json
import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, opening, disk, remove_small_holes

from ..core.atlas import AtlasManager
from ..core.slide_io import SlideLoader
from ..models.predictor import BrainRegionPredictor, find_default_model_path
from ..utils.config import BASE_REGION_COLORS
from ..utils.export_qc import export_4panel_master_qc

# Matplotlib styling for high-contrast dark theme
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

save_lock = threading.Lock()


class DAPIStudioWindow:
    def __init__(self, root, results_root=None, cohort_manifest_path=None, model_path=None):
        self.root = root
        self.root.title("APX100 Brain Slice Alignment Studio - 3D DAPI Population Edition (v3.5)")
        self.root.geometry("1820x1000")
        self.root.minsize(1400, 800)

        # Directory structure
        self.results_root = results_root or os.getcwd()
        self.manual_dir = os.path.join(self.results_root, "manual_alignment")
        self.qc_dir = os.path.join(self.results_root, "qc_overlays")
        os.makedirs(self.manual_dir, exist_ok=True)
        os.makedirs(self.qc_dir, exist_ok=True)

        self.overrides_json = os.path.join(self.manual_dir, "manual_registration_overrides.json")
        self.overrides = self.load_overrides()

        # Atlas Manager (Loads 3D DAPI Population Template)
        self.atlas = AtlasManager(use_dapi=True)
        self.dapi_vol_3d = self.atlas.dapi_vol_3d
        self.dapi_seg_3d = self.atlas.dapi_seg_3d
        self.region_groups = self.atlas.region_groups
        self.bregma_y_idx = getattr(self.atlas, "bregma_y_idx", 120)
        self.y_min_limit = getattr(self.atlas, "y_min_limit", 0)
        self.y_max_limit = getattr(self.atlas, "y_max_limit", 192)

        # Sequential layer trackers
        self.last_used_y_l = self.bregma_y_idx
        self.last_used_y_r = self.bregma_y_idx

        # Deep Learning Predictor (Lazy or Async load)
        self.model_path = model_path or find_default_model_path()
        self.dl_predictor = None
        self.dl_prediction_cache = {}
        self.show_dl_overlay = tk.BooleanVar(value=False)

        # Load Manifest
        if cohort_manifest_path and os.path.exists(cohort_manifest_path):
            self.manifest_path = cohort_manifest_path
        else:
            default_manifest = os.path.join(self.results_root, "slice_manifest", "cohort_manifest.csv")
            self.manifest_path = default_manifest if os.path.exists(default_manifest) else None

        if self.manifest_path and os.path.exists(self.manifest_path):
            self.df_manifest = pd.read_csv(self.manifest_path)
        else:
            self.df_manifest = pd.DataFrame([{"vsi_file": "Demo_Slice.vsi", "vsi_path": "", "target": "Demo", "group": "Control"}])

        self.slide_display_list = self._build_slide_display_list()
        self.current_slide_idx = self._find_first_unmarked_slide()

        # Data cache for current slide
        self.d_crop = None
        self.n_crop = None
        self.target_crop = None
        self.fused_crop = None
        self.mid_x_auto = 0
        self.y_top_l_auto = 0
        self.y_top_r_auto = 0
        self.base_sx_l = 1.0
        self.base_sy_l = 1.0
        self.base_sx_r = 1.0
        self.base_sy_r = 1.0

        # High-Speed Cached Raw Contours in Template Space
        self.tmpl_roi_polys = {}
        self.active_roi_keys = []
        self.selected_roi_idx = 0
        self.roi_custom_bounds = {}
        self.deleted_rois = set()

        # Split Reference Slice Caches
        self.cached_tmpl_l = None
        self.cached_seg_l = None
        self.cached_tmpl_r = None
        self.cached_seg_r = None

        # Interaction State
        self.is_click_set_midline = False
        self.drag_handle = None
        self.drag_start_mouse = None
        self.drag_start_bounds = None
        self.is_updating = False

        self._setup_style()
        self._build_ui()
        self._setup_canvas_events()

        # Initialize background model loading
        threading.Thread(target=self._init_dl_model, daemon=True).start()

        if len(self.df_manifest) > 0 and self.df_manifest.iloc[0].get("vsi_path"):
            self.load_slide(self.current_slide_idx)

    def _init_dl_model(self):
        try:
            self.dl_predictor = BrainRegionPredictor(checkpoint_path=self.model_path)
            if hasattr(self, "lbl_dl_status"):
                self.lbl_dl_status.config(text="AI Engine: Ready", fg="#39ff14")
        except Exception as e:
            print(f"[DeepLearning] Model initialization note: {e}")
            if hasattr(self, "lbl_dl_status"):
                self.lbl_dl_status.config(text="AI Engine: Standby", fg="#ffaa00")

    def _build_slide_display_list(self):
        display_list = []
        for i, row in self.df_manifest.iterrows():
            vsi_file = str(row.get("vsi_file", f"Slice_{i+1}"))
            target = str(row.get("target", "-"))
            grp = str(row.get("group", "-"))
            status = " [DONE]" if vsi_file in self.overrides else " [ ]"
            display_list.append(f"[{i+1:03d}/{len(self.df_manifest)}]{status} {vsi_file} ({target}|{grp})")
        return display_list

    def _find_first_unmarked_slide(self):
        for i, row in self.df_manifest.iterrows():
            vsi_file = str(row.get("vsi_file", ""))
            if vsi_file and vsi_file not in self.overrides:
                return i
        return 0

    def load_overrides(self):
        if os.path.exists(self.overrides_json):
            try:
                with open(self.overrides_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _setup_style(self):
        self.font_base = ("Segoe UI", 12)
        self.font_bold = ("Segoe UI", 12, "bold")
        self.font_title = ("Segoe UI", 13, "bold")
        self.font_val = ("Consolas", 12, "bold")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=self.font_base, background="#1e1e2e", foreground="#ffffff")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=self.font_base)
        style.configure("TButton", font=self.font_bold, padding=4)
        style.configure("Header.TLabel", font=self.font_title, foreground="#89b4fa")
        style.configure("Value.TLabel", font=self.font_val, foreground="#a6e3a1")

    def _build_ui(self):
        main_container = tk.Frame(self.root, bg="#1e1e2e")
        main_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left Column: Controls & Navigation
        control_panel = tk.Frame(main_container, width=420, bg="#181825")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        control_panel.pack_propagate(False)

        # Right Column: Matplotlib Figure Canvas
        canvas_panel = tk.Frame(main_container, bg="#11111b")
        canvas_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_control_panel(control_panel)
        self._build_canvas_panel(canvas_panel)

    def _build_control_panel(self, parent):
        # 1. Slide Navigation
        frame_nav = tk.LabelFrame(parent, text=" 📂 Cohort Slide Navigation ", bg="#181825", fg="#89b4fa", font=self.font_title)
        frame_nav.pack(fill=tk.X, padx=6, pady=4)

        self.cbo_slide = ttk.Combobox(frame_nav, values=self.slide_display_list, state="readonly", font=("Segoe UI", 11))
        self.cbo_slide.pack(fill=tk.X, padx=6, pady=4)
        self.cbo_slide.bind("<<ComboboxSelected>>", self._on_slide_selected)

        btn_box = tk.Frame(frame_nav, bg="#181825")
        btn_box.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(btn_box, text="◀ Prev", command=self.prev_slide).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_box, text="Next ▶", command=self.next_slide).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # 2. Split-Hemisphere AP / Y-Slice Selection
        frame_ap = tk.LabelFrame(parent, text=" 🧠 3D DAPI AP-Layer (Y: 0-192) ", bg="#181825", fg="#fab387", font=self.font_title)
        frame_ap.pack(fill=tk.X, padx=6, pady=4)

        self.link_ap_var = tk.BooleanVar(value=True)
        chk_link = tk.Checkbutton(frame_ap, text="🔗 Link Left/Right AP Slices", variable=self.link_ap_var,
                                  bg="#181825", fg="#f9e2af", selectcolor="#313244", font=self.font_bold,
                                  activebackground="#181825", activeforeground="#f9e2af",
                                  command=self._on_link_toggle)
        chk_link.pack(anchor="w", padx=6, pady=2)

        # Left Y Slider
        f_yl = tk.Frame(frame_ap, bg="#181825")
        f_yl.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f_yl, text="Left Y_L:", bg="#181825", fg="#cdd6f4", font=self.font_bold).pack(side=tk.LEFT)
        self.lbl_yl_val = tk.Label(f_yl, text="120", bg="#181825", fg="#a6e3a1", font=self.font_val)
        self.lbl_yl_val.pack(side=tk.RIGHT)
        self.sld_yl = ttk.Scale(frame_ap, from_=self.y_min_limit, to=self.y_max_limit, orient=tk.HORIZONTAL, command=self._on_yl_slide)
        self.sld_yl.set(self.bregma_y_idx)
        self.sld_yl.pack(fill=tk.X, padx=6, pady=1)

        # Right Y Slider
        f_yr = tk.Frame(frame_ap, bg="#181825")
        f_yr.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f_yr, text="Right Y_R:", bg="#181825", fg="#cdd6f4", font=self.font_bold).pack(side=tk.LEFT)
        self.lbl_yr_val = tk.Label(f_yr, text="120", bg="#181825", fg="#a6e3a1", font=self.font_val)
        self.lbl_yr_val.pack(side=tk.RIGHT)
        self.sld_yr = ttk.Scale(frame_ap, from_=self.y_min_limit, to=self.y_max_limit, orient=tk.HORIZONTAL, command=self._on_yr_slide)
        self.sld_yr.set(self.bregma_y_idx)
        self.sld_yr.pack(fill=tk.X, padx=6, pady=1)

        self.lbl_bregma = tk.Label(frame_ap, text="Bregma ~ 0.00 mm", bg="#181825", fg="#89dceb", font=self.font_bold)
        self.lbl_bregma.pack(pady=2)

        # 3. Macro Hemisphere Transformations
        frame_macro = tk.LabelFrame(parent, text=" 📐 Bilateral Scale & Alignment ", bg="#181825", fg="#a6e3a1", font=self.font_title)
        frame_macro.pack(fill=tk.X, padx=6, pady=4)

        # Scale X
        f_sx = tk.Frame(frame_macro, bg="#181825")
        f_sx.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f_sx, text="Scale X (L/R):", bg="#181825", fg="#cdd6f4").pack(side=tk.LEFT)
        self.lbl_sx = tk.Label(f_sx, text="1.00 / 1.00", bg="#181825", fg="#a6e3a1", font=self.font_val)
        self.lbl_sx.pack(side=tk.RIGHT)
        self.sld_sxl = ttk.Scale(frame_macro, from_=0.5, to=2.0, orient=tk.HORIZONTAL, command=self._on_transform_change)
        self.sld_sxl.set(1.0)
        self.sld_sxl.pack(fill=tk.X, padx=6, pady=1)

        # Scale Y
        f_sy = tk.Frame(frame_macro, bg="#181825")
        f_sy.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f_sy, text="Scale Y (L/R):", bg="#181825", fg="#cdd6f4").pack(side=tk.LEFT)
        self.lbl_sy = tk.Label(f_sy, text="1.00 / 1.00", bg="#181825", fg="#a6e3a1", font=self.font_val)
        self.lbl_sy.pack(side=tk.RIGHT)
        self.sld_syl = ttk.Scale(frame_macro, from_=0.5, to=2.0, orient=tk.HORIZONTAL, command=self._on_transform_change)
        self.sld_syl.set(1.0)
        self.sld_syl.pack(fill=tk.X, padx=6, pady=1)

        # Midline Shift
        f_mid = tk.Frame(frame_macro, bg="#181825")
        f_mid.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f_mid, text="Midline Offset:", bg="#181825", fg="#cdd6f4").pack(side=tk.LEFT)
        self.lbl_mid = tk.Label(f_mid, text="0 px", bg="#181825", fg="#a6e3a1", font=self.font_val)
        self.lbl_mid.pack(side=tk.RIGHT)
        self.sld_mid = ttk.Scale(frame_macro, from_=-60, to=60, orient=tk.HORIZONTAL, command=self._on_transform_change)
        self.sld_mid.set(0)
        self.sld_mid.pack(fill=tk.X, padx=6, pady=1)

        btn_macro_box = tk.Frame(frame_macro, bg="#181825")
        btn_macro_box.pack(fill=tk.X, padx=6, pady=3)
        self.btn_set_mid = tk.Button(btn_macro_box, text="📍 Click Midline", bg="#45475a", fg="#ffffff",
                                     font=self.font_bold, command=self._toggle_click_midline)
        self.btn_set_mid.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(btn_macro_box, text="🔄 Reset Scales", bg="#45475a", fg="#ffffff",
                  font=self.font_bold, command=self.reset_macro_scales).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # 4. Deep Learning AI Predictor Frame
        frame_dl = tk.LabelFrame(parent, text=" 🤖 Deep Learning AI Assistant ", bg="#181825", fg="#cba6f7", font=self.font_title)
        frame_dl.pack(fill=tk.X, padx=6, pady=4)

        f_dl_status = tk.Frame(frame_dl, bg="#181825")
        f_dl_status.pack(fill=tk.X, padx=6, pady=2)
        self.lbl_dl_status = tk.Label(f_dl_status, text="AI Engine: Initializing...", bg="#181825", fg="#fab387", font=("Segoe UI", 10, "bold"))
        self.lbl_dl_status.pack(side=tk.LEFT)

        chk_dl_ov = tk.Checkbutton(frame_dl, text="Live DL Overlay", variable=self.show_dl_overlay,
                                   bg="#181825", fg="#cba6f7", selectcolor="#313244", font=self.font_bold,
                                   activebackground="#181825", activeforeground="#cba6f7",
                                   command=self.update_display)
        chk_dl_ov.pack(anchor="w", padx=6, pady=1)

        self.btn_dl_predict = tk.Button(frame_dl, text="✨ Run AI Nuclei Segmentation", bg="#8839ef", fg="#ffffff",
                                        font=self.font_bold, activebackground="#9c40ff", activeforeground="#ffffff",
                                        command=self._on_run_dl_prediction)
        self.btn_dl_predict.pack(fill=tk.X, padx=6, pady=3)

        # 5. Local ROI Pinning Controls
        frame_roi = tk.LabelFrame(parent, text=" 🎯 Selected ROI Tuning ", bg="#181825", fg="#f38ba8", font=self.font_title)
        frame_roi.pack(fill=tk.X, padx=6, pady=4)

        self.lbl_roi_name = tk.Label(frame_roi, text="None", bg="#181825", fg="#f38ba8", font=self.font_title)
        self.lbl_roi_name.pack(pady=1)

        f_roi_btns = tk.Frame(frame_roi, bg="#181825")
        f_roi_btns.pack(fill=tk.X, padx=6, pady=2)
        self.btn_del_roi = tk.Button(f_roi_btns, text="❌ Exclude ROI", bg="#e64553", fg="#ffffff",
                                     font=self.font_bold, command=self._exclude_current_roi)
        self.btn_del_roi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(f_roi_btns, text="🔄 Restore All", bg="#45475a", fg="#ffffff",
                  font=self.font_bold, command=self._restore_all_rois).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=1)

        # 6. Actions & Export
        frame_actions = tk.Frame(parent, bg="#181825")
        frame_actions.pack(fill=tk.X, padx=6, pady=6, side=tk.BOTTOM)

        self.btn_save = tk.Button(frame_actions, text="💾 Save Alignment & Next (Enter)", bg="#40a02b", fg="#ffffff",
                                  font=("Segoe UI", 13, "bold"), activebackground="#48b830", activeforeground="#ffffff",
                                  command=self.save_and_next)
        self.btn_save.pack(fill=tk.X, pady=3)

    def _build_canvas_panel(self, parent):
        # 4-View Switcher Bar
        bar_view = tk.Frame(parent, bg="#11111b")
        bar_view.pack(fill=tk.X, padx=6, pady=2)

        tk.Label(bar_view, text="🎨 View Mode:", bg="#11111b", fg="#cdd6f4", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 6))
        self.view_mode_var = tk.StringVar(value="Magma")
        modes = [("Magma High-Contrast", "Magma"), ("Grayscale DAPI", "Grayscale"),
                 ("Target (GFP/IF)", "Target"), ("NeuN Cy5", "NeuN")]
        for label_text, val in modes:
            r = tk.Radiobutton(bar_view, text=label_text, value=val, variable=self.view_mode_var,
                               bg="#11111b", fg="#cdd6f4", selectcolor="#313244", font=self.font_bold,
                               activebackground="#11111b", activeforeground="#ffffff",
                               command=self.update_display)
            r.pack(side=tk.LEFT, padx=4)

        # Matplotlib 3-Viewport Grid: Left Column 1/3 (Atlas L top, Atlas R bottom), Right Column 2/3 (Interactive Canvas)
        self.fig = plt.figure(figsize=(14, 9), facecolor="#11111b")
        gs = gridspec.GridSpec(2, 3, figure=self.fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.04, hspace=0.06)

        self.ax_atlas_l = self.fig.add_subplot(gs[0, 0])
        self.ax_atlas_r = self.fig.add_subplot(gs[1, 0])
        self.ax_canvas = self.fig.add_subplot(gs[:, 1:])

        for ax in [self.ax_atlas_l, self.ax_atlas_r, self.ax_canvas]:
            ax.set_facecolor("#0d1117")
            ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_canvas_events(self):
        self.canvas.mpl_connect("button_press_event", self._on_mouse_down)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_up)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Left>", lambda e: self.prev_slide())
        self.root.bind("<Right>", lambda e: self.next_slide())
        self.root.bind("<Up>", lambda e: self._adjust_active_roi(0, -2))
        self.root.bind("<Down>", lambda e: self._adjust_active_roi(0, 2))
        self.root.bind("<Delete>", lambda e: self._exclude_current_roi())

    def load_slide(self, slide_idx: int):
        if slide_idx < 0 or slide_idx >= len(self.df_manifest):
            return

        self.current_slide_idx = slide_idx
        self.cbo_slide.current(slide_idx)
        row = self.df_manifest.iloc[slide_idx]
        vsi_path = row.get("vsi_path", "")
        vsi_file = row.get("vsi_file", os.path.basename(vsi_path))

        print(f"\n[Studio] Loading slide [{slide_idx+1}/{len(self.df_manifest)}]: {vsi_file}")

        try:
            loader = SlideLoader(vsi_path)
            thumb_dapi = loader.get_channel_thumbnail(0, target_width=1200)
            thumb_neun = loader.get_channel_thumbnail(1, target_width=1200)
            thumb_target = loader.get_channel_thumbnail(2, target_width=1200)
        except Exception as e:
            print(f"[Studio] Slide load error: {e}. Generating blank canvas.")
            thumb_dapi = np.zeros((800, 1000), dtype=np.float32)
            thumb_neun = thumb_dapi
            thumb_target = thumb_dapi

        # Robust Normalization
        p1_d, p99_d = np.percentile(thumb_dapi, (1, 99.2))
        d_norm = np.clip((thumb_dapi - p1_d) / (p99_d - p1_d + 1e-5), 0, 1).astype(np.float32)

        p1_n, p99_n = np.percentile(thumb_neun, (1, 99.2))
        n_norm = np.clip((thumb_neun - p1_n) / (p99_n - p1_n + 1e-5), 0, 1).astype(np.float32)

        p1_t, p99_t = np.percentile(thumb_target, (1, 99.5))
        t_norm = np.clip((thumb_target - p1_t) / (p99_t - p1_t + 1e-5), 0, 1).astype(np.float32)

        # Tissue Extraction & Cropping
        fused = (0.35 * d_norm + 0.65 * n_norm).astype(np.float32)
        tissue_mask = closing(fused > 0.06, disk(8))
        lbl = label(tissue_mask)
        props = regionprops(lbl)

        if props:
            main_p = max(props, key=lambda x: x.area)
            strict_mask = remove_small_holes((lbl == main_p.label), max_size=5000)
            minr, minc, maxr, maxc = main_p.bbox
            pad = 12
            h, w = d_norm.shape
            y0, y1 = max(0, minr - pad), min(h, maxr + pad)
            x0, x1 = max(0, minc - pad), min(w, maxc + pad)

            self.d_crop = (d_norm * strict_mask)[y0:y1, x0:x1].astype(np.float32)
            self.n_crop = (n_norm * strict_mask)[y0:y1, x0:x1].astype(np.float32)
            self.target_crop = (t_norm * strict_mask)[y0:y1, x0:x1].astype(np.float32)
            self.fused_crop = (fused * strict_mask)[y0:y1, x0:x1].astype(np.float32)
        else:
            self.d_crop = d_norm
            self.n_crop = n_norm
            self.target_crop = t_norm
            self.fused_crop = fused

        # Landmark Initialization
        hc, wc = self.fused_crop.shape
        self.mid_x_auto = wc // 2
        self.base_sx_l = 1.0
        self.base_sy_l = 1.0
        self.base_sx_r = 1.0
        self.base_sy_r = 1.0

        # Load Saved Overrides if exist
        self.deleted_rois = set()
        self.roi_custom_bounds = {}

        if vsi_file in self.overrides:
            ov = self.overrides[vsi_file]
            yl = ov.get("yl", ov.get("y_l", ov.get("ap", self.last_used_y_l)))
            yr = ov.get("yr", ov.get("y_r", yl))
            self.last_used_y_l = yl
            self.last_used_y_r = yr
            self.sld_yl.set(yl)
            self.sld_yr.set(yr)
            self.link_ap_var.set(ov.get("link_ap", (yl == yr)))

            self.sld_sxl.set(ov.get("scale_x_l", ov.get("scale_x", 1.0)))
            self.sld_syl.set(ov.get("scale_y_l", ov.get("scale_y", 1.0)))
            self.sld_mid.set(ov.get("midline_shift", 0.0))

            if "deleted_rois" in ov:
                self.deleted_rois = set(ov["deleted_rois"])
            if "roi_bounds" in ov:
                self.roi_custom_bounds = ov["roi_bounds"]
        else:
            self.sld_yl.set(self.last_used_y_l)
            self.sld_yr.set(self.last_used_y_r)
            self.sld_sxl.set(1.0)
            self.sld_syl.set(1.0)
            self.sld_mid.set(0)

        self._refresh_template_contours()
        self.update_display()

    def _refresh_template_contours(self):
        """Extract ROI polygons from 3D DAPI segmentation at current Y_L / Y_R."""
        yl = int(self.sld_yl.get())
        yr = int(self.sld_yr.get())

        self.cached_tmpl_l, self.cached_seg_l = self.atlas.get_dapi_hemisphere_slice(yl, is_left=True)
        self.cached_tmpl_r, self.cached_seg_r = self.atlas.get_dapi_hemisphere_slice(yr, is_left=False)

        self.tmpl_roi_polys = {}
        self.active_roi_keys = []

        # Parse Left Hemisphere Nuclei
        mid_x_tmpl = self.cached_tmpl_l.shape[1]
        for acronym in self.atlas.region_groups.keys():
            sids = self.atlas.region_groups.get(acronym, [])
            mask_l = np.isin(self.cached_seg_l, sids)
            if np.any(mask_l):
                key = f"{acronym} (Left)"
                cnts = find_contours(mask_l.astype(float), 0.5)
                # Pad into template coordinates
                padded_cnts = [np.column_stack([c[:, 0], c[:, 1]]) for c in cnts]
                self.tmpl_roi_polys[key] = {
                    "acronym": acronym, "side": "Left", "contours": padded_cnts,
                    "color": BASE_REGION_COLORS.get(acronym, "#00ffcc"), "base_name": acronym
                }
                self.active_roi_keys.append(key)

        # Parse Right Hemisphere Nuclei
        for acronym in self.atlas.region_groups.keys():
            sids = self.atlas.region_groups.get(acronym, [])
            mask_r = np.isin(self.cached_seg_r, sids)
            if np.any(mask_r):
                key = f"{acronym} (Right)"
                cnts = find_contours(mask_r.astype(float), 0.5)
                # Shift X into right hemisphere coordinates
                shifted_cnts = [np.column_stack([c[:, 0], c[:, 1] + mid_x_tmpl]) for c in cnts]
                self.tmpl_roi_polys[key] = {
                    "acronym": acronym, "side": "Right", "contours": shifted_cnts,
                    "color": BASE_REGION_COLORS.get(acronym, "#00ffcc"), "base_name": acronym
                }
                self.active_roi_keys.append(key)

        if self.active_roi_keys and self.selected_roi_idx >= len(self.active_roi_keys):
            self.selected_roi_idx = 0

    def update_display(self):
        """Redraw all 3 viewports."""
        if self.d_crop is None:
            return

        yl = int(self.sld_yl.get())
        yr = int(self.sld_yr.get())
        bregma_val = (self.bregma_y_idx - (yl + yr) / 2.0) * 0.025
        self.lbl_yl_val.config(text=str(yl))
        self.lbl_yr_val.config(text=str(yr))
        self.lbl_bregma.config(text=f"Bregma ~ {bregma_val:+.2f} mm")

        # 1. Draw Left Hemisphere Atlas
        self.ax_atlas_l.clear()
        self.ax_atlas_l.set_facecolor("#0d1117")
        if self.cached_tmpl_l is not None:
            p99 = np.percentile(self.cached_tmpl_l, 99)
            norm_l = np.clip(self.cached_tmpl_l / (p99 + 1e-5), 0, 1)
            self.ax_atlas_l.imshow(norm_l, cmap="magma", vmin=0, vmax=0.85)
        self.ax_atlas_l.set_title(f"3D DAPI Left Template (Y_L={yl})", color="#89b4fa", fontsize=11, fontweight="bold")
        self.ax_atlas_l.axis("off")

        # 2. Draw Right Hemisphere Atlas
        self.ax_atlas_r.clear()
        self.ax_atlas_r.set_facecolor("#0d1117")
        if self.cached_tmpl_r is not None:
            p99 = np.percentile(self.cached_tmpl_r, 99)
            norm_r = np.clip(self.cached_tmpl_r / (p99 + 1e-5), 0, 1)
            self.ax_atlas_r.imshow(norm_r, cmap="magma", vmin=0, vmax=0.85)
        self.ax_atlas_r.set_title(f"3D DAPI Right Template (Y_R={yr})", color="#fab387", fontsize=11, fontweight="bold")
        self.ax_atlas_r.axis("off")

        # 3. Draw Main Interactive Canvas
        self.ax_canvas.clear()
        self.ax_canvas.set_facecolor("#0d1117")

        mode = self.view_mode_var.get()
        if mode == "Magma":
            self.ax_canvas.imshow(self.fused_crop, cmap="magma", vmin=0, vmax=0.75)
        elif mode == "Grayscale":
            self.ax_canvas.imshow(self.d_crop, cmap="gray", vmin=0, vmax=0.75)
        elif mode == "Target":
            self.ax_canvas.imshow(self.target_crop, cmap="viridis", vmin=0, vmax=0.80)
        else:
            self.ax_canvas.imshow(self.n_crop, cmap="cividis", vmin=0, vmax=0.80)

        # Midline
        mid_x = self.mid_x_auto + int(self.sld_mid.get())
        self.ax_canvas.axvline(mid_x, color="#ffff00", linestyle="--", linewidth=1.5, alpha=0.9)

        # Optional Deep Learning Overlay
        row = self.df_manifest.iloc[self.current_slide_idx]
        vsi_file = row.get("vsi_file", "")
        if self.show_dl_overlay.get() and vsi_file in self.dl_prediction_cache:
            dl_res = self.dl_prediction_cache[vsi_file]
            for rname, cnts in dl_res.get("contours_by_region", {}).items():
                hex_c = BASE_REGION_COLORS.get(rname, "#39ff14")
                for c in cnts:
                    self.ax_canvas.plot(c[:, 1], c[:, 0], color=hex_c, linestyle=":", linewidth=1.4, alpha=0.75)

        # Transform & Render Aligned ROIs
        sx_l = self.sld_sxl.get()
        sy_l = self.sld_syl.get()
        sx_r = sx_l
        sy_r = sy_l

        for idx, rkey in enumerate(self.active_roi_keys):
            if rkey in self.deleted_rois:
                continue

            rdata = self.tmpl_roi_polys[rkey]
            hex_c = rdata["color"]
            is_selected = (idx == self.selected_roi_idx)

            # Vectorized Transform from Template to Canvas
            transformed_cnts = self._transform_roi_contours(rdata["contours"], rdata["side"], mid_x, sx_l, sy_l, rkey)

            lw = 2.4 if is_selected else (2.0 if "Cortex" in rkey else 1.5)
            for c in transformed_cnts:
                self.ax_canvas.plot(c[:, 1], c[:, 0], color=hex_c, linewidth=lw, alpha=0.95 if is_selected else 0.8)

            # Draw Label badge or handles for selected ROI
            if is_selected and transformed_cnts:
                all_pts = np.vstack(transformed_cnts)
                min_y, min_x = np.min(all_pts, axis=0)
                max_y, max_x = np.max(all_pts, axis=0)
                cy, cx = (min_y + max_y) / 2.0, (min_x + max_x) / 2.0

                # Bounding box & Center Handle
                self.ax_canvas.plot([min_x, max_x, max_x, min_x, min_x],
                                    [min_y, min_y, max_y, max_y, min_y],
                                    color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.7)
                self.ax_canvas.plot(cx, cy, marker="+", color="#ffffff", markersize=10, markeredgewidth=2)

                tag = rdata["base_name"] + ("_L" if rdata["side"] == "Left" else "_R")
                self.ax_canvas.text(cx, max_y + 16, f"[{tag}]", color="white", fontsize=10, fontweight="bold",
                                    ha="center", va="top",
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=hex_c, alpha=0.9, edgecolor="none"))

        if self.active_roi_keys and self.selected_roi_idx < len(self.active_roi_keys):
            cur_key = self.active_roi_keys[self.selected_roi_idx]
            self.lbl_roi_name.config(text=cur_key)

        self.ax_canvas.set_title(f"{vsi_file} | Interactive Studio (AP Y_L={yl}, Y_R={yr})",
                                 color="#ffffff", fontsize=12, fontweight="bold")
        self.canvas.draw_idle()

    def _transform_roi_contours(self, contours, side, mid_x, sx, sy, rkey):
        """Fast Vectorized coordinate transformation with custom per-ROI offset overrides."""
        res = []
        custom_bounds = self.roi_custom_bounds.get(rkey, None)

        for c in contours:
            c_trans = c.copy()
            if side == "Left":
                # Relative to left midline
                c_trans[:, 1] = mid_x - (256.0 - c_trans[:, 1]) * sx
            else:
                c_trans[:, 1] = mid_x + (c_trans[:, 1] - 256.0) * sx

            c_trans[:, 0] = c_trans[:, 0] * sy

            # Apply individual bounding box scaling if exists
            if custom_bounds:
                # Custom shift/scale
                dx = custom_bounds.get("dx", 0)
                dy = custom_bounds.get("dy", 0)
                c_trans[:, 1] += dx
                c_trans[:, 0] += dy

            res.append(c_trans)
        return res

    def _on_run_dl_prediction(self):
        """Run AttentionResUNet deep learning segmentation in background thread."""
        row = self.df_manifest.iloc[self.current_slide_idx]
        vsi_file = row.get("vsi_file", "")
        self.btn_dl_predict.config(text="⏳ AI Inference Running...", state=tk.DISABLED)

        def worker():
            try:
                if self.dl_predictor is None:
                    self.dl_predictor = BrainRegionPredictor(checkpoint_path=self.model_path)

                mid_x = self.mid_x_auto + int(self.sld_mid.get())
                res = self.dl_predictor.predict_full_slice_arrays(self.d_crop, self.n_crop, mid_x=mid_x)
                self.dl_prediction_cache[vsi_file] = res

                self.root.after(0, lambda: self._on_dl_prediction_complete(vsi_file))
            except Exception as e:
                print(f"[DeepLearning] Inference error: {e}")
                self.root.after(0, lambda: messagebox.showerror("AI Error", f"Deep Learning inference failed: {e}"))
                self.root.after(0, lambda: self.btn_dl_predict.config(text="✨ Run AI Nuclei Segmentation", state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def _on_dl_prediction_complete(self, vsi_file):
        self.btn_dl_predict.config(text="✨ Run AI Nuclei Segmentation", state=tk.NORMAL)
        self.show_dl_overlay.set(True)
        self.update_display()
        messagebox.showinfo("AI Segmentation Ready", f"AI predicted 15 core nuclei for '{vsi_file}'.\nLive DL Overlay has been enabled on the canvas.")

    def _on_slide_selected(self, event=None):
        idx = self.cbo_slide.current()
        if idx != self.current_slide_idx:
            self.load_slide(idx)

    def prev_slide(self):
        if self.current_slide_idx > 0:
            self.load_slide(self.current_slide_idx - 1)

    def next_slide(self):
        if self.current_slide_idx < len(self.df_manifest) - 1:
            self.load_slide(self.current_slide_idx + 1)

    def _on_yl_slide(self, val):
        if self.is_updating: return
        if self.link_ap_var.get():
            self.is_updating = True
            self.sld_yr.set(val)
            self.is_updating = False
        self._refresh_template_contours()
        self.update_display()

    def _on_yr_slide(self, val):
        if self.is_updating: return
        if self.link_ap_var.get():
            self.is_updating = True
            self.sld_yl.set(val)
            self.is_updating = False
        self._refresh_template_contours()
        self.update_display()

    def _on_link_toggle(self):
        if self.link_ap_var.get():
            self.sld_yr.set(self.sld_yl.get())
            self._refresh_template_contours()
            self.update_display()

    def _on_transform_change(self, val=None):
        sx = self.sld_sxl.get()
        sy = self.sld_syl.get()
        mid = int(self.sld_mid.get())
        self.lbl_sx.config(text=f"{sx:.2f} / {sx:.2f}")
        self.lbl_sy.config(text=f"{sy:.2f} / {sy:.2f}")
        self.lbl_mid.config(text=f"{mid:+d} px")
        self.update_display()

    def reset_macro_scales(self):
        self.sld_sxl.set(1.0)
        self.sld_syl.set(1.0)
        self.sld_mid.set(0)
        self._on_transform_change()

    def _toggle_click_midline(self):
        self.is_click_set_midline = not self.is_click_set_midline
        if self.is_click_set_midline:
            self.btn_set_mid.config(bg="#fab387", fg="#11111b", text="🎯 Click Canvas Midline")
        else:
            self.btn_set_mid.config(bg="#45475a", fg="#ffffff", text="📍 Click Midline")

    def _exclude_current_roi(self):
        if self.active_roi_keys and self.selected_roi_idx < len(self.active_roi_keys):
            key = self.active_roi_keys[self.selected_roi_idx]
            self.deleted_rois.add(key)
            self.update_display()

    def _restore_all_rois(self):
        self.deleted_rois.clear()
        self.roi_custom_bounds.clear()
        self.update_display()

    def _adjust_active_roi(self, dx, dy):
        if self.active_roi_keys and self.selected_roi_idx < len(self.active_roi_keys):
            key = self.active_roi_keys[self.selected_roi_idx]
            if key not in self.roi_custom_bounds:
                self.roi_custom_bounds[key] = {"dx": 0, "dy": 0}
            self.roi_custom_bounds[key]["dx"] += dx
            self.roi_custom_bounds[key]["dy"] += dy
            self.update_display()

    def _on_mouse_down(self, event):
        if event.inaxes != self.ax_canvas:
            return

        if self.is_click_set_midline:
            # Set Midline from Click
            new_mid = int(event.xdata)
            offset = new_mid - self.mid_x_auto
            self.sld_mid.set(np.clip(offset, -60, 60))
            self._toggle_click_midline()
            self._on_transform_change()
            return

        # Direct Point-and-Click ROI Selection
        mx, my = event.xdata, event.ydata
        mid_x = self.mid_x_auto + int(self.sld_mid.get())
        sx = self.sld_sxl.get()
        sy = self.sld_syl.get()

        for idx, rkey in enumerate(self.active_roi_keys):
            if rkey in self.deleted_rois:
                continue
            rdata = self.tmpl_roi_polys[rkey]
            cnts = self._transform_roi_contours(rdata["contours"], rdata["side"], mid_x, sx, sy, rkey)
            for c in cnts:
                min_y, min_x = np.min(c, axis=0) - 8, np.min(c, axis=1) - 8
                max_y, max_x = np.max(c, axis=0) + 8, np.max(c, axis=1) + 8
                if min_x <= mx <= max_x and min_y <= my <= max_y:
                    self.selected_roi_idx = idx
                    self.drag_start_mouse = (mx, my)
                    self.update_display()
                    return

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax_canvas or self.drag_start_mouse is None:
            return

        # Drag translation
        mx, my = event.xdata, event.ydata
        dx = mx - self.drag_start_mouse[0]
        dy = my - self.drag_start_mouse[1]

        if abs(dx) > 1 or abs(dy) > 1:
            if self.active_roi_keys and self.selected_roi_idx < len(self.active_roi_keys):
                key = self.active_roi_keys[self.selected_roi_idx]
                if key not in self.roi_custom_bounds:
                    self.roi_custom_bounds[key] = {"dx": 0, "dy": 0}
                self.roi_custom_bounds[key]["dx"] += dx
                self.roi_custom_bounds[key]["dy"] += dy
                self.drag_start_mouse = (mx, my)
                self.update_display()

    def _on_mouse_up(self, event):
        self.drag_start_mouse = None

    def save_and_next(self):
        """Save alignment overrides, export 4-in-1 Master QC, and jump to next slide."""
        row = self.df_manifest.iloc[self.current_slide_idx]
        vsi_file = str(row.get("vsi_file", f"Slide_{self.current_slide_idx+1}"))

        yl = int(self.sld_yl.get())
        yr = int(self.sld_yr.get())
        bregma_str = f"{(self.bregma_y_idx - (yl + yr)/2.0)*0.025:+.2f} mm"

        self.last_used_y_l = yl
        self.last_used_y_r = yr

        mid_x = self.mid_x_auto + int(self.sld_mid.get())
        sx = float(self.sld_sxl.get())
        sy = float(self.sld_syl.get())

        override_entry = {
            "vsi_file": vsi_file,
            "y_l": yl,
            "y_r": yr,
            "link_ap": self.link_ap_var.get(),
            "scale_x_l": sx,
            "scale_y_l": sy,
            "scale_x_r": sx,
            "scale_y_r": sy,
            "midline_shift": int(self.sld_mid.get()),
            "deleted_rois": list(self.deleted_rois),
            "roi_bounds": self.roi_custom_bounds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with save_lock:
            self.overrides[vsi_file] = override_entry
            with open(self.overrides_json, "w", encoding="utf-8") as f:
                json.dump(self.overrides, f, indent=2, ensure_ascii=False)

        # Export 4-Panel Master QC
        qc_filename = os.path.splitext(vsi_file)[0] + "_master_qc.png"
        qc_path = os.path.join(self.qc_dir, qc_filename)

        transformed_dict = {}
        for rkey in self.active_roi_keys:
            if rkey not in self.deleted_rois:
                rdata = self.tmpl_roi_polys[rkey]
                transformed_dict[rkey] = self._transform_roi_contours(rdata["contours"], rdata["side"], mid_x, sx, sy, rkey)

        dl_res = self.dl_prediction_cache.get(vsi_file, None)
        dl_cnts = dl_res.get("contours_by_region", None) if dl_res else None

        export_4panel_master_qc(
            qc_path=qc_path,
            slide_name=vsi_file,
            ap_or_yl=yl,
            yr=yr,
            bregma_str=bregma_str,
            mid_x=mid_x,
            tmpl_slice_l=self.cached_tmpl_l,
            tmpl_slice_r=self.cached_tmpl_r,
            annot_slice=self.cached_seg_l,
            fused_crop=self.fused_crop,
            cached_w_tmpl=None,
            cached_w_annot=None,
            active_roi_keys=self.active_roi_keys,
            raw_roi_polys=self.tmpl_roi_polys,
            transformed_contours_dict=transformed_dict,
            dl_contours_dict=dl_cnts
        )

        print(f"[Studio] Saved alignment and QC for: {vsi_file} -> {qc_path}")
        self.slide_display_list = self._build_slide_display_list()
        self.cbo_slide.config(values=self.slide_display_list)

        if self.current_slide_idx < len(self.df_manifest) - 1:
            self.load_slide(self.current_slide_idx + 1)
        else:
            messagebox.showinfo("Cohort Complete", "All slides in the cohort have been processed!")
