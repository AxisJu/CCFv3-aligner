# -*- coding: utf-8 -*-
"""
ccfv3_aligner.utils.export_qc
Comprehensive 4-Panel (2x2) Master Quality Control Figure Generator (English Edition).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy.ndimage import center_of_mass
from .config import BASE_REGION_COLORS


def export_4panel_master_qc(qc_path, slide_name, ap_or_yl, yr, bregma_str, mid_x,
                            tmpl_slice_l, tmpl_slice_r, annot_slice, fused_crop,
                            cached_w_tmpl, cached_w_annot,
                            active_roi_keys, raw_roi_polys, transformed_contours_dict,
                            dl_contours_dict=None):
    """
    Renders and exports a 2x2 Master QC Figure in High-Resolution Dark Theme:
    Panel A: Standard 3D DAPI Population Template @ Y_L / Y_R
    Panel B: Dual Anatomical Overlay (Cyan: Experimental Slice, Magenta: DAPI Template)
    Panel C: 3D DAPI Spatial Warped Model Baseline
    Panel D: Final Transformed Target Subregions & Cortex/Midline Segmentation (with optional DL prediction overlay)
    """
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    hc, wc = fused_crop.shape
    fig_qc, axes_qc = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d1117")
    plt.subplots_adjust(wspace=0.04, hspace=0.08, left=0.01, right=0.99, top=0.93, bottom=0.02)

    # --- Panel A: Reference DAPI Template ---
    tmpl_combined = np.hstack([tmpl_slice_l, tmpl_slice_r]) if tmpl_slice_r is not None else tmpl_slice_l
    p99 = np.percentile(tmpl_combined, 99)
    tmpl_norm = np.clip(tmpl_combined / (p99 + 1e-5), 0, 1)
    axes_qc[0, 0].imshow(tmpl_norm, cmap="gray", vmin=0, vmax=0.85)

    if annot_slice is not None:
        vent_mask = np.isin(annot_slice, [145, 129, 60])
        for c in find_contours(vent_mask.astype(float), 0.5):
            axes_qc[0, 0].plot(c[:, 1], c[:, 0], color="#00ffff", linewidth=1.5)

    mid_tmpl = tmpl_combined.shape[1] // 2
    axes_qc[0, 0].axvline(mid_tmpl, color="#ffff00", linestyle="--", linewidth=1.5, alpha=0.8)
    axes_qc[0, 0].set_title(f"A. 3D DAPI Population Template (Y_L={ap_or_yl}, Y_R={yr}, Bregma ~ {bregma_str})",
                           color="white", fontsize=13, fontweight="bold")
    axes_qc[0, 0].axis("off")

    # --- Panel B: Dual Anatomical Overlay ---
    rgb_ov = np.zeros((hc, wc, 3), dtype=float)
    if cached_w_tmpl is not None and cached_w_tmpl.shape == fused_crop.shape:
        rgb_ov[:, :, 0] = cached_w_tmpl * 0.9
        rgb_ov[:, :, 1] = fused_crop * 0.9
        rgb_ov[:, :, 2] = np.maximum(cached_w_tmpl, fused_crop) * 0.9
    else:
        rgb_ov[:, :, 1] = fused_crop

    axes_qc[0, 1].imshow(np.clip(rgb_ov, 0, 1))
    axes_qc[0, 1].axvline(mid_x, color="#ffff00", linestyle="--", linewidth=1.5)
    axes_qc[0, 1].set_title("B. Dual Anatomical Overlay (Cyan: Experimental, Magenta: Atlas)",
                           color="white", fontsize=13, fontweight="bold")
    axes_qc[0, 1].axis("off")

    # --- Panel C: 3D Spatial Warped Model Baseline ---
    if cached_w_tmpl is not None:
        axes_qc[1, 0].imshow(cached_w_tmpl, cmap="magma", vmin=0, vmax=0.9)
    else:
        axes_qc[1, 0].imshow(fused_crop, cmap="magma", vmin=0, vmax=0.9)

    for rkey in active_roi_keys:
        if rkey in raw_roi_polys:
            rdata = raw_roi_polys[rkey]
            for c in rdata.get('contours', []):
                axes_qc[1, 0].plot(c[:, 1], c[:, 0], color="#ffffff", linewidth=1.1, alpha=0.7)

    axes_qc[1, 0].axvline(mid_x, color="#ffff00", linestyle="--", linewidth=1.5)
    axes_qc[1, 0].set_title("C. Warped DAPI Template Baseline & Initial Boundaries",
                           color="white", fontsize=13, fontweight="bold")
    axes_qc[1, 0].axis("off")

    # --- Panel D: Final Transformed Target Subregions ---
    axes_qc[1, 1].imshow(fused_crop, cmap="gray", vmin=0, vmax=0.70)

    # Optional DL overlay in dashed lines
    if dl_contours_dict:
        for rname, cnts in dl_contours_dict.items():
            hex_c = BASE_REGION_COLORS.get(rname, "#39ff14")
            for c in cnts:
                axes_qc[1, 1].plot(c[:, 1], c[:, 0], color=hex_c, linestyle=":", linewidth=1.2, alpha=0.6)

    for rkey in active_roi_keys:
        contours = transformed_contours_dict.get(rkey, [])
        if not contours:
            continue
        rdata = raw_roi_polys.get(rkey, {})
        hex_c = rdata.get('color', BASE_REGION_COLORS.get(rkey, "#ffffff"))
        lw = 2.2 if 'Cortex' in rkey else 1.6
        for c in contours:
            axes_qc[1, 1].plot(c[:, 1], c[:, 0], color=hex_c, linewidth=lw)

        all_pts = np.vstack(contours)
        cy, cx = np.mean(all_pts, axis=0)
        tag_name = rdata.get('base_name', rkey)
        if 'Left' in rkey or '_L' in rkey or '左' in rkey:
            tag_name += "_L"
        if 'Right' in rkey or '_R' in rkey or '右' in rkey:
            tag_name += "_R"
        if 'Cortex' not in rkey:
            axes_qc[1, 1].text(cx, cy, tag_name, color="white", fontsize=9, fontweight="bold",
                               ha="center", va="center",
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=hex_c, alpha=0.85, edgecolor="none"))

    axes_qc[1, 1].axvline(mid_x, color="#ffff00", linestyle="--", linewidth=1.6, alpha=0.9)
    axes_qc[1, 1].set_title("D. Final Aligned Nuclei & Outline Annotations",
                           color="#ffff00", fontsize=13, fontweight="bold")
    axes_qc[1, 1].axis("off")

    fig_qc.suptitle(f"{slide_name} - Master 4-Panel Registration QC (Y_L={ap_or_yl}, Y_R={yr}, Bregma {bregma_str})",
                    color="white", fontsize=15, fontweight="bold")
    fig_qc.savefig(qc_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig_qc)
