# -*- coding: utf-8 -*-
"""
scripts/predict_slice.py
Command-Line Interface for Deep Learning Multi-Nuclei Brain Slice Segmentation.
Uses AttentionResUNet with Canonical Hemisphere Mirroring & Assembly.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure package root is in python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ccfv3_aligner.core.slide_io import SlideLoader
from ccfv3_aligner.models.predictor import BrainRegionPredictor, find_default_model_path
from ccfv3_aligner.models.config import CLASSES, NUM_CLASSES


def run_prediction_on_file(input_path, output_dir, predictor):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"\n[Predict] Processing: {input_path}")

    # Load slide
    loader = SlideLoader(input_path)
    dapi = loader.get_channel_thumbnail(0, target_width=1200)
    neun = loader.get_channel_thumbnail(1, target_width=1200)

    # Denoising & normalization
    p1_d, p99_d = np.percentile(dapi, (1, 99.2))
    d_norm = np.clip((dapi - p1_d) / (p99_d - p1_d + 1e-5), 0, 1).astype(np.float32)

    p1_n, p99_n = np.percentile(neun, (1, 99.2))
    n_norm = np.clip((neun - p1_n) / (p99_n - p1_n + 1e-5), 0, 1).astype(np.float32)

    # Predict
    res = predictor.predict_full_slice_arrays(d_norm, n_norm)
    mask = res["pred_class_mask"]
    rgb = res["pred_rgb"]

    # 1. Save Segmentation Mask (PNG & NPY)
    mask_png_path = os.path.join(output_dir, f"{base_name}_pred_mask.png")
    mask_npy_path = os.path.join(output_dir, f"{base_name}_pred_mask.npy")
    Image.fromarray(mask).save(mask_png_path)
    np.save(mask_npy_path, mask)

    # 2. Save Overlay Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
    plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.90, bottom=0.05)

    fused = 0.5 * d_norm + 0.5 * n_norm
    axes[0].imshow(fused, cmap="gray", vmin=0, vmax=0.75)
    axes[0].set_title("Experimental Fluorescence (DAPI+NeuN)", color="white", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].set_title("AI Multi-Nuclei Segmentation", color="#39ff14", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Blended
    blended = np.clip(np.stack([fused, fused, fused], axis=-1) * 0.5 + rgb * 0.5, 0, 1)
    axes[2].imshow(blended)
    axes[2].set_title("Blended Overlay", color="#89b4fa", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"{base_name} - Deep Learning 15 Core Nuclei Segmentation", color="white", fontsize=14, fontweight="bold")
    fig_path = os.path.join(output_dir, f"{base_name}_pred_overlay.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # 3. Export Region Pixel Statistics
    stats = []
    for c_info in CLASSES:
        idx = c_info["idx"]
        if idx == 0: continue
        px_count = int(np.sum(mask == idx))
        stats.append({
            "region_idx": idx,
            "acronym": c_info["acronym"],
            "name": c_info["name"],
            "allen_id": c_info["allen_id"],
            "pixel_count": px_count
        })
    df_stats = pd.DataFrame(stats)
    csv_path = os.path.join(output_dir, f"{base_name}_region_stats.csv")
    df_stats.to_csv(csv_path, index=False)

    print(f"[Predict] Saved results -> {output_dir}")
    return df_stats


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Nuclei Segmentation for Brain Slices")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image/slide (.vsi, .tif, .png) or folder")
    parser.add_argument("--output", "-o", type=str, default="results/dl_predictions", help="Directory to save output masks & QC")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to best_model.pth checkpoint")
    args = parser.parse_args()

    predictor = BrainRegionPredictor(checkpoint_path=args.checkpoint)

    if os.path.isfile(args.input):
        run_prediction_on_file(args.input, args.output, predictor)
    elif os.path.isdir(args.input):
        valid_exts = [".vsi", ".tif", ".tiff", ".png", ".jpg"]
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in valid_exts]
        print(f"[Predict] Found {len(files)} files in directory: {args.input}")
        for f in files:
            try:
                run_prediction_on_file(f, args.output, predictor)
            except Exception as e:
                print(f"[Predict] Error processing {f}: {e}")


if __name__ == "__main__":
    main()
