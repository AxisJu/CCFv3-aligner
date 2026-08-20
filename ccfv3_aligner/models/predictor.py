# -*- coding: utf-8 -*-
"""
ccfv3_aligner.models.predictor
Deep Learning Inference Engine for 15-Nuclei Segmentation with Canonical Hemisphere Mirroring.
"""

import os
import sys
import numpy as np
import torch
from skimage.transform import resize
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, opening, disk, remove_small_holes

from .config import (
    CLASSES, NUM_CLASSES, ALLEN_TO_IDX, IDX_TO_ALLEN,
    IDX_TO_ACRONYM, IDX_TO_COLOR, DEVICE, IN_CHANNELS,
    TARGET_SIZE, DEFAULT_MODEL_FILENAME
)
from .network import AttentionResUNet
from .denoise import denoise_fluorescence_channel


def find_default_model_path():
    """Locate the best available deep learning model checkpoint."""
    # 1. Environment variable override
    env_path = os.environ.get("CCFV3_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Local repo models directory
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_models_dir = os.path.join(os.path.dirname(pkg_dir), "models")
    candidate_1 = os.path.join(repo_models_dir, DEFAULT_MODEL_FILENAME)
    if os.path.exists(candidate_1):
        return candidate_1

    # 3. Known project directory
    candidate_2 = r"D:\2024_DBSeq\results\260814_Rsrc1-Sf3b1 IF\260818_deeplearning\models\best_model.pth"
    if os.path.exists(candidate_2):
        return candidate_2

    return None


class BrainRegionPredictor:
    """
    Inference Engine using trained Dual-Channel Attention ResUNet.
    Employs Canonical Hemisphere Mirroring:
    - Splits full brain slice into Left & Right halves.
    - Left half is mirrored (fliplr) to canonical right orientation for prediction.
    - Left prediction is mirrored back and stitched with Right prediction into full slice.
    """
    def __init__(self, checkpoint_path=None, device=DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path or find_default_model_path()
        self.model = AttentionResUNet(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES
        ).to(self.device)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                val_dice = ckpt.get("best_val_dice", ckpt.get("latest_val_dice", None))
                dice_info = f" (Val Dice: {val_dice:.4f})" if val_dice is not None else ""
                print(f"[DeepLearning] Loaded model from: {self.checkpoint_path}{dice_info}")
            except Exception as e:
                print(f"[DeepLearning] Error loading checkpoint {self.checkpoint_path}: {e}")
        else:
            print(f"[DeepLearning] Warning: Checkpoint not found. Model running with uninitialized weights.")

        self.model.eval()

    def predict_half_array(self, dapi_half, neun_half=None):
        """
        Predicts a single hemisphere array (H, W) in canonical right orientation.

        Parameters
        ----------
        dapi_half : np.ndarray
            DAPI channel normalized to [0, 1].
        neun_half : np.ndarray, optional
            NeuN channel normalized to [0, 1]. If None, dapi_half is duplicated.

        Returns
        -------
        pred_class_half : np.ndarray (H, W) uint8 in [0..15]
        probs_orig : np.ndarray (16, H, W) float32
        """
        if neun_half is None:
            neun_half = dapi_half

        orig_h, orig_w = dapi_half.shape
        d_res = resize(dapi_half, TARGET_SIZE, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        n_res = resize(neun_half, TARGET_SIZE, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

        inp_arr = np.stack([d_res, n_res], axis=0)  # (2, 512, 384)
        inp_tensor = torch.from_numpy(inp_arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(inp_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (16, 512, 384)

        probs_orig = np.zeros((NUM_CLASSES, orig_h, orig_w), dtype=np.float32)
        for c in range(NUM_CLASSES):
            probs_orig[c] = resize(probs[c], (orig_h, orig_w), order=1, preserve_range=True, anti_aliasing=True)

        pred_class_half = np.argmax(probs_orig, axis=0).astype(np.uint8)
        return pred_class_half, probs_orig

    def predict_full_slice_arrays(self, dapi_crop, neun_crop=None, mid_x=None):
        """
        Predicts a full 2-channel brain slice via Hemisphere Splitting & Mirror Assembly.

        Parameters
        ----------
        dapi_crop : np.ndarray
            DAPI slice array (H, W) in [0, 1].
        neun_crop : np.ndarray, optional
            NeuN slice array (H, W) in [0, 1].
        mid_x : int, optional
            Midline coordinate. If None, mid_x = W // 2.

        Returns
        -------
        dict containing:
            - pred_class_mask: np.ndarray (H, W) uint8 in [0..15]
            - pred_allen_mask: np.ndarray (H, W) uint32 with Allen CCF IDs
            - pred_rgb: np.ndarray (H, W, 3) float32 RGB overlay
            - mid_x: int
            - contours_by_region: dict of region acronym -> list of (N, 2) [y, x] contours
        """
        if neun_crop is None:
            neun_crop = dapi_crop

        orig_h, orig_w = dapi_crop.shape
        if mid_x is None or mid_x <= 0 or mid_x >= orig_w:
            mid_x = orig_w // 2

        # Left hemisphere: mirrored horizontally to canonical right orientation
        d_left_mirrored = np.ascontiguousarray(np.fliplr(dapi_crop[:, :mid_x]))
        n_left_mirrored = np.ascontiguousarray(np.fliplr(neun_crop[:, :mid_x]))

        # Right hemisphere: canonical right orientation
        d_right = np.ascontiguousarray(dapi_crop[:, mid_x:])
        n_right = np.ascontiguousarray(neun_crop[:, mid_x:])

        pred_l_mirr, _ = self.predict_half_array(d_left_mirrored, n_left_mirrored)
        pred_r, _ = self.predict_half_array(d_right, n_right)

        # Mirror left prediction back
        pred_l = np.ascontiguousarray(np.fliplr(pred_l_mirr))

        # Re-assemble full mask
        pred_class_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
        pred_class_full[:, :mid_x] = pred_l
        pred_class_full[:, mid_x:] = pred_r

        # Map to Allen CCF IDs
        pred_allen_full = np.zeros((orig_h, orig_w), dtype=np.uint32)
        for c_idx, allen_id in IDX_TO_ALLEN.items():
            pred_allen_full[pred_class_full == c_idx] = allen_id

        # Render RGB overlay
        pred_rgb_full = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        contours_by_region = {}

        for c_info in CLASSES:
            idx = c_info["idx"]
            if idx == 0:
                continue
            hex_c = c_info["color"].lstrip("#")
            rgb_c = tuple(int(hex_c[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            c_mask = (pred_class_full == idx)
            pred_rgb_full[c_mask] = rgb_c

            # Extract contours
            if np.any(c_mask):
                cnts = find_contours(c_mask.astype(float), 0.5)
                contours_by_region[c_info["acronym"]] = cnts

        return {
            "pred_class_mask": pred_class_full,
            "pred_allen_mask": pred_allen_full,
            "pred_rgb": pred_rgb_full,
            "mid_x": mid_x,
            "contours_by_region": contours_by_region
        }
