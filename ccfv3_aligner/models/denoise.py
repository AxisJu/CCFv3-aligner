# -*- coding: utf-8 -*-
"""
ccfv3_aligner.models.denoise
Fluorescence Channel Denoising, Background Estimation, and Normalization.
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def denoise_fluorescence_channel(img_norm):
    """
    Robust Denoising for Fluorescence Channels (DAPI / NeuN):
    1. Hot-pixel & speckle noise suppression using 3x3 median filter.
    2. Edge-preserving Gaussian smoothing (sigma=0.6).
    3. Low-frequency background estimation & soft suppression.
    4. Contrast stretching & robust percentile normalization to [0, 1].

    Parameters
    ----------
    img_norm : np.ndarray
        Input channel image (2D float32 or uint16/uint8 normalized to [0, 1]).

    Returns
    -------
    np.ndarray
        Cleaned, background-suppressed, normalized float32 array in [0, 1].
    """
    img_float = img_norm.astype(np.float32)
    med = median_filter(img_float, size=3)
    smoothed = gaussian_filter(med, sigma=0.6)
    bg_lowpass = gaussian_filter(smoothed, sigma=25.0)
    bg_subtracted = np.maximum(0.0, smoothed - 0.40 * bg_lowpass)
    p_low, p_high = np.percentile(bg_subtracted, (0.5, 99.5))
    if p_high - p_low > 1e-6:
        denoised = np.clip((bg_subtracted - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        denoised = bg_subtracted
    return denoised.astype(np.float32)
