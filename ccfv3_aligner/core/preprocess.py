"""
ccfv3_aligner.core.preprocess
Tissue parenchyma extraction, multi-channel outlier dust filtering & normalization.
"""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening, disk, remove_small_holes

def preprocess_multichannel_slice(dapi, neun, gfp=None, min_tissue_thresh=0.07):
    """
    Cleans background artifacts, filters high-saturation multi-channel dust,
    and isolates the main brain slice parenchyma.
    """
    h, w = dapi.shape
    
    # 1. Multi-channel outlier dust filter
    p999_d = np.percentile(dapi, 99.9)
    p999_n = np.percentile(neun, 99.9)
    if gfp is not None:
        p999_g = np.percentile(gfp, 99.9)
        dust_mask = (dapi > p999_d * 1.5) & (neun > p999_n * 1.5) & (gfp > p999_g * 1.5)
    else:
        dust_mask = (dapi > p999_d * 1.5) & (neun > p999_n * 1.5)

    # 2. Robust quantile normalization
    p1_d, p99_d = np.percentile(dapi, (1, 99.2))
    d_norm = np.clip((dapi - p1_d) / (p99_d - p1_d + 1e-5), 0, 1)

    p1_n, p99_n = np.percentile(neun, (1, 99.2))
    n_norm = np.clip((neun - p1_n) / (p99_n - p1_n + 1e-5), 0, 1)

    d_norm[dust_mask] = 0
    n_norm[dust_mask] = 0

    # 3. Dual morphological opening/closing
    fused_raw = 0.35 * d_norm + 0.65 * n_norm
    tissue_closed = closing(fused_raw > min_tissue_thresh, disk(8))
    tissue_opened = opening(tissue_closed, disk(4))
    lbl = label(tissue_opened)
    props = regionprops(lbl)
    if not props:
        return d_norm, n_norm, fused_raw, (0, 0, h, w)

    main_p = max(props, key=lambda x: x.area)
    strict_main_mask = (lbl == main_p.label)
    strict_main_mask = remove_small_holes(strict_main_mask, max_size=5000)

    d_pure = d_norm * strict_main_mask
    n_pure = n_norm * strict_main_mask
    fused_pure = 0.35 * d_pure + 0.65 * n_pure

    minr, minc, maxr, maxc = main_p.bbox
    pad = 12
    y0, y1 = max(0, minr - pad), min(h, maxr + pad)
    x0, x1 = max(0, minc - pad), min(w, maxc + pad)

    d_crop = d_pure[y0:y1, x0:x1]
    n_crop = n_pure[y0:y1, x0:x1]
    fused_crop = fused_pure[y0:y1, x0:x1]

    return d_crop, n_crop, fused_crop, (x0, y0, x1, y1)
