"""
ccfv3_aligner.core.landmark
Anatomical Landmark Estimator: Midline, Dorsal Apex, and Split-Hemisphere Scales.
"""

import numpy as np

def estimate_landmarks(fused_crop, tmpl_raw, mid_ccf=228.0):
    """
    Estimates initial slice midline and left/right independent baseline scales.
    """
    hc, wc = fused_crop.shape

    # 1. Midline estimation via vertical profile minimum near center
    mid_init = wc // 2
    prof = np.sum(fused_crop[int(hc * 0.2):int(hc * 0.7), :], axis=0)
    search_r = min(40, mid_init // 2)
    mid_x_auto = int(np.argmin(prof[mid_init - search_r:mid_init + search_r])) + (mid_init - search_r)

    # 2. Dorsal apex landmarks for left & right
    y_l, x_l = np.where(fused_crop[:, :mid_x_auto] > 0.06)
    y_top_l = np.min(y_l) if len(y_l) > 0 else 10
    h_l = (np.max(y_l) - y_top_l) if len(y_l) > 0 else hc
    w_l = (mid_x_auto - np.min(x_l)) if len(x_l) > 0 else (wc // 2)

    y_r, x_r = np.where(fused_crop[:, mid_x_auto:] > 0.06)
    y_top_r = np.min(y_r) if len(y_r) > 0 else 10
    h_r = (np.max(y_r) - y_top_r) if len(y_r) > 0 else hc
    w_r = (np.max(x_r)) if len(x_r) > 0 else (wc // 2)

    # 3. Reference CCF dimensions
    y_tl, x_tl = np.where(tmpl_raw[:, :int(mid_ccf)] > 10)
    h_ccf_l = (np.max(y_tl) - np.min(y_tl)) if len(y_tl) > 0 else 200
    w_ccf_l = (mid_ccf - np.min(x_tl)) if len(x_tl) > 0 else 200

    y_tr, x_tr = np.where(tmpl_raw[:, int(mid_ccf):] > 10)
    h_ccf_r = (np.max(y_tr) - np.min(y_tr)) if len(y_tr) > 0 else 200
    w_ccf_r = (np.max(x_tr)) if len(x_tr) > 0 else 200

    base_sx_l = float(w_l) / float(w_ccf_l)
    base_sy_l = float(h_l) / float(h_ccf_l)
    base_sx_r = float(w_r) / float(w_ccf_r)
    base_sy_r = float(h_r) / float(h_ccf_r)

    return {
        'mid_x': mid_x_auto,
        'y_top_l': y_top_l,
        'y_top_r': y_top_r,
        'base_sx_l': base_sx_l,
        'base_sy_l': base_sy_l,
        'base_sx_r': base_sx_r,
        'base_sy_r': base_sy_r
    }
