# -*- coding: utf-8 -*-
"""
ccfv3_aligner.core.atlas
3D DAPI Population Atlas and Allen CCFv3 Multi-Modal Atlas Manager.
Supports both the high-resolution 3D DAPI Population Template and BrainGlobe CCFv3.
"""

import os
import sys
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
except ImportError:
    try:
        from bg_atlasapi import BrainGlobeAtlas
    except ImportError:
        BrainGlobeAtlas = None

from ..utils.config import BASE_REGION_COLORS, TARGET_ACRONYMS


def find_default_dapi_atlas_paths():
    """Locate 3D DAPI template and segmentation volumes."""
    # 1. Environment variable override
    env_dir = os.environ.get("CCFV3_DAPI_ATLAS_DIR")
    if env_dir and os.path.exists(env_dir):
        nii = os.path.join(env_dir, "dapi_template.nii.gz")
        seg = os.path.join(env_dir, "dapi_template_segmentation_full.nii.gz")
        if os.path.exists(nii) and os.path.exists(seg):
            return nii, seg

    # 2. Known project paths
    candidate_dirs = [
        r"D:\2024_DBSeq\results\260814_Rsrc1-Sf3b1 IF\dapi_atlas_paper",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dapi_atlas_paper"),
    ]
    for c_dir in candidate_dirs:
        nii = os.path.join(c_dir, "dapi_template.nii.gz")
        seg = os.path.join(c_dir, "dapi_template_segmentation_full.nii.gz")
        if os.path.exists(nii) and os.path.exists(seg):
            return nii, seg

    return None, None


class AtlasManager:
    """
    Unified Atlas Manager providing access to:
    1. 3D DAPI Population Template & Multi-Nuclei Annotation
    2. Standard Allen CCFv3 (via BrainGlobe API fallback)
    """
    def __init__(self, dapi_nii_path=None, dapi_seg_path=None, use_dapi=True):
        self.use_dapi = use_dapi
        self.dapi_vol_3d = None
        self.dapi_seg_3d = None
        self.bg_atlas = None
        self.bg_reference = None
        self.bg_annotation = None
        self.region_groups = {}
        self.region_colors = BASE_REGION_COLORS

        # Try loading 3D DAPI template
        if use_dapi:
            if not dapi_nii_path or not dapi_seg_path:
                auto_nii, auto_seg = find_default_dapi_atlas_paths()
                dapi_nii_path = dapi_nii_path or auto_nii
                dapi_seg_path = dapi_seg_path or auto_seg

            if dapi_nii_path and dapi_seg_path and os.path.exists(dapi_nii_path) and os.path.exists(dapi_seg_path) and sitk is not None:
                try:
                    print(f"[AtlasManager] Loading 3D DAPI Template: {dapi_nii_path}")
                    img_obj = sitk.ReadImage(dapi_nii_path)
                    self.dapi_vol_3d = sitk.GetArrayFromImage(img_obj).astype(np.float32)

                    seg_obj = sitk.ReadImage(dapi_seg_path)
                    self.dapi_seg_3d = sitk.GetArrayFromImage(seg_obj).astype(np.uint32)
                    self.y_min_limit = 0
                    self.y_max_limit = self.dapi_vol_3d.shape[0] - 1
                    self.bregma_y_idx = min(120, self.y_max_limit)
                    print(f"[AtlasManager] 3D DAPI Loaded. Shape: {self.dapi_vol_3d.shape}, AP Slices: {self.y_max_limit+1}")
                except Exception as e:
                    print(f"[AtlasManager] Failed to read DAPI volumes: {e}")

        # Fallback / Secondary: BrainGlobe Atlas
        if BrainGlobeAtlas is not None:
            try:
                self.bg_atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
                self.bg_reference = self.bg_atlas.template
                self.bg_annotation = self.bg_atlas.annotation
                self.structures = self.bg_atlas.structures
                self.region_groups = self._build_region_groups()
            except Exception as e:
                print(f"[AtlasManager] BrainGlobe CCFv3 not initialized ({e}). Using basic ontology.")

        if not self.region_groups:
            self.region_groups = {k: [i+1] for i, k in enumerate(TARGET_ACRONYMS)}

    def _get_all_descendants(self, acronym: str) -> list:
        if not self.bg_atlas or not hasattr(self, "structures"):
            return []
        try:
            struct_info = self.structures[acronym]
            struct_id = struct_info["id"]
            descendants = [struct_id]

            def recurse_tree(sid):
                for child in self.bg_atlas.hierarchy.children(sid):
                    cid = child.identifier
                    descendants.append(cid)
                    recurse_tree(cid)

            recurse_tree(struct_id)
            return list(set(descendants))
        except Exception:
            return []

    def _build_region_groups(self) -> dict:
        groups = {}
        for acronym in ["RSPd", "CA3", "CA1", "DG", "GPe", "CP", "LSc", "RT", "TRS", "MH", "LH", "RE", "PF", "CM"]:
            groups[acronym] = self._get_all_descendants(acronym)

        atn_ids = []
        for a in ["AD", "AV", "AM"]:
            atn_ids.extend(self._get_all_descendants(a))
        groups["ATN"] = list(set(atn_ids))
        groups["Cortex"] = self._get_all_descendants("Isocortex")
        return groups

    def get_dapi_slice(self, y_idx: int):
        """Extract a single coronal slice from the 3D DAPI Population Template."""
        if self.dapi_vol_3d is None or self.dapi_seg_3d is None:
            # Generate placeholder if volume not available
            h, w = 384, 512
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.uint32)

        y = int(np.clip(y_idx, 0, self.dapi_vol_3d.shape[0] - 1))
        tmpl = self.dapi_vol_3d[y, :, :].copy()
        seg = self.dapi_seg_3d[y, :, :].copy()
        return tmpl, seg

    def get_dapi_hemisphere_slice(self, y_idx: int, is_left: bool = True):
        """Extract single hemisphere slice from 3D DAPI volume."""
        tmpl_full, seg_full = self.get_dapi_slice(y_idx)
        mid_x = tmpl_full.shape[1] // 2
        if is_left:
            return tmpl_full[:, :mid_x], seg_full[:, :mid_x]
        else:
            return tmpl_full[:, mid_x:], seg_full[:, mid_x:]


# Alias for backwards compatibility
AllenCCFAtlas = AtlasManager
