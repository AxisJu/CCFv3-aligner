# -*- coding: utf-8 -*-
"""
ccfv3_aligner.core.slide_io
Robust Multi-Format Whole Slide Image (WSI) and Multichannel Slide Reader.
Supports Olympus APX100 VSI (via slideio or ETS fallback), TIFF, OME-TIFF, and Standard Images.
"""

import os
import glob
import numpy as np
from skimage.transform import resize

try:
    import slideio
except ImportError:
    slideio = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        iio = None


class SlideLoader:
    """
    Universal Slide Loader supporting:
    - Olympus VSI (via slideio or pyramid ETS tifffile)
    - Multichannel TIFF / OME-TIFF
    - PNG / JPEG standard images
    """
    def __init__(self, slide_path: str):
        self.slide_path = slide_path
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide not found: {slide_path}")

        self.ext = os.path.splitext(slide_path)[1].lower()
        self.slideio_slide = None
        self.scene = None
        self.width = 1000
        self.height = 1000
        self.num_channels = 1
        self._ets_files = []

        if self.ext == ".vsi" and slideio is not None:
            try:
                self.slideio_slide = slideio.open_slide(slide_path, "VSI")
                if self.slideio_slide.num_scenes > 0:
                    self.scene = self.slideio_slide.get_scene(0)
                    rect = self.scene.rect
                    self.width = rect[2]
                    self.height = rect[3]
                    self.num_channels = self.scene.num_channels
            except Exception as e:
                # Fallback to ETS discovery
                self._discover_vsi_ets()
        elif self.ext == ".vsi":
            self._discover_vsi_ets()

    def _discover_vsi_ets(self):
        base_dir = os.path.dirname(self.slide_path)
        base_name = os.path.splitext(os.path.basename(self.slide_path))[0]
        data_dir = os.path.join(base_dir, f"_{base_name}_")
        if os.path.exists(data_dir):
            self._ets_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.ets"), recursive=True))

    def get_thumbnail(self, target_width: int = 1200) -> np.ndarray:
        """
        Extract downsampled whole slide image (H, W, C) or (H, W).
        """
        if self.scene is not None:
            scale = max(1, self.width // target_width)
            target_h = max(1, self.height // scale)
            target_w = max(1, self.width // scale)
            return self.scene.read_block((0, 0, self.width, self.height), (target_w, target_h))

        if self.ext == ".vsi" and self._ets_files:
            return self._read_vsi_ets_all(target_width)

        if self.ext in [".tif", ".tiff"] and tifffile is not None:
            img = tifffile.imread(self.slide_path)
            if img.ndim == 3 and img.shape[0] < 10 and img.shape[0] < img.shape[1]:
                # Channel-first format (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            h, w = img.shape[:2]
            scale = target_width / w
            target_h = int(h * scale)
            return resize(img, (target_h, target_width), preserve_range=True, order=1).astype(np.float32)

        if iio is not None:
            img = iio.imread(self.slide_path)
            h, w = img.shape[:2]
            scale = target_width / w
            target_h = int(h * scale)
            return resize(img, (target_h, target_width), preserve_range=True, order=1).astype(np.float32)

        return np.zeros((800, target_width, 3), dtype=np.float32)

    def _read_vsi_ets_all(self, target_width: int):
        channels = []
        for ch_idx in range(min(4, len(self._ets_files))):
            ch_arr = self._read_vsi_ets_channel(ch_idx, target_width)
            channels.append(ch_arr)
        if len(channels) == 1:
            return channels[0]
        return np.stack(channels, axis=-1)

    def _read_vsi_ets_channel(self, channel_idx: int, target_width: int):
        ch_files = [f for f in self._ets_files if f"C{channel_idx:02d}" in f or f"c{channel_idx}" in f]
        target_file = ch_files[0] if ch_files else (self._ets_files[channel_idx] if channel_idx < len(self._ets_files) else self._ets_files[0])
        try:
            with tifffile.TiffFile(target_file) as tif:
                series = tif.series[0]
                levels = series.levels
                best_level = levels[-1]
                for lvl in reversed(levels):
                    if lvl.shape[-1] >= target_width // 2:
                        best_level = lvl
                        break
                img = best_level.asarray()
                if img.ndim == 3:
                    img = img[0]
                h, w = img.shape
                if w != target_width:
                    th = int(h * (target_width / w))
                    img = resize(img, (th, target_width), preserve_range=True, order=1).astype(np.float32)
                return img.astype(np.float32)
        except Exception:
            return np.zeros((800, target_width), dtype=np.float32)

    def get_channel_thumbnail(self, channel_idx: int = 0, target_width: int = 1200) -> np.ndarray:
        thumb = self.get_thumbnail(target_width)
        if thumb.ndim == 3 and thumb.shape[2] > channel_idx:
            return thumb[:, :, channel_idx].astype(np.float32)
        elif thumb.ndim == 2:
            return thumb.astype(np.float32)
        return thumb[:, :, 0].astype(np.float32)


# Backwards compatibility alias
APX100Slide = SlideLoader
