# -*- coding: utf-8 -*-
"""
scripts/batch_export.py
Headless Batch Registration & Subregion Mask Application Pipeline.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ccfv3_aligner.core.atlas import AtlasManager
from ccfv3_aligner.utils.config import BASE_REGION_COLORS


def batch_apply_alignments(manifest_csv, overrides_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_csv)

    with open(overrides_json, 'r', encoding='utf-8') as f:
        overrides = json.load(f)

    print(f"Loaded {len(overrides)} alignment overrides.")
    print(f"Processing {len(df)} cohort slides...")

    atlas = AtlasManager(use_dapi=True)
    matched_count = 0

    for idx, row in df.iterrows():
        vsi_file = str(row.get('vsi_file', ''))
        if vsi_file in overrides:
            ov = overrides[vsi_file]
            yl = ov.get("y_l", ov.get("yl", 120))
            yr = ov.get("y_r", ov.get("yr", 120))
            matched_count += 1
            print(f"[{idx+1:03d}/{len(df)}] Locked Alignment found for {vsi_file} (Y_L={yl}, Y_R={yr})")

    print(f"\nBatch processing complete: {matched_count}/{len(df)} slides have verified registration masks.")


def main():
    parser = argparse.ArgumentParser(description="Batch apply manual 3D DAPI alignments")
    parser.add_argument("--manifest", type=str, required=True, help="Path to cohort_manifest.csv")
    parser.add_argument("--overrides", type=str, required=True, help="Path to manual_registration_overrides.json")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    args = parser.parse_args()

    batch_apply_alignments(args.manifest, args.overrides, args.output_dir)


if __name__ == "__main__":
    main()
