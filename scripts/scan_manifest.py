"""
scripts/scan_manifest.py
Recursively discovers and indexes VSI/TIFF slide files into a structured cohort manifest.
"""

import os
import argparse
import pandas as pd

def scan_cohort(data_dir, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    slides = []
    
    print(f"Scanning directory: {data_dir} ...")
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if (f.lower().endswith('.vsi') or f.lower().endswith('.tif') or f.lower().endswith('.tiff')) and not f.startswith('.'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, data_dir)
                parts = rel_path.split(os.sep)

                target = 'Unknown'
                group = 'Unknown'
                mouse_id = 'Unknown'

                for p in parts:
                    if 'rsrc1' in p.lower(): target = 'RSRC1'
                    elif 'sf3b1' in p.lower(): target = 'SF3B1'
                    if 'saline' in p.lower(): group = 'Saline'
                    elif 'ka' in p.lower(): group = 'KA'
                    if '_m' in p.lower() or 'mouse' in p.lower():
                        mouse_id = p

                slides.append({
                    'vsi_file': f,
                    'vsi_path': full_path,
                    'rel_path': rel_path,
                    'target': target,
                    'group': group,
                    'mouse_id': mouse_id,
                    'folder': parts[0] if len(parts) > 1 else ''
                })

    df = pd.DataFrame(slides)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Successfully indexed {len(df)} slides to {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Scan and generate slice cohort manifest")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder containing raw VSI/TIFF slides")
    parser.add_argument("--output-csv", type=str, required=True, help="Target path for cohort_manifest.csv")
    args = parser.parse_args()

    scan_cohort(args.data_dir, args.output_csv)

if __name__ == "__main__":
    main()
