#!/usr/bin/env bash
# APX100 3D DAPI Alignment Studio Bash Launcher

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "==============================================================================="
echo "  APX100 Mouse Brain Slice Interactive Alignment Studio"
echo "  3D DAPI Population Template & Deep Learning Nuclei Segmentation"
echo "==============================================================================="

python3 scripts/run_studio.py "$@"
