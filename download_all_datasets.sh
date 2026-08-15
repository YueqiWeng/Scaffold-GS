#!/usr/bin/env bash
# Download all Scaffold-GS benchmark datasets into data/
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p data downloads

echo "=== [1/4] Tanks&Temples + Deep Blending (~650 MB) ==="
if [[ -d data/tandt/train/sparse && -d data/db/drjohnson/sparse ]]; then
    echo "Already present, skipping."
else
    wget -c -O downloads/tandt_db.zip \
        https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
    unzip -qo downloads/tandt_db.zip -d data
    rm -f downloads/tandt_db.zip
fi
# Deep Blending zip may extract as db/ (used as data/db/ in sweeps)
ln -sfn db data/blending

echo "=== [2/4] MipNeRF360 (~12 GB) ==="
if [[ -d data/mipnerf360/bicycle/sparse || -d data/mipnerf360/bicycle/images ]]; then
    echo "Already present, skipping."
else
    mkdir -p data/mipnerf360
    wget -c -O downloads/360_v2.zip \
        http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
    unzip -qo downloads/360_v2.zip -d data/mipnerf360
    rm -f downloads/360_v2.zip
fi

echo "=== [3/4] BungeeNeRF (~8 GB, Google Drive — file is tar.gz, not zip) ==="
if [[ -d data/bungeenerf/amsterdam/sparse ]]; then
    echo "Already present, skipping."
elif [[ -f downloads/bungeenerf.zip || -f downloads/bungeenerf.tar.gz ]]; then
    ARCHIVE="downloads/bungeenerf.tar.gz"
    [[ -f downloads/bungeenerf.zip ]] && ARCHIVE="downloads/bungeenerf.zip"
    tar -xzf "$ARCHIVE" -C data
    rm -f downloads/bungeenerf.zip downloads/bungeenerf.tar.gz
else
    pip install -q gdown
    gdown 1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww -O downloads/bungeenerf.tar.gz
    tar -xzf downloads/bungeenerf.tar.gz -C data
    rm -f downloads/bungeenerf.tar.gz
fi

echo "=== [4/4] NeRF Synthetic (~600 MB) ==="
if [[ -f data/nerfsynthetic/chair/transforms_train.json ]]; then
    echo "Already present, skipping."
else
    mkdir -p data/nerfsynthetic
    pip install -q huggingface_hub
    echo "Downloading from HuggingFace mirror (Berkeley link is dead)..."
    for s in chair drums ficus hotdog lego materials mic ship; do
        echo "  -> $s"
        hf download "pablovela5620/nerf-synthetic-mirror" \
            --repo-type dataset \
            --local-dir "data/nerfsynthetic" \
            --include "${s}/*"
    done
fi

echo ""
echo "=== Dataset summary ==="
find data -maxdepth 3 -type d \( -name sparse -o -name images \) | sed 's|/sparse||;s|/images||' | sort -u
