#!/bin/bash
# Vast.ai setup script for SSDLite MobileNetV2 training
#
# Usage:
#   1. Rent a GPU instance on vast.ai with:
#      - Docker image: pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
#      - Disk space: 50GB+ (or use persistent /workspace storage)
#   2. SSH into the instance
#   3. Run: bash setup_vastai.sh
#
# COCO dataset is stored in /workspace/coco so it persists across instances.
# If /workspace is not available (no persistent storage), falls back to local data/COCO.

set -e

# Branch carrying the active training recipe + eval-harness fixes.
# NOTE: the repo's default branch (master) does NOT have these — always deploy this branch.
BRANCH="${TRAIN_BRANCH:-map-improvements}"

echo "=== SSDLite Training Setup for vast.ai (branch: $BRANCH) ==="

# ---- 1. Clone repo ----
echo "[1/5] Setting up repository..."
if [ -f "setup_vastai.sh" ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "  Already inside repo, syncing to $BRANCH"
    git fetch --quiet origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --quiet --ff-only origin "$BRANCH"
else
    REPO_DIR="MIRPR-pedestrian-and-vehicle-detection-SSDLite"
    if [ ! -d "$REPO_DIR" ]; then
        git clone --branch "$BRANCH" https://github.com/pasandrei/MIRPR-pedestrian-and-vehicle-detection-SSDLite.git
    fi
    cd "$REPO_DIR"
    git checkout "$BRANCH"
    git pull --quiet --ff-only origin "$BRANCH"
fi
echo "  Deployed commit: $(git log -1 --format='%h %s')"

# ---- 2. Install dependencies ----
echo "[2/5] Installing Python dependencies..."
pip install --quiet \
    'albumentations<2' \
    pycocotools \
    tensorboard \
    opencv-python-headless \
    aria2p
apt-get update -qq && apt-get install -y -qq aria2 unzip gcc >/dev/null 2>&1 || true
# torch.compile needs libcuda.so symlink (runtime images only have libcuda.so.1)
if [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ] && [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi

# ---- 3. Download COCO dataset ----
# Use /workspace for persistent storage across vast.ai instances
if [ -d "/workspace" ]; then
    COCO_DIR="/workspace/coco"
    echo "[3/5] Using persistent storage at $COCO_DIR"
else
    COCO_DIR="$(pwd)/data/COCO"
    echo "[3/5] No persistent storage found, using local $COCO_DIR"
fi

mkdir -p "$COCO_DIR"

REPO_ROOT="$(pwd)"

download_coco() {
    local dir="$1"
    cd "$dir"

    HF_BASE="https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/main"

    if [ ! -d "train2017" ]; then
        echo "  Downloading train2017 (~19GB)..."
        aria2c -x 16 -s 16 -q --auto-file-renaming=false -o train2017.zip "$HF_BASE/train2017.zip"
        unzip -q train2017.zip
        rm train2017.zip
    else
        echo "  train2017 already exists, skipping"
    fi

    if [ ! -d "val2017" ]; then
        echo "  Downloading val2017 (~800MB)..."
        aria2c -x 16 -s 16 -q --auto-file-renaming=false -o val2017.zip "$HF_BASE/val2017.zip"
        unzip -q val2017.zip
        rm val2017.zip
    else
        echo "  val2017 already exists, skipping"
    fi

    if [ ! -d "annotations" ]; then
        echo "  Downloading annotations (~250MB)..."
        aria2c -x 16 -s 16 -q --auto-file-renaming=false -o annotations_trainval2017.zip "$HF_BASE/annotations_trainval2017.zip"
        unzip -q annotations_trainval2017.zip
        rm annotations_trainval2017.zip
    else
        echo "  annotations already exists, skipping"
    fi
}

download_coco "$COCO_DIR"
cd "$REPO_ROOT"

# Symlink persistent COCO into repo if using /workspace
if [ -d "/workspace" ]; then
    mkdir -p data
    if [ -L "data/COCO" ]; then
        rm data/COCO
    fi
    ln -sf "$COCO_DIR" data/COCO
    echo "  Symlinked data/COCO -> $COCO_DIR"
fi

# ---- 4. Verify setup ----
echo "[4/5] Verifying setup..."
python -c "
import torch
print(f'PyTorch: {torch.__version__} (built for CUDA {torch.version.cuda})')
assert torch.cuda.is_available(), 'CUDA not available'
props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print(f'GPU: {props.name} | sm_{cap[0]}{cap[1]} | {props.total_memory / 1e9:.1f} GB')
# Real kernel launch — fails loudly with 'no kernel image available' if this
# PyTorch build lacks kernels for the GPU's compute capability (e.g. Blackwell
# sm_120 needs a cu128+ build). is_available() alone does NOT catch this.
_x = torch.randn(2048, 2048, device='cuda')
torch.cuda.synchronize()
print(f'GPU compute test: OK ({(_x @ _x).sum().item():.0f})')

from pathlib import Path
coco = Path('data/COCO')
assert (coco / 'train2017').exists(), 'train2017 not found'
assert (coco / 'val2017').exists(), 'val2017 not found'
assert (coco / 'annotations/instances_train2017.json').exists(), 'train annotations not found'
assert (coco / 'annotations/instances_val2017.json').exists(), 'val annotations not found'
print('COCO dataset: OK')

import albumentations, pycocotools
print('Dependencies: OK')
"

# ---- 5. Print run instructions ----
echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training from scratch (use screen to survive SSH disconnect):"
echo "  screen -S train"
echo "  RUN=b32_lr0165_zoomout20pct_\$(date +%m%d)"
echo "  PYTHONUNBUFFERED=1 python -c \"from main import run; run(mixed_precision=True, run_name='\$RUN')\" > training_\$RUN.log 2>&1 &"
echo "  tail -f training_\$RUN.log"
echo "  # Ctrl+A, D to detach; screen -r train to reattach"
echo ""
echo "To monitor progress:"
echo "  grep -E 'Epoch:|Average Precision.*all.*100|Model saved' training.log | tail -20"
