# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSD-based object detection on COCO, implementing three model variants selectable via `general_config.model_id`:
- **ssdlite** — MobileNetV2 + SSDLite (depthwise separable convolutions), 80-class COCO
- **resnetssd** — ResNet + SSD300, 80-class COCO (benchmarking baseline)
- **ssdlite_1_class** — Modified SSDLite, person-only detection (~2M params)

## Commands

```bash
# Activate environment
source .venv/bin/activate

# Train (from scratch)
python main.py

# Train with mixed precision
python -c "from main import run; run(mixed_precision=True)"

# Resume from checkpoint
python -c "from main import run; run(load_checkpoint=True)"

# Validation only
python -c "from main import run; run(train_model=False, load_checkpoint=True, validate=True)"

# Monitor training
grep -E 'Epoch:|Average Precision.*all.*100|Model saved' training.log | tail -20
```

## Architecture

### Configuration-Driven Model Switching

Everything switches through `general_config.model_id` (set in `general_config/general_config.py`):
- Hyperparameters load from `misc/experiments/{model_id}/params.json`
- Anchor configs, class mappings, and model construction all key off this single ID
- Stats (best mAP, loss) tracked in `misc/experiments/{model_id}/stats.json`

### Data Pipeline

`data/dataset.py:CocoDetection` → loads COCO images + annotations → applies albumentations augmentation → matches anchors to GT boxes via IOU (`utils/preprocessing.py:map_to_ground_truth`) → returns `[images (B×3×H×W), [bbox_targets, class_targets], image_info]`

Background class ID is **100** (unmatched anchors).

### Model Architecture (SSDLite)

`architectures/backbones/MobileNet.py` → pretrained MobileNetV2 extracts two feature maps (expansion of layer 14 at stride 16, last layer at stride 32) → `architectures/models/SSDLite.py:SSD_Head` adds depthwise separable additional blocks → per-anchor loc (4) and conf (n_classes) predictions.

Output format: `[locs: B×4×anchors, confs: B×n_classes×anchors]`

### Training Loop

`main.py:run()` orchestrates everything. `train/train.py:train()` runs epoch loop with:
- Optional warm-up (linear LR ramp during first epoch)
- LR decay policies: `retina` (step decay at fixed epochs) or `poly` (polynomial)
- Progressive backbone unfreezing for ssdlite (`train/backbone_freezer.py`)
- Mixed precision via `torch.amp.GradScaler` (passed as `scaler` argument)
- Validation every `eval_step` epochs, saves checkpoint when mAP improves

### Inference Pipeline

Raw model output → decode offsets with anchors (`misc/model_output_handler.py`) → confidence threshold → NMS (`utils/postprocessing.py`) → COCO evaluation via pycocotools.

### Anchor System

Defined in `general_config/anchor_config.py`. `utils/preprocessing.py:DefaultBoxes` generates anchor boxes. For each aspect ratio `alpha`, generates both `(w,h)` and `(h,w)` when `only_vertical=False`. Scales are absolute pixel values divided by `fig_size` internally.

## Key Conventions

- All hyperparameters via `train/params.py:Params` class (JSON-backed, attribute access)
- Device set in `general_config/general_config.py` (auto-detects CUDA)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- COCO dataset expected at path defined in `general_config/constants.py:dataset_root`
- Checkpoints saved to `misc/experiments/{model_id}/model_checkpoint`
- TensorBoard logs written to `runs/`
- Temporary COCO eval results written to `fisierul.json` in project root

## Dependencies

Python 3.12, PyTorch (CUDA), torchvision, albumentations 1.x, pycocotools, tensorboard, opencv-python-headless.
