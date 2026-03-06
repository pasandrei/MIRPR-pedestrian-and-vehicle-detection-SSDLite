# SSDLite Training Configuration Comparison

Comparison between our original implementation, the MobileNetV2 paper (arXiv:1801.04381), and the TensorFlow Object Detection API config for SSD MobileNetV2 on COCO.

The paper defers most SSDLite training details to "Open Source TensorFlow Object Detection API [38]", but it does provide the backbone architecture and pretraining recipe in detail.

## What the paper explicitly says

### SSDLite detection (Section 6.2)
- Input resolution: **320×320**
- Reported mAP: **22.1** on COCO test-dev
- Training set: **trainval35k** (not train2017)
- Model params: **4.3M**
- Framework: TensorFlow Object Detection API [38]
- Architecture: replace all regular convolutions in SSD prediction layers with **depthwise separable convolutions** (depthwise followed by 1×1 projection)
- Feature attachment: first SSDLite layer attached to **expansion of layer 15** (output stride 16), second and rest attached on top of **last layer** (output stride 32)

### MobileNetV2 backbone architecture (Sections 3-4)
- **19 residual bottleneck layers** (Table 2) with initial 32-filter conv layer
- **ReLU6** as non-linearity (for robustness with low-precision computation)
- **Linear bottlenecks**: no non-linearity on the narrow (output) layers of each bottleneck — important for maintaining representational power
- **Expansion factor 6** for all main experiments
- **Width multiplier 1** for primary network
- All spatial convolutions use **3×3 kernels**
- **Dropout and batch normalization** during training
- For width multipliers < 1, apply to all layers **except the very last convolutional layer**

### ImageNet backbone pretraining (Section 6.1)
- Optimizer: **RMSPropOptimizer** (decay=0.9, momentum=0.9)
- **Batch normalization** after every layer
- Weight decay: **0.00004**
- Initial learning rate: **0.045**, decay rate **0.98 per epoch**
- **16 GPU async workers**, batch size **96**

Note: we use PyTorch's pretrained MobileNetV2 weights, which may use a different pretraining recipe than the paper's TensorFlow setup.

## Detailed comparison

| Setting | Our Original | Paper (explicit) | TF OD API Config |
|---|---|---|---|
| **Input resolution** | 300×300 | 320×320 | 300×300 |
| **Training set** | train2017 (118k imgs) | trainval35k | — |
| **Optimizer** | SGD (momentum=0.9) | — | RMSProp (momentum=0.9, decay=0.9) |
| **Learning rate** | 0.0026 | — | 0.004 |
| **LR schedule** | Step decay ×0.1 at epochs 42, 55 | — | Exponential decay (factor=0.95, per 800k steps) |
| **Warm-up** | Yes (1 epoch, linear) | — | Not in config |
| **Weight decay** | 0.0002 | 0.00004 (for ImageNet) | 0.00004 (L2 regularizer) |
| **Batch size** | 32 | — | 24 |
| **Training duration** | 64 epochs (~236k steps) | — | 200,000 steps |
| **Loss (classification)** | Softmax cross-entropy | — | Weighted sigmoid (BCE) |
| **Loss (localization)** | Smooth L1 | — | Weighted smooth L1 |
| **Hard negative mining** | Yes (3:1 ratio) | — | Yes (max 3000, 3:1 ratio) |
| **Focal loss** | No | — | No |
| **Batch norm** | Yes | Yes (every layer) | Yes (decay=0.9997, eps=0.001) |
| **Dropout** | No | Yes (mentioned for backbone) | — |
| **Non-linearity** | ReLU6 | ReLU6 | — |
| **Linear bottlenecks** | Yes (no activation on narrow layers) | Yes (critical for performance) | — |
| **Expansion factor** | 6 | 6 | — |
| **Width multiplier** | 1 | 1 | — |
| **Zero weight decay on BN/bias** | Yes | — | Implied (L2 on conv only) |
| **Backbone freezing** | Progressive unfreezing | — | Not mentioned |
| **Backbone pretraining** | PyTorch ImageNet MobileNetV2 | TF ImageNet (RMSProp, lr=0.045) | — |
| **Anchor min scale** | ~0.15 (45/300) | — | 0.2 |
| **Anchor max scale** | ~1.05 (315/300) | — | 0.95 |
| **Anchor aspect ratios** | [2] first layer, [2,3] rest | — | [1.0, 2.0, 0.5, 3.0, 0.333] all layers |
| **Anchor layers** | 6 (feat: 19,10,5,3,2,1) | — | 6 |
| **Anchors per cell** | 4 first, 6 rest | — | 6 all layers |
| **Data augmentation** | Random crop, rotation ±10°, flip, brightness/contrast, grayscale | — | Random horizontal flip + SSD random crop |
| **Evaluation set** | val2017 | test-dev | — |
| **# classes** | 81 (80 + background) | 80 + background | 90 |

## Notes on anchor generation

Our implementation specifies aspect ratios as `[2]` or `[2, 3]` and generates both orientations (wide and tall) via the `DefaultBoxes` class, plus two square anchors. So `[2, 3]` produces 6 anchors per cell, effectively equivalent to TF's `[1.0, 2.0, 0.5, 3.0, 0.333]` + extra square.

The first layer difference (4 vs 6 anchors) means 2×20×20 = 800 fewer anchors on the highest-resolution grid, potentially affecting small object detection.

## Notes on training set

The paper uses **trainval35k** (train2014 + 35k subset of val2014), an older COCO split. We use **train2017** which contains 118,287 images. train2017 is equivalent to trainval35k in content (same images, reorganized), so this should not cause a meaningful difference.

## Architecture items already matching the paper

These are correctly implemented in our codebase:
- ReLU6 non-linearity (`architectures/backbones/MobileNet.py` uses `nn.ReLU6`)
- Linear bottlenecks (no activation on narrow output layers in `InvertedResidual`)
- Expansion factor 6 (in `inverted_residual_setting`)
- Width multiplier 1 (default)
- Depthwise separable convolutions in SSD prediction layers (`SSDLite.py`)
- Feature attachment at expansion of layer 14/15 (stride 16) + last layer (stride 32)

## Changes applied so far

| Setting | Original → Changed |
|---|---|
| Input resolution | 300×300 → **320×320** |
| Weight decay | 0.0002 → **0.00004** |
| Anchor fig_size | 300 → **320** |
| Anchor feat_size | [19,10,5,3,2,1] → **[20,10,5,3,2,1]** |
| Anchor scales | [45,99,153,207,261,280,315] → **[48,106,163,221,278,299,336]** |

## Current config vs paper/TF

| Setting | Current (v2) | Paper / TF | Match? |
|---|---|---|---|
| **Input resolution** | 320×320 | 320×320 | Yes |
| **Weight decay** | 0.00004 | 0.00004 | Yes |
| **Optimizer** | SGD (momentum=0.9) | RMSProp (momentum=0.9, decay=0.9) | No |
| **Learning rate** | 0.0026 | 0.004 | No |
| **LR schedule** | Step decay ×0.1 at epochs 42, 55 | Exponential decay (0.95/800k steps) | No |
| **Warm-up** | Yes (1 epoch, linear) | Not in config | Extra |
| **Batch size** | 32 | 24 | No |
| **Training duration** | 64 epochs (~236k steps) | 200k steps | Close |
| **Loss (classification)** | Softmax cross-entropy | Weighted sigmoid (BCE) | No |
| **Loss (localization)** | Smooth L1 | Weighted smooth L1 | Yes |
| **Hard negative mining** | Yes (3:1 ratio) | Yes (3:1 ratio) | Yes |
| **Batch norm** | Yes | Yes | Yes |
| **Dropout** | No | Mentioned for backbone | No |
| **Non-linearity** | ReLU6 | ReLU6 | Yes |
| **Linear bottlenecks** | Yes | Yes | Yes |
| **Expansion factor** | 6 | 6 | Yes |
| **Width multiplier** | 1 | 1 | Yes |
| **Zero wd on BN/bias** | Yes | Implied | Yes |
| **Backbone freezing** | Progressive unfreezing | Not mentioned | Extra |
| **Backbone pretraining** | PyTorch ImageNet | TF ImageNet | Different source |
| **Anchor min scale** | ~0.15 (48/320) | 0.2 | No |
| **Anchor max scale** | ~1.05 (336/320) | 0.95 | No |
| **Anchor aspect ratios** | [2] first, [2,3] rest | [1,2,0.5,3,0.333] all | Equivalent (see notes) |
| **Anchors per cell** | 4 first, 6 rest | 6 all layers | Close |
| **Data augmentation** | Crop, rotation, flip, color, gray | Flip + SSD crop | Ours is richer |
| **Evaluation set** | val2017 | test-dev | Different |
| **# classes** | 81 | 90 | Minor |

## Remaining differences to investigate

1. Optimizer: SGD → RMSProp
2. Learning rate: 0.0026 → 0.004
3. LR schedule: step decay → exponential decay
4. Loss function: softmax → sigmoid (BCE)
5. Anchor scales: min ~0.15 → 0.2, max ~1.05 → 0.95
6. First layer anchors: 4 → 6
7. Dropout: not used → mentioned in paper for backbone training
8. Backbone pretraining: PyTorch weights vs TF weights (different training recipes)

## Results tracking

| Run | Config Changes | mAP (val2017) |
|---|---|---|
| README (original) | baseline (300×300, wd=0.0002) | 0.177 |
| v1 | fresh run, same config | ~0.167 (stopped at epoch 48) |
| v2 | 320×320, wd=0.00004 | training... |
| Paper | — | 0.221 (test-dev) |
