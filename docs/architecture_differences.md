# Architectural Differences: Our SSDLite vs TF OD API SSDLite

Comparison between our PyTorch implementation and the TensorFlow Object Detection API's
`ssdlite_mobilenet_v2_coco` configuration. Goal: close the ~5 mAP gap (ours: 17.1, paper: 22.1).

## 1. Prediction heads missing ReLU6 — HIGH impact — FIXED

**Ours (before fix):** depthwise → BN → pointwise (no activation)
**TF OD API / torchvision:** depthwise → BN → **ReLU6** → pointwise

The non-linearity after depthwise BN allows the prediction heads to learn more expressive
features before the linear projection.

**Fix:** Added `nn.ReLU6(inplace=True)` to `DepthWiseConv_No_ReLu` in `SSDLite.py`.

## 2. Anchor count mismatch — HIGH impact — TESTED

| | Ours | TF OD API |
|---|---|---|
| Input size | 320×320 | 300×300 |
| First layer anchors/cell | 6 | 3 (`reduce_boxes_in_lowest_layer`) |
| Other layers anchors/cell | 6 | 6 |
| Feature map sizes | 20, 10, 5, 3, 2, 1 | 19, 10, 5, 3, 2, 1 |
| **Total anchors** | **3234** | **~1917** |

TF reduces the densest feature map (first layer) to only 3 anchors per cell. We have 6 on
the 20×20 map = ~1200 extra negative anchors, which affects hard negative mining balance.

**Tested (experiment #6):** Reduced first layer to 3 anchors/cell (drop aspect ratio 3 + extra
scale), total anchors 3234 → 2034. Combined with loss norm + weight init.
Result: **0.176 mAP** — same ceiling as weight init alone, no improvement from fewer anchors.

## 3. Loss normalization — MEDIUM impact — TESTED

**Ours:** Normalize each image's loss by its positive anchor count, then mean over batch.
```
loss = mean(loss_per_image / pos_count_per_image)
```

**TF OD API:** Sum all losses, divide by total positives across the entire batch.
```
loss = sum(all_losses) / sum(all_pos_counts)
```

Our method amplifies gradients from images with few objects. TF gives equal weight to each
positive anchor regardless of which image it came from.

**Tested (experiment #3, v12 local):** Result: **0.174 mAP** — matched v11 ceiling but converged
faster (0.174 at epoch 79 vs epoch 83).

## 4. Weight initialization — LOW impact — TESTED

| | Ours | TF / torchvision |
|---|---|---|
| Method | `xavier_uniform_` | `normal_(mean=0, std=0.03)` |

Smaller initial weights (normal 0.03) produce near-zero initial predictions, which is
generally better for detection heads.

**Tested (experiment #4, vast.ai):** Result: **0.176 mAP** — new best, broke through the 0.174
ceiling. Converged fastest (0.174 at epoch 76 vs v11's epoch 83).

## 5. BN momentum on head layers — LOW impact — APPLIED (v7)

Applied intermediate BN momentum=0.01 (between PyTorch default 0.1 and TF's 0.0003) and
eps=0.001 on new layers only (additional_blocks, loc, conf). Backbone keeps PyTorch defaults.

Result: v7 peaked at 0.172, marginal improvement over baseline 0.170.

## 6. Anchor scales — LOW impact — APPLIED (v8)

Adjusted scales from `[48, 106, 163, 221, 278, 299, 336]` to `[64, 112, 160, 208, 256, 304, 320]`
to match TF's min_scale=0.2, max_scale=0.95.

Result: v8 peaked at 0.171, no improvement.

## Training results tracker

### Early runs (exploring architecture and hyperparameters)

| Run | Changes | Best mAP | Epochs |
|---|---|---|---|
| v2 | Baseline: SGD, softmax, 64 epochs, 4 anchors first layer | 0.170 | 56 |
| v3 | v2 variant | 0.165 | 47 |
| v4 | + RMSProp, flat LR 0.004, 6 anchors all layers | 0.159 | 64 |
| v5 | + RMSProp, BCE, no rotation | 0.021 | 26 (collapsed) |
| v6 | SGD, softmax, 6 anchors, no rotation | 0.170 | 59 |
| v7 | v6 + BN momentum=0.01 on head layers | 0.172 | 56 |
| v8 | v7 + TF anchor scales, 100 epochs | 0.171 | 100 |

### Architecture fixes (matching TF OD API / paper)

| Run | Changes | Best mAP | Epochs |
|---|---|---|---|
| v9 | v8 + ReLU6 in prediction heads | 0.172 | 83 |
| v10 | v9 + cosine LR (broken run) | 0.128 | 21 |
| v11 | v9 + cosine LR, batch 64, 100 epochs, 320×320 | 0.174 | 100 |

### Current `map-improvements` branch

Cumulative changes merged: ReLU6 fix + cosine LR + batch-level loss normalization +
normal_(std=0.03) weight init + SSD random crop augmentation + reduced first-layer anchors
(3/cell). Best result so far: **0.2007 mAP** (batch 32, LR 0.0165, 660 epochs, zoom-out
augmentation at p=0.2 — see run #20 in the vast.ai table below).

### Experiments (individual changes tested on top of v11)

| # | Run | Change tested | Best mAP | Peak epoch | vs v11 |
|---|---|---|---|---|---|
| 3 | v12 | Batch-level loss normalization | 0.174 | 79 | Same ceiling, faster convergence |
| 4 | v_wi | normal_(std=0.03) weight init | **0.176** | 76 | **+0.002**, new best |
| 5 | v13 | Loss norm + weight init (combined) | 0.174 | 82 | Same as v11/v12 ceiling, no additive gain |
| 6 | v_ra | Loss norm + weight init + reduced anchors (3/cell first layer) | 0.176 | 88 | Same as #4, anchors didn't help |
| 7 | v_sc | Loss norm + weight init + SSD random crop | **0.182** | 98 | **+0.008**, new best, not yet plateaued |
| 8 | v_b128 | All map-improvements + batch 128, LR 0.02, 200 epochs | **0.188** | 158 | **+0.014**, new best, plateaued ~epoch 148 |
| 9 | v_ra2 | All map-improvements + reduced anchors (3/cell first layer) + SSD crop | 0.177 | 63 | +0.003, stopped early, still climbing |
| 10 | v_ra_b128 | Reduced anchors + batch 128, LR 0.02, 200 epochs | 0.187 | 174 | Slightly worse than 6 anchors, slower convergence |
| 11 | v_b192 | Batch 192, LR 0.05, 5-epoch warmup (LR/100), 200 epochs | **0.191** | 199 | **+0.017**, new best, still climbing at epoch 199 |

### Long runs on vast.ai (March 12–26)

Results recovered from TensorBoard event files in `runs/runs/` (vast.ai containers
`179ccc2b9283`, `d6b2ae0edccd`); these were never logged here at the time. Configs
reconstructed from run names (`b32_lr0165` = batch 32, LR 0.0165), the March 17 stash
(`stash@{0}`), and `training_660ep_lr01_b96.log` (local twin of run #14). Runs marked
"config not recorded" left no trace beyond the event file.

| # | TensorBoard run | Config | Best mAP | Peak epoch |
|---|---|---|---|---|
| 12 | 2026-03-12_11-44-20 | 660 epochs, eval every 2 (config not recorded) | 0.195 | 637 |
| 13 | 2026-03-13_22-26-09 | 660 epochs (config not recorded) | 0.196 | 609 |
| 14 | 2026-03-15_09-50-45 | SGD, batch 96, LR 0.1, cosine 660 ep, 10-ep warmup (LR/200 start) | **0.199** | 639 |
| 15 | 2026-03-16_23-41-23 | 660 ep, batch-level loss norm (sum/num_batches) | 0.197 | 651 |
| 16 | 2026-03-18_09-28-11 | 990 epochs, eval every 3 | 0.199 | 986 |
| 17 | 2026-03-22_14-21-50 | batch 32, LR 0.0165, 660 ep | 0.197 | 644 |
| 18 | 2026-03-23_23-21-08 b32_lr0165_brightness | + RandomBrightnessContrast | 0.196 | 644 |
| 19 | 2026-03-25_13-29-55 b32_lr0165_zoomout | + zoom-out augmentation @ p=0.5 | 0.166 | 188 (stopped ~ep 210) |
| 20 | 2026-03-25_23-29-33 b32_lr0165_zoomout20pct | SGD, batch 32, LR 0.0165, cosine 660 ep, 10-ep warmup (LR/100 start), zoom-out p=0.2 (config confirmed by author) | **0.201** | 635 |

**Takeaways:** long schedules (660 ep) are worth ~+0.008 over the 200-epoch runs and peak
around epoch 610–650; zoom-out at p=0.5 badly hurts convergence but p=0.2 gave the all-time
best **0.2007**; brightness augmentation was neutral-to-negative; batch 32 with LR 0.0165
matches batch 96/LR 0.1 at a fraction of the memory.

### Future experiments

- ~~400-660 epoch run with batch 192, LR 0.05, 5-epoch warmup~~ — done (runs #12–16), plateaus ~0.199; 990 epochs added nothing over 660
- ~~SSD expand (zoom-out) augmentation~~ — done: p=0.2 → 0.201 (best); p=0.5 harmful
- ~~Small object AP (~0.009) is the gap~~ — disproven: torchvision's 21.3-mAP SSDLite scores 0.011
  small on the same val2017 (scored from `torchvision_preds.json`); SSD@320 is just bad at small objects
- **The real gap is recall**: our AR@100 = 0.268 vs torchvision's 0.334 (AP medium −0.019, large −0.024).
  Suspects: eval `conf_threshold` 0.03 (torchvision keeps the ~0.001 tail, top-100) and
  `agnostic_nms = True` (COCO convention is per-class NMS). Fix the eval harness before training anything
- **Eval harness fixed (commit d6d5488): +0.006 mAP for free.** conf 0.03→0.001, per-class NMS,
  pre-NMS cap 200→400, float boxes (they were int-cast twice, hurting AP75). On the local ep-45
  checkpoint: 0.1497 → 0.1557, AR@100 0.216 → 0.243. All historical numbers in the tables above were
  measured with the old harness and underreport by roughly this margin
- **Multi-label decode + pre-NMS cap 1000 (commit 858e0dd): another +0.003.** Each anchor now emits
  every class above threshold instead of only its argmax. Cumulative harness fixes on the ep-45
  checkpoint: **0.1497 → 0.1590** (AR@100 0.216 → 0.274) with zero training. The eval harness now
  matches torchvision's decode semantics; remaining AR gap to torchvision's 0.334 is model quality,
  to be re-measured after re-running the 0.2007 recipe
- Test `fix/extra-block-bottleneck` (inverted-bottleneck extra blocks matching reference SSDLite) —
  written Mar 17, never conclusively tested; also closes the param-count gap (ours 4.74M vs paper 4.3M)
- **Full remaining-differences audit done 2026-07-17 — see `docs/remaining_differences.md`.**
  Core pipeline (anchors, box coder, matcher, loss, schedule) verified identical to torchvision;
  remaining levers ranked: weight EMA (TF default, decay 0.9999) > wider crop range (our min
  area 0.3 vs references ~0.1) > bottleneck extra blocks > LR 0.025@b32 > TF cluster
  (sigmoid + RMSProp eps=1.0 — v4/v5 likely failed on the eps detail)
- **Bottleneck extra blocks tested (branch `bottleneck-extra-blocks`, run 2026-07-19_20-21-50,
  inverted 1×1→DW3×3s2→1×1 blocks matching reference): best mAP 0.2112 @ epoch 494 — dead tie
  with wider-crop's 0.2111, params 4.74M → 4.49M (−5.4%). Adopt: same accuracy, smaller model,
  matches reference architecture. Third consecutive run saturating ~ep500 → future schedules
  should use cosine-to-500. num_workers 8 vs 6 on the 12-vCPU box: pace-identical (compute-bound)
- **Wider random crop tested (branch `wider-crop`, run 2026-07-18_07-43-51, crop area
  [0.3,1.0]→[0.1,1.0] matching TF): best mAP 0.2111 @ epoch 569 — +0.0024 over EMA-only,
  NEW ALL-TIME BEST.** Same mild overfit as the EMA run (val cls loss bottoms ~ep490, mAP
  plateaus ~0.210 from ~ep550); recipe saturates ~ep500–570, 660-epoch schedule is ~20% waste
- **Weight EMA tested (branch `ema`, run 2026-07-16_22-21-34, decay 0.9999): peak-neutral.**
  Best mAP 0.2087 @ epoch 500 vs no-EMA baseline 0.2084 @ epoch 656 on the identical recipe —
  a tie on peak quality (TF's +0.5–1.0 doesn't reproduce), but peak reached ~156 epochs (~25%)
  earlier with much lower/smoother val loss mid-training. Next lever: wider crop range
