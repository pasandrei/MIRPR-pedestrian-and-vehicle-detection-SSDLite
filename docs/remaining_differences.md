# Remaining Differences vs Reference SSDLite Recipes

Full three-way diff of our pipeline (branch `map-improvements`, 0.208 mAP val2017) against
the two reference implementations, done 2026-07-17 after the 660-epoch reproduction run:

- **torchvision** `ssdlite320_mobilenet_v3_large` — 21.3 mAP val2017 (source pinned at v0.15.2,
  cross-checked against installed 0.25.0)
- **TF OD API** `ssdlite_mobilenet_v2_coco.config` — 22.0 mAP on COCO14 minival @300×300; the
  paper's 22.1 @320×320 test-dev config was **never released**, so TF@300 is the reproducible
  reference

## Context worth knowing before chasing the paper number

- The paper's 22.1 is at 320×320 on **test-dev** with a config that isn't public. The released
  TF config is 300×300 and scores 22.0 on **minival** (8k images, different split from val2017).
  These two numbers are only loosely comparable to each other, let alone to our val2017 figure.
- torchvision's 21.3 was trained **from scratch** (no ImageNet init!), with a **reduced-tail**
  MobileNetV3-Large (C5 = 480 ch, 3.44M params total) — it is not a paper reproduction either.
- Realistic target for our MobileNetV2 @320 on val2017: **~21–21.5**. We are at 20.8.

## What is now confirmed reference-grade (no action)

Verified identical to torchvision (file:line refs in repo):

| Component | Ours | torchvision | Match |
|---|---|---|---|
| Anchors | 6 maps (20,10,5,3,2,1), scales 0.2–0.95, 6/cell, **3234 total** | same, **3234 total** | exact |
| Box coder | weights (10,10,5,5), smooth L1 on scaled offsets (`train/loss_fn.py:177`) | same | exact |
| Matcher | IoU 0.5, no ignore band, force-match best anchor per GT (`utils/preprocessing.py:27`) | `SSDMatcher(0.5)` | exact |
| Cls loss | softmax CE incl. background, hard-neg 3:1 per image sorted by CE loss | same | exact |
| Loss norm | both losses / batch-total positives, clamp ≥1, 1:1 loc:cls (`train/loss_fn.py:124`) | same | exact |
| Smooth L1 | β = 1.0 | β = 1.0 | exact |
| Head | DW 3×3 + BN + ReLU6 + 1×1 proj, init N(0, 0.03) (`SSDLite.py:11,75`) | same | exact |
| Schedule | cosine 660 ep, stepped per epoch, →~0 | `CosineAnnealingLR(T_max=660)` | exact |
| Resize | plain warp to 320×320, no letterbox | same | exact |
| Eval decode | conf 0.001, multi-label, per-class NMS | 0.001 / batched_nms | equivalent |

TF also matches on box coder (10,10,5,5), matcher (0.5/0.5, `force_match_for_each_row: true`),
smooth L1 delta 1.0, and anchor scale range 0.2–0.95.

**Implication:** the doc `final_results.md` §6 suggestion (sigmoid/BCE as the top lever) is
**demoted** — torchvision reaches 21.3 with the exact softmax + hard-neg setup we already have.
The loss function is not what separates us from ~21.3.

## Genuine remaining differences (prioritized)

### 1. Weight EMA — TESTED 2026-07-18: peak-neutral, converges ~25% faster
- **TF OD API trains with an exponential moving average of weights by default**
  (`optimizer.proto`: `use_moving_average` default **true**, decay **0.9999**) and evaluates the
  EMA variables. This is silently part of the 22.x recipe. torchvision has no EMA and lands at
  21.3 — plausibly a chunk of the TF-vs-torchvision gap.
- We have none. EMA is typically worth **+0.5–1.0 mAP** on long cosine schedules, for free.
- Implementation: `torch.optim.swa_utils.AveragedModel` with
  `get_ema_multi_avg_fn(0.9999)`, update per optimizer step, run validation/checkpointing on the
  EMA weights. At batch 32, 1/(1−0.9999) = 10k steps ≈ 2.7 epochs of horizon — sensible as-is.
- **Result** (branch `ema`, run `2026-07-16_22-21-34_..._ema_nw6`, same recipe as the 0.2084
  baseline): best mAP **0.2087 @ epoch 500** vs baseline **0.2084 @ epoch 656**. Peak quality is
  a statistical tie (+0.0003), i.e. the TF +0.5–1.0 gain does **not** reproduce here — but EMA
  reaches peak ~156 epochs (~25%) sooner and gives much smoother/lower val loss mid-training.
  After epoch ~500 the EMA mAP drifts down slightly and val classification loss ticks up
  (underlying model mildly overfits; EMA tracks it), while the baseline's late LR decay closes
  the gap for free. First ~10 epochs read very low mAP (EMA cold start) — expected, harmless.
  Verdict: keep for faster convergence / shorter schedules, not as a peak-mAP lever.

### 2. Random-crop aggressiveness — TESTED 2026-07-19: +0.0024, new best 0.2111
- Ours: crop **area** ∈ [0.3, 1.0] (`data/dataset.py:89`) → max zoom-in ≈ **1.8×** linear.
- torchvision `RandomIoUCrop`: w and h ratios sampled **independently** ∈ [0.3, 1.0] → area down
  to 0.09, max zoom-in ≈ **3.3×**. TF `ssd_random_crop`: area ∈ [0.1, 1.0], same ≈3.2×.
- The crop is the SSD recipe's main scale augmentation and it was our single biggest training win
  (+0.008 when introduced). Ours is materially weaker than both references.
- Change: sample `crop_w`, `crop_h` fractions independently from [0.3, 1.0] (torchvision
  semantics) or `area_frac ∈ [0.1, 1.0]` (TF semantics).
- Our acceptance test (max GT coverage ≥ threshold, `dataset.py:110`) matches TF's
  `min_object_covered` semantics; torchvision instead tests IoU(GT, crop window). Keep ours.
- **Result** (branch `wider-crop` = `ema` + area∈[0.1,1.0], run `2026-07-18_07-43-51_..._ema_widercrop_nw6`):
  best mAP **0.2111 @ epoch 569** vs EMA-only 0.2087 — **+0.0024, new all-time best**. Same mild
  overfit signature as the EMA run, slightly later onset: val classification loss bottoms
  ~epoch 490 then rises ~0.5%, mAP plateaus ~0.210–0.211 from ~epoch 550 (train losses still
  falling). Best-mAP checkpointing captures the peak, so the tail costs nothing, but both runs
  now agree the recipe saturates ~epoch 500–570 — a 660-epoch schedule is ~20% wasted compute.

### 3. Extra feature blocks — TESTED 2026-07-21: mAP-neutral, −5.4% params
- Ours: plain depthwise-separable (DW 3×3 s2 → 1×1), last block a k=2 conv (`SSDLite.py:63-73`).
- **Both references use a squeeze bottleneck**: 1×1 → C/2 → DW 3×3 s2 → 1×1 → C
  (torchvision `_extra_block`; TF `layer_depth/2` intermediates). Also explains most of our
  param gap (4.74M vs paper 4.3M).
- The fix exists on branch `fix/extra-block-bottleneck` (written Mar 17, never conclusively
  tested). Rebase onto current `map-improvements` and test.
- **Result** (branch `bottleneck-extra-blocks` = `wider-crop` + cherry-picked 90e72d9, run
  `2026-07-19_20-21-50_..._bottleneck_nw8`): best mAP **0.2112 @ epoch 494** vs wider-crop's
  0.2111 @ 569 — a dead tie, with params down 4,744,308 → **4,490,164** (−5.4%) and identical
  training pace. Same saturation/overfit signature as the two previous runs (val cls loss
  bottoms ~ep503, mAP declines to 0.2093 by 659). Verdict: adopt — same accuracy from a
  smaller model that matches the reference architecture; three runs now agree the recipe
  saturates ~epoch 500.

### 4. LR / effective batch — MEDIUM priority
- torchvision: effective batch **192**, LR **0.15**, warmup only ~1 epoch (LinearLR 1/1000,
  1000 iters). Linear-scaled to our batch 32 that is LR **0.025**; we run **0.0165** (~34% lower)
  with a 10-epoch warmup.
- Internal evidence is mixed (b96@0.1 ≈ b32@0.0165 at 660 ep), but the scaled-torchvision point
  (b32 @ 0.025, or b192 @ 0.15 if VRAM allows) was never run at 660 epochs.

### 5. Full TF recipe cluster (sigmoid + RMSProp + EMA) — LOWER priority, higher variance
- TF trains 90-way **sigmoid** (no background class), NMS-based hard-example miner
  (3:1, min 3 neg/img), **RMSProp with decay 0.9, momentum 0.9, epsilon 1.0**.
- Two important footnotes for any retry:
  - Our v5 BCE collapse predates the loss-normalization fix **and** almost certainly used
    RMSProp's default eps 1e-8. TF's **eps = 1.0** makes RMSProp behave far closer to SGD;
    this detail is likely why naive RMSProp attempts (v4: 0.159, v5: collapse) underperformed.
  - torchvision proves softmax reaches 21.3, so this cluster is only relevant for the last
    ~0.5 toward the TF/paper number, not for closing the current gap.

### 6. Minor deltas — LOW priority (~±0.001 each)
| Item | Ours | Reference |
|---|---|---|
| Backbone BN momentum | 0.1 (PyTorch default; heads 0.01) | tv: 0.03 everywhere; TF: 3e-4 |
| Head 1×1 bias init | PyTorch default uniform | tv: zeros |
| Pre-NMS candidates | top-1000 flat across classes | tv: top-300 **per class** |
| Post-NMS cap | none (COCOeval's maxDets=100 applies) | tv: 300/img; TF: 100/img |
| NMS IoU | 0.5333 | tv: 0.55; TF: 0.6 |
| Warmup | 10 ep from LR/100 | tv: ~1 ep from LR/1000 |
| Input norm | ImageNet mean/std (matches our pretrained backbone — correct as-is) | both: [−1, 1] |
| Weight decay scope | conv weights only (BN/bias excluded) | tv: all params; TF: conv only |
| Zoom-out p=0.2 | we use it (+0.004 for us) | tv ssdlite preset: none; TF: none |

Also noted (housekeeping, not accuracy): variance factors 10/5 are hardcoded in both
`train/loss_fn.py:93` and `misc/model_output_handler.py:24`, shadowing the unused
`DefaultBoxes.scale_xy/scale_wh`; the `ssd_loss` docstring (`loss_fn.py:104`) describes the old
per-image normalization; NMS uses rank-based surrogate scores (`utils/postprocessing.py:33`) —
equivalent given the pre-sort, but fragile.

## Suggested experiment order

1. ~~**EMA**~~ — done: peak-neutral (0.2087 vs 0.2084), ~25% faster convergence (see §1).
2. ~~**Wider crop range**~~ — done: **+0.0024, new best 0.2111** (see §2).
3. ~~**Extra-block bottleneck**~~ — done: mAP-neutral tie at 0.2112, −5.4% params — adopt (see §3).
4. LR 0.025 @ b32 as a cheap schedule probe; use cosine-to-500 (three runs saturate there).
5. Full TF cluster (sigmoid + RMSProp eps=1.0 + EMA) only if chasing the last ~0.5 to 22.
