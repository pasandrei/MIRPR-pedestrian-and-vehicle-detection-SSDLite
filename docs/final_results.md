# Reproduction Run Results: SSDLite / MobileNetV2 on COCO

> **Superseded.** This documents the July 14 reproduction run (0.2084), which was the best
> result at the time of writing. Four further experiments took the campaign to **0.2121**,
> which **matches** torchvision's 0.213. See [`experiment_report.md`](experiment_report.md)
> for the final results. The analysis below is still accurate for this run.

**Run:** `b32_lr0165_zoomout20pct_nw8` · completed 2026-07-14 · 660/660 epochs
**Best checkpoint:** epoch 656 · **mAP 0.2084** (val2017)
**Hardware:** vast.ai instance 44632832 (RTX 5070 Ti), eager (no `torch.compile`)

---

## 1. Final metrics (our model, val2017)

Full COCO `bbox` breakdown at the best checkpoint (epoch 656), scored with our
inference harness (multi-label decode, conf 0.001, per-class NMS, pre-NMS cap 1000):

| Metric | Value |
|---|---|
| **AP @[.50:.95] all** | **0.208** |
| AP @.50 | 0.339 |
| AP @.75 | 0.214 |
| AP small | 0.011 |
| AP medium | 0.196 |
| AP large | 0.435 |
| AR @1 | 0.207 |
| AR @10 | 0.299 |
| AR @100 | 0.323 |
| AR small | 0.041 |
| AR medium | 0.333 |
| AR large | 0.629 |

Model size: **4,744,308 params (~4.74M)**.

---

## 2. Comparison: paper vs PyTorch reproduction vs ours

| | Backbone | Eval set | **mAP** | AP50 | AP75 | AP-S | AP-M | AP-L | AR@100 | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| **Paper** (MobileNetV2 paper, §6.2) | MobileNet**V2** | test-dev | **0.221** | — | — | — | — | — | — | 4.3M |
| **torchvision** `ssdlite320_mobilenet_v3_large` | MobileNet**V3-Large** | val2017 | **0.213** | 0.343 | 0.221 | 0.011 | 0.202 | 0.444 | 0.334 | ~3.4M |
| **Ours** (final, ep656) | MobileNet**V2** | val2017 | **0.208** | 0.339 | 0.214 | 0.011 | 0.196 | 0.435 | 0.323 | 4.74M |

**Fairness of the comparison:**
- **Ours vs torchvision is apples-to-apples** — both scored on val2017 through the
  same pycocotools `COCOeval`, and our eval harness now matches torchvision's decode
  semantics. torchvision numbers obtained by scoring `torchvision_preds.json`.
- **Paper's 0.221 is on test-dev** (a different set; typically within ~0.5 mAP of val)
  and is the only figure the paper reports for SSDLite.

---

## 3. Findings

### vs the PyTorch reproduction: 0.005 short here, matched by the end of the campaign
- **0.208 vs torchvision's 0.213 at this checkpoint**, with every sub-metric close:
  small AP **tied (0.011)**, medium −0.006, large −0.009, AR@100 −0.011.
- The four later experiments (EMA, wider crop, bottleneck blocks, 520-epoch schedule) closed
  this remainder: the campaign's final model scores **0.2121, which matches torchvision's
  0.213** (a 0.0009 difference, inside run-to-run noise). See
  [`experiment_report.md`](experiment_report.md).
- We achieve this with the **older MobileNetV2 backbone** while torchvision uses the
  newer, stronger **MobileNetV3-Large**. torchvision is *not* a reproduction of the
  paper — it is a different (V3) model. Ours is the V2 one.
- The recall gap that looked alarming mid-run (our 0.268 vs their 0.334) **closed to
  −0.011 by convergence**. Small-object AP was never the gap.

### vs the paper — ~1.3 mAP short
- 0.208 (val) vs 0.221 (test-dev) ≈ **−1.3 mAP**. Both are MobileNetV2, so this is the
  true reproduction gap. It is **not** eval-harness, architecture, or small objects.
- Remaining levers are all **training recipe** differences vs the TF OD API setup:
  SGD vs **RMSProp**, softmax cross-entropy vs **weighted sigmoid/BCE**, no dropout,
  and PyTorch's ImageNet-V2 pretraining vs the paper's TF pretraining (RMSProp, lr 0.045).
  (test-dev vs val2017 can also account for up to ~0.5 mAP on its own.)

### Internal reproducibility check
- New harness inflates mAP by **~+0.010** vs the old harness (measured on the ep120
  checkpoint: 0.166 → 0.176). So **0.208 new-harness ≈ 0.198 old-harness**.
- Our previous internal best (`2026-03-25_23-29-33`, same recipe) was **0.2003 old-harness**.
  → **Statistical tie.** This run *reproduced* the best recipe; it did not beat it.
  The TensorBoard curves that appeared "better" were ~70% harness change + ~30% noise.

---

## 4. Recipe (as run)

| Setting | Value |
|---|---|
| Optimizer | SGD (momentum 0.9) |
| Learning rate | 0.0165, cosine schedule |
| Epochs | 660 |
| Warm-up | 10 epochs, linear from LR/100 |
| Batch size | 32 |
| Weight decay | 4e-5 (zero on BN/bias) |
| Loss | softmax cross-entropy + smooth L1 |
| Hard-neg mining | 3:1 |
| Augmentation | SSD zoom-out p=0.2, random crop, flip |
| Input | 320×320, ImageNet norm |
| Backbone | MobileNetV2 (ImageNet pretrained), progressive unfreeze |
| Eval | conf 0.001, suppress 0.5333, eval every 3 epochs |
| Compile | eager (torch.compile measured net-negative end-to-end) |

---

## 5. Artifacts (preserved locally)

`preserved_runs/b32_lr0165_zoomout20pct_nw8_final/`
- `model_checkpoint_bestmAP_0.2084` — best-mAP weights (epoch 656)
- `model_checkpoint_bestloss` — best-val-loss weights
- `stats.json` — `{"loss": 3.7513, "mAP": 0.20842}`
- `training_b32_lr0165_zoomout20pct_nw8.log` — full training log
- `metrics_comparison.json` — machine-readable metrics for this document

TensorBoard event file: `runs/2026-07-12_21-41-07_053af6c9f511_b32_lr0165_zoomout20pct_nw8/`.

---

## 6. Suggested next step to close the paper gap
**Superseded by `docs/remaining_differences.md` (2026-07-17 full diff vs torchvision + TF OD API).**
Key revision: sigmoid/BCE is demoted — torchvision reaches 21.3 with our exact softmax +
hard-neg setup. The top levers are now **weight EMA** (TF trains with EMA decay 0.9999 by
default; we have none) and a **more aggressive random crop** (ours zooms in at most 1.8×,
both references ~3.3×), followed by the inverted-bottleneck extra blocks.
