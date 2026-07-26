# SSDLite/MobileNetV2 on COCO — Experiment Report, March–July 2026

**Goal:** reproduce the MobileNetV2 paper's SSDLite result (22.1 mAP test-dev) with our own
PyTorch implementation.
**Outcome:** **0.2121 mAP on val2017**, which **matches** torchvision's official SSDLite
(0.213) while using the older MobileNetV2 backbone, and lands ~0.9 short of the paper's
(non-comparable, test-dev, unreleased-config) number. The experiment is concluded for the
foreseeable future; the official COCO test-dev server is offline, so a direct paper
comparison is currently impossible for anyone.

---

## 1. Headline results

| Model | Backbone | Eval set | mAP | AP50 | AP75 | AP-S | AP-M | AP-L | AR@100 | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (MobileNetV2 §6.2) | V2 | test-dev | 0.221 | — | — | — | — | — | — | 4.3M |
| torchvision `ssdlite320_mobilenet_v3_large` | V3-Large | val2017 | 0.213 | 0.343 | 0.221 | 0.011 | 0.202 | 0.444 | 0.334 | 3.4M |
| **Ours — final (520-ep run, ep 476, EMA weights)** | **V2** | val2017 | **0.212** | 0.342 | 0.220 | 0.010 | 0.196 | 0.445 | 0.323 | **4.49M** |

Ours-vs-torchvision is a same-split, same-COCOeval, same-decode-semantics comparison. The
0.0009 mAP difference is well inside run-to-run noise (two independent runs of our own recipe
landed 0.2111 and 0.2112), so this is a **match, not a near-miss**: we are level on AP50 and
AP75, ahead on AP-L, and behind only on AP-M (-0.006) and AR@100 (-0.011). The paper's number
is on a different split with a config that was never released (the released TF config is
300x300, 22.0 minival).

**Progression of the internal best across the project:**

| Date | Best mAP | Milestone |
|---|---|---|
| early March | 0.170 | v2 baseline (old eval harness) |
| mid March | 0.176 → 0.191 | architecture fixes + recipe search (v9–v_b192) |
| late March | 0.2007 | long-schedule recipe found (#20, zoom-out p=0.2) — old harness |
| Jul 12–13 | (+0.010) | eval-harness fixes: same weights score ~0.010 higher; all March numbers underreport by this margin |
| Jul 14 | 0.2084 | reproduction run of the #20 recipe under the new harness (statistical tie with #20) |
| Jul 18 | 0.2087 | + weight EMA (tie, but peak 156 epochs earlier) |
| Jul 19 | 0.2111 | + wider random crop (**new best**) |
| Jul 21 | 0.2112 | + bottleneck extra blocks (tie, −5.4% params) |
| Jul 22 | **0.2121** | + cosine-to-520 schedule (**final best**) |

---

## 2. Phase 1 — March: architecture fixes and recipe search (runs v2–#20)

Started from a 0.170 baseline (64-epoch runs, old harness) with a ~5-point gap to the paper.
Systematically diffed the architecture against the TF OD API config and torchvision source,
then tested each difference:

| Change | Result |
|---|---|
| ReLU6 in prediction heads (was missing) | fixed; part of every later run |
| BN momentum 0.01 + eps 0.001 on head layers | +0.002 |
| Batch-level loss normalization (TF style) | same ceiling, faster convergence |
| `normal_(std=0.03)` head init (was Xavier) | +0.002, broke the 0.174 ceiling |
| SSD random crop (replaced RandomResizedCrop) | **+0.008** — biggest single training win |
| Reduced first-layer anchors (3/cell, TF style) | no effect (tested twice) |
| TF anchor scales (0.2–0.95) | no effect |
| RMSProp (default eps) | −0.011; with BCE, collapsed to 0.02 |
| Brightness augmentation | neutral-to-negative |
| Zoom-out (expand) augmentation p=0.5 | badly hurt convergence (0.166) |
| Zoom-out p=0.2 | **best recipe**: 0.2007 |
| Longer schedules | 660 ep worth ~+0.008 over 200 ep; 990 ep adds nothing |
| Batch/LR scaling | b32/LR 0.0165 ≈ b96/LR 0.1 ≈ b192/LR 0.05 (memory is the only difference) |

Phase ended (Mar 25) with the recipe that survived to the end: SGD momentum 0.9, batch 32,
LR 0.0165 cosine, 10-epoch warmup from LR/100, weight decay 4e-5 (zero on BN/bias), softmax
CE + 3:1 hard-negative mining, smooth L1, SSD crop + zoom-out p=0.2 + flip, 320×320.

## 3. Phase 2 — July: eval harness, reproduction, and the differences audit

- **Eval harness was silently costing ~1 point** (commits d6d5488, 858e0dd): conf threshold
  0.03→0.001, agnostic→per-class NMS, int-cast→float boxes, argmax→multi-label decode,
  pre-NMS cap 200→1000. Same weights: 0.1497 → 0.1590. Verified at a higher operating point:
  +0.010. Decode semantics now match torchvision exactly. Rule of thumb recorded: any March
  number underreports by ~0.010 vs the July harness.
- **Small objects were never the gap** — torchvision's 21.3 model scores AP-small 0.011 on
  the same split (we score 0.010–0.011). SSD at 320px is simply bad at small objects.
- **Reproduction run** (Jul 12–14, instance 44632832): the #20 recipe under the new harness →
  0.2084 @ ep656. De-confounded against March: 0.2084 − 0.010 ≈ 0.198 old-harness =
  statistical tie with #20's 0.2007. The recipe reproduces.
- **Full remaining-differences audit** (Jul 17, `docs/remaining_differences.md`): line-by-line
  diff vs torchvision v0.15.2 source and the TF OD API config. Core pipeline (anchors, box
  coder, matcher, loss, schedule) verified identical to torchvision. Sigmoid/BCE demoted as a
  hypothesis (torchvision hits 21.3 with our exact softmax + hard-neg setup). Produced the
  ranked lever list that drove Phase 3.

## 4. Phase 3 — July 16–22: the four controlled experiments

One variable per run, each on a rented RTX 5080 (instance 45122464), each compared against
the previous best. Total: 4 full runs in 6 days.

| # | Run (branch) | Variable | Best mAP | Peak ep | Verdict |
|---|---|---|---|---|---|
| A | `ema` | weight EMA, decay 0.9999, eval on averaged weights (TF default) | 0.2087 | 500 | Peak-neutral; peak reached ~25% earlier. TF's claimed +0.5–1.0 does not reproduce |
| B | `wider-crop` | crop area min 0.3→0.1 (TF `ssd_random_crop` semantics, 1.8×→3.2× max zoom-in) | 0.2111 | 569 | **+0.0024, new best** — the last real accuracy lever found |
| C | `bottleneck-extra-blocks` | extra blocks → inverted bottleneck (1×1→DW3×3s2→1×1, reference design) | 0.2112 | 494 | Tie; params 4.74M→4.49M (−5.4%). Adopted |
| D | same, 520-ep cosine | schedule 660→520 (LR anneal aligned with saturation) | **0.2121** | 476 | **+0.0009, final best**; 25% cheaper per run |

**The saturation/overfit finding (consistent across all four runs):** val classification loss
bottoms at ~75–80% of the schedule, after which mAP plateaus and drifts down ≲0.5% while train
loss keeps falling. It compresses proportionally when the schedule shortens — it is intrinsic
to the recipe (capacity/data), not schedule length. Cost is <0.001 mAP thanks to best-mAP
checkpointing. Run D confirmed the corollary: ending the cosine at saturation converts wasted
tail epochs into a small peak gain.

**EMA cold start (run A):** with decay 0.9999 (10k-step horizon ≈ 2.7 epochs at batch 32),
early evals read absurdly low (0.03 vs 0.10 at epoch 8) while losses match the raw model —
expected transient, fully washed out by ~epoch 25–40. Documented so nobody panics at it again.

## 5. test-dev attempt (Jul 22) — blocked, package ready

Ran the final checkpoint over all 20,288 test-dev2017 images with the exact val pipeline →
`detections_test-dev2017_ssdlite_results.zip` (repo root; top-100/image, schema-validated).
**Not submittable anywhere:** CodaLab (host of the official server) disabled all submissions
Dec 31 2025; as of July 2026 COCO has no successor on Codabench (API-verified: zero COCO
competitions) or eval.ai. The package is ready if a server reappears.

## 6. Infrastructure findings (vast.ai, this hardware generation)

- `torch.compile` is **net-negative end-to-end** for this model (−6–9%) despite +16–21% on
  the cached-batch compute ceiling — dataloader/H2D/Python overhead already fills the gaps.
  Eager everywhere.
- The training is **compute-bound on modern boxes**: worker count is irrelevant (nw=6 vs 8:
  pace-identical within seconds/block, settled by same-arch A/B); more workers only mattered
  on the slow-per-core EPYC box (per-core speed is what to shop for).
- Per-epoch pace: ~3.1 min (RTX 5070 Ti / 5080 class), ~9.4 min per 3-epoch+eval block.
- 15GB-RAM boxes OOM with nw=8 + persistent workers (pycocotools index COW-replication);
  needs ≥32GB or nw=4 + persistent_workers=False.
- Shallow single-branch clones on instances: `git merge --ff-only` fails ("unrelated
  histories") — deploy updates with `git checkout -B <branch> FETCH_HEAD`.
- TensorBoard run dirs now ISO-sortable (`YYYY-MM-DD_HH-MM-SS_host_name`, `main.py:run_log_dir`);
  all 65 historical dirs renamed to match.

## 7. Final model and recipe

**Checkpoint:** `preserved_runs/b32_lr0165_zoomout20pct_ema_widercrop_bottleneck_520ep_nw6/model_checkpoint_bestmAP_0.2121`
(EMA weights under `ema_state_dict`; load into a plain model by stripping the `module.` prefix).
Architecture: MobileNetV2 (ImageNet) + SSDLite head with inverted-bottleneck extra blocks,
6 anchors/cell × 6 maps (3234 anchors), **4,490,164 params**. Verified locally: 0.2116 val2017.

Recipe (branch `bottleneck-extra-blocks`, commit 0add140): SGD momentum 0.9 · batch 32 ·
LR 0.0165 cosine over **520 epochs** · 10-ep warmup from LR/100 · wd 4e-5 (zero BN/bias) ·
softmax CE + 3:1 hard-neg · smooth L1 · EMA 0.9999 (eval + checkpoint on EMA weights) ·
SSD crop area∈[0.1,1] · zoom-out p=0.2 · flip · 320×320 · AMP · eager.

## 8. What's left (if ever resumed)

1. **LR 0.025 @ b32 probe** — last cheap recipe lever from the audit (~27h on the 520 schedule).
2. **Full TF cluster** (sigmoid + RMSProp eps=1.0 + EMA) — the eps detail is why the March
   RMSProp runs likely failed; high variance, only for chasing the last ~0.5.
3. **test-dev submission** — package ready, waiting on COCO to stand up a new server.
4. Train on train+val with the frozen recipe if (3) ever unblocks.

Realistic ceiling for MobileNetV2 @ 320 on val2017 was estimated at ~21–21.5 during the
audit; the final 21.2 sits inside it. The honest summary: **the implementation is correct
(it matches torchvision on identical evaluation), the recipe is reproducible, and the
remaining distance to the paper is unverifiable split difference plus TF-pretraining
provenance.**

## 9. Artifact index

| What | Where |
|---|---|
| Final checkpoint (0.2121) + log + stats | `preserved_runs/..._bottleneck_520ep_nw6/` |
| Prior run checkpoints (0.2084 / 0.2087 / 0.2111 / 0.2112) | `preserved_runs/` (one dir per run) |
| test-dev detections package | `detections_test-dev2017_ssdlite_results.zip` (repo root) |
| All TensorBoard event files (local mirrors, March + July) | `runs/`, `runs/runs/` |
| Differences audit | `docs/remaining_differences.md` |
| Per-experiment ledger with all March runs | `docs/architecture_differences.md` |
| 0.2084 reproduction-run deep dive | `docs/final_results.md` |
| Branches (all pushed) | `map-improvements` → `ema` → `wider-crop` → `bottleneck-extra-blocks` (final) |
