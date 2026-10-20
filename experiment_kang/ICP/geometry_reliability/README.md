# D/E Geometry Reliability — independent ICP research extension

Designed against `Ocean-kang/unpaired-rosetta` main, Oct 20, 2026. Extract this ZIP **at the repository root**. Additive only: **do not overwrite** existing `experiment_kang/ICP/{aligners,run,evaluate}.py`, `unpaired_rosetta/`, `experiments/`, or `storage/`.

## Scientific purpose

1. **D:** replace whole-batch Gram correlation with per-match kNN topology and local-distance consistency; use soft confidence floors, not hard trimming; test conservative Stiefel updates. **Global pre-processing, QAP initialization, readout and Hungarian are original.**
2. **E:** reuse an *already refined* official global map `Wg`; compute source-derived clusters and target clusters induced by `Wg`; fit each local map via regularized Procrustes; use soft/hard mixture gates and a residual: `normalize(x@Wg + sum_k gate_k(x)*alpha_k*x@(Wk-Wg))`.
3. **D+E:** use D's pseudo-pair geometry score for local Procrustes and to modulate local alpha. Compare fixed / reliability-proportional / inverse-reliability. All computations are unpaired (the two COCO train halves do not share items).

**Important:** kNN agreement of arbitrary Hungarian pseudo-matches is not a ground-truth match probability. Confidence is an *unsupervised heuristic*; the inverse-gating comparator tests a competing hypothesis. Unpaired-Rosetta's original FOSCTTM + zero-shot metrics, not proxy scores, decide effectiveness.

## Install

```bash
cd /path/to/unpaired-rosetta
unzip -o /path/to/experiment_kang_geometry_reliability.zip
# only adds experiment_kang/ICP/geometry_reliability/
export UNPAIRED_ROSETTA_ROOT="$PWD/storage"  # optional; default ./storage
```

Expected existing files (already downloaded by you):

```text
storage/embeddings/coco_train2014/vision/dinov2_vit-b14@224_mean.pt
storage/embeddings/coco_train2014/language/mpnet.pt
storage/embeddings/coco_val2014/vision/dinov2_vit-b14@224_mean.pt
storage/embeddings/coco_val2014/language/mpnet.pt
storage/embeddings/CIFAR-10/vision/dinov2_vit-b14@224_mean.pt
storage/embeddings/CIFAR-10/language_b953684b7ed3a62cbea9fe4e30935ee2/mpnet.pt
storage/embeddings/CIFAR-10/labels.pt
# analogous original CIFAR-100 and ImageNet-100 embeddings and labels
```

## The controlled experiment matrix

| Method | Description | QAP/readout batch | Refinement batch |
| --- | --- | ---: | ---: |
| `Original` | Original `WassersteinProcrustes.fit` | 10000 | 10000 |
| `O-1k` / `D0` | Original refinement, *only* refin. batch = 1024, duplicated sanity control | 10000 | 1024 |
| `D1` | Gram correlation soft weighting | 10000 | 1024 |
| `D2` | Local kNN topology soft weighting | 10000 | 1024 |
| `D3` | Topology + local-distance soft weighting | 10000 | 1024 |
| `D4` | Match similarity + topology + distance soft weighting | 10000 | 1024 |
| `D5` | D4 + projected conservative step (`eta=0.1`) | 10000 | 1024 |
| `E0` | Alpha=0: **exact Original mapping**; no local fitting | 10000 | official original |
| `E1` | Hard gates + alpha=.10, no extra global anchor | 10000 | official original |
| `E2` | Soft gates + alpha=.10, no extra global anchor | 10000 | official original |
| `E3` | Soft gates + alpha=.25, no extra global anchor | 10000 | official original |
| `E4` | Soft residual + alpha=.10, global anchor=3 | 10000 | official original |
| `E5` | Soft residual + alpha=.25, global anchor=3 | 10000 | official original |
| `DE-fixed` | Geometry weights + fixed alpha=.10 + global anchor=3 | 10000 | official original |
| `DE-gated` | Geometry weights + reliability-proportional alpha | 10000 | official original |
| `DE-inverse` | Geometry weights + inverse-reliability alpha; counter-hypothesis | 10000 | official original |

Full source and target local embeddings remain disjoint; **no true image-caption pairs from train or validation enter training**.

## Quick syntax/unit tests

```bash
pixi run --frozen python -m pytest -q experiment_kang/ICP/geometry_reliability/test_geometry_reliability.py
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run --dry-run --methods all --seeds 734796314
```

## Smoke test (small, not comparable to reported numbers)

```bash
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run \
  --methods Original O-1k D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 DE-fixed DE-gated DE-inverse \
  --seeds 734796314 --train-size 256 --validation-size 128 \
  --skip-classification --num-clusters 6 --num-restarts 2 \
  --batch-size 64 --refinement-batch-size 32 --num-iterations 2 \
  --local-clusters 3 --local-batch-size 24 --local-neighbors 5 --local-iterations 1
```

**Official DINOv2-B/14 + MPNet metrics** (omit `--train-size`, `--validation-size` and `--skip-classification`):

```bash
# One seed, D-controlled ablations
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run \
  --methods Original O-1k D0 D1 D2 D3 D4 D5 --seeds 734796314

# E/DE, automatically reuses matching Original checkpoint from old experiment_kang/ICP
# or from our local global cache. If none exists, trains Original once per seed.
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run \
  --methods E0 E1 E2 E3 E4 E5 DE-fixed DE-gated DE-inverse --seeds 734796314

# Five seeds for promising methods (SEEDS imported from original repository)
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run \
  --methods Original O-1k D4 D5 E0 E4 DE-fixed DE-gated \
  --seeds 734796314 576165995 2197670066 839703249 2584932063
```

Run a small parameter sweep without silently altering the original method:

```bash
# E4: alpha=.05 / .1 / .25, same original global map
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run \
  --methods E4 --seeds 734796314 --local-alpha 0.05
```

The override becomes part of the saved configuration, so results are separate. `--overwrite` replaces matching run configurations. `--no-reuse-global` forces retraining the official global map. Both `O-1k` and `D0` intentionally implement the same calculation and should match exactly.

## Evaluation: *identical* to original `experiments.alignment.evaluate`

Every fit automatically invokes `experiments.alignment.evaluate` and saves precisely the original numeric metrics:

- `foscttm`, `mean_rank`, `num_validation_samples` on **COCO val2014** (`40,504` paired items, for evaluation **only**)
- `CIFAR-10_top1`, `CIFAR-10_top5`, `CIFAR-100_top1`, `CIFAR-100_top5`, `ImageNet-100_top1`, `ImageNet-100_top5`, computed by **original zero-shot evaluator** (the paper reports C10 Top1, C100 and IN100 Top5)
- `fit_and_evaluation_seconds`

Note: `mean_rank` and `foscttm` encode the same rank information; classification `top5` **is not** image-text retrieval R@5. To preserve exact original metrics, we do **not** modify `unpaired_rosetta.evaluation` or add R@k by default.

Evaluate any saved checkpoint (model will **not** be refit):

```bash
find storage/checkpoints/experiment_kang_ICP_geometry/runs -name '*.pt'
pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.evaluate \
  --checkpoint storage/checkpoints/experiment_kang_ICP_geometry/runs/<actual-id>.pt
```

Smoke-check saved checkpoints with `--validation-size 128 --skip-classification` (not for comparison).

## Results and checkpoints

```text
storage/results/experiment_kang_ICP_geometry/
    <run-id>.json            # original save_result format
    all_runs.csv
    summary_mean_std.csv     # grouped by ALL configuration fields except seed
storage/checkpoints/experiment_kang_ICP_geometry/
    global/*.pt              # cached Original maps (validated protocol settings)
    runs/*.pt                # complete fitted model / local maps, independent run id
    diagnostics/*.json       # Rk, Neff, alpha, drift per cluster
```

`summary_mean_std.csv` **never mixes 128-sample smoke and 40,504-sample full runs**, unlike the previous ICP summary code. A one-seed std is blank (not zero). Models with different hyperparameters are also kept distinct; select only comparable records when preparing tables.

## Research notes and computational caveats

- The global map is optionally reused only when **training corpus, models, seed, train size, num clusters/restarts, initial/readout batch, and number of iterations all match**; caching never uses validation labels. Cached maps are fitted global W, not an earlier raw QAP `W0`.
- `D*` keep original 10k initialization/readout even when their refinement batch is 1k; these are *controls*, not identical original configurations. Hungarian on 10k can be CPU/memory expensive.
- `E` uses original global `Wg` without modification; `alpha=0` exactly reproduces Original predictions, provided the cached map matches. Multiple local maps are blended and normalized; the overall map is **not guaranteed orthogonal**.
- Soft topology scores have nonzero random-baseline overlap; they are corrected by `k/(n-1)`. Structural reliability is **not calibrated**; the inverse gate compares a plausible opposite hypothesis.
- Local graph consistency depends on Hungarian pseudo-pairs. Its high score is not proof of a true match. Compare on strict same-seed, same-split tests and the original independent zero-shot tasks. Reusing COCO validation to choose hyperparameters would bias claims: reserve a development partition if undertaking extensive tuning.
- No full-scale COCO run is included in the zip; it uses your server's existing embeddings, Pixi and MPOpt/pylibmgm environment.
