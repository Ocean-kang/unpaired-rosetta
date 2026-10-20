# ICP-inspired Unpaired-Rosetta experiments

This package adds six isolated experiments under `experiment_kang/ICP/`. It does **not** modify the original
`unpaired_rosetta/` or `experiments/` implementation. It loads the existing stored embeddings and reuses the official
Unpaired-Rosetta data split, centering/normalization, `GeometricInitialization`, read-out, Procrustes polar factor,
retrieval evaluator, and zero-shot classification evaluator.

## Methods

| Name | Method | Main change to original refinement |
|---|---|---|
| `original` | Original Unpaired-Rosetta | Unmodified `WassersteinProcrustes` baseline |
| `A` | Robust ICP | Trim low-confidence Hungarian pseudo-pairs; confidence-weighted Procrustes |
| `B` | Multi-Scale ICP | Progress from cluster-level matches (default 30/100/300 centers) to instance matches |
| `C` | Covariance-Aware ICP | Estimate within-modality local kNN variance and downweight uncertain matches; scalable scalar-covariance approximation to GICP |
| `D` | Structure-Aware ICP | Weight matched pairs by consistency of the matched visual/text similarity graphs |
| `E` | Piecewise Orthogonal ICP | Fit a global map, then regularized local orthogonal maps for independently clustered regions |
| `F` | Hybrid ICP | Combine A + B + D: robust trimming, multiscale cluster constraints and graph consistency |

`F` is explicitly defined as the combined model because the preceding proposal named A–E but did not separately specify F.

All ICP variants implement the **zero-known-pairs** setting and raise an error if paired examples are passed. The local-variance weighting in C is a computationally tractable GICP-inspired approximation; it does not form or invert a full 768x768 covariance matrix for every match.

## Install / placement

Extract the ZIP at the **root of your existing `Ocean-kang/unpaired-rosetta` checkout**, so the result is:

```text
unpaired-rosetta/
├── experiment_kang/
│   ├── __init__.py
│   └── ICP/
│       ├── __init__.py
│       ├── aligners.py
│       ├── run.py
│       └── README.md
├── experiments/
├── unpaired_rosetta/
└── storage/
```

Run all commands from the repository root. The default embedding root is `./storage`, matching the original repository. If your embeddings live elsewhere, set `UNPAIRED_ROSETTA_ROOT` to the directory containing `embeddings/`:

```bash
export UNPAIRED_ROSETTA_ROOT=/absolute/path/to/unpaired-rosetta/storage
```

## 1. Smoke test (small, fast, not a reportable result)

This checks imports, the embedding loader, initialization, all six variants and the evaluator using a small subset.

```bash
pixi run --frozen python -m experiment_kang.ICP.run \
  --methods original A B C D E F \
  --seeds 734796314 \
  --train-size 1024 --validation-size 256 --skip-classification \
  --num-clusters 10 --num-restarts 2 --num-iterations 2 \
  --batch-size 256 --geometry-batch-size 128 \
  --stage-clusters 4 8 12 --local-clusters 4 --local-batch-size 64
```

Do not use the smoke-test output to compare with paper or full-validation metrics; the train and validation subsets are intentionally smaller.

## 2. Full evaluation on DINOv2-B/14 + MPNet

The defaults use the stored `coco_train2014` embeddings for unpaired training and `coco_val2014` for paired validation. With no `--train-size` or `--validation-size`, the original split logic is used: disjoint unpaired training halves and the full 40,504-item COCO validation set. Zero-shot metrics use the same CIFAR-10, CIFAR-100 and ImageNet-100 embeddings and labels as `experiments/unpaired.py`.

First run one seed to validate runtime and output:

```bash
pixi run --frozen python -m experiment_kang.ICP.run \
  --methods original A B C D E F \
  --seeds 734796314 \
  --num-clusters 30 --num-restarts 30 --batch-size 10000 --num-iterations 100 \
  --geometry-batch-size 1024 --local-clusters 8
```

Then run the repository's standard five seeds:

```bash
pixi run --frozen python -m experiment_kang.ICP.run \
  --methods original A B C D E F \
  --seeds 734796314 576165995 2197670066 839703249 2584932063
```

For a full comparison, keep all parameters identical across methods. Variant-specific parameters are saved in each result config.

## 3. Evaluation metrics (same implementation as original)

By default the runner calls `experiments.alignment.evaluate`, which uses the original metrics:

- `foscttm` (lower is better)
- `mean_rank` (lower is better)
- `CIFAR-10_top1`
- `CIFAR-100_top5`
- `ImageNet-100_top5`
- `num_validation_samples`
- `fit_and_evaluation_seconds`

To run retrieval only, add `--skip-classification`. This is useful for debugging but does not reproduce the full original metric set.

The fitted map is also saved as a checkpoint so evaluation can be repeated without retraining. Results are written to:

```text
$UNPAIRED_ROSETTA_ROOT/results/experiment_kang_ICP/
├── <one JSON per method/seed>.json
├── all_runs.csv
└── summary_mean_std.csv

$UNPAIRED_ROSETTA_ROOT/checkpoints/experiment_kang_ICP/
└── <one .pt checkpoint per method/seed>.pt
```

Existing JSON results are skipped on restart. Add `--overwrite` to recompute them. The summary CSV reports each metric's mean and standard deviation grouped by method. Use `--dry-run` to print configs without running.

## 4. Evaluate a saved checkpoint separately

The training command already runs evaluation after fitting. To re-run the original evaluation later without retraining, use the checkpoint path printed at the end of a run:

```bash
pixi run --frozen python -m experiment_kang.ICP.evaluate \
  --checkpoint storage/checkpoints/experiment_kang_ICP/<checkpoint-id>.pt
```

By default this computes the full original metric set using the saved seed and validation configuration. Add `--skip-classification` for retrieval-only evaluation or `--validation-size 5000` to evaluate a smaller validation subset. The latter changes the evaluation sample count and should not be compared directly with the full 40,504-item results.

## 5. Notes for interpreting results

- `original` is the official `WassersteinProcrustes` class instantiated with the same runtime hyperparameters as the variants.
- The initial geometric alignment and read-out are unchanged for A–F; the changes are confined to refinement or (E) a post-fit local-map stage.
- A and B can use the full `--batch-size` for assignment; C/D/F cap their neighborhood-geometry batch at `--geometry-batch-size` because graph/covariance construction scales quadratically in that batch size.
- E adds local maps after the global fit and is therefore the most expensive variant at inference time. It preserves a global fallback map for sparse clusters.
- First compare a single seed and confirm that all expected metric keys are present; then run the five-seed experiment and compare means **and** standard deviations. No performance improvement is assumed before measurement.
