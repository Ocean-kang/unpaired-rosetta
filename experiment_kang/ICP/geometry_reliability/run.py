"""D/E controlled ablations on DINOv2-B/14 + MPNet; official evaluation.

Run this from the unpaired-rosetta repository root:
    pixi run --frozen python -m experiment_kang.ICP.geometry_reliability.run --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from experiments.alignment import AlignmentRun, evaluate, load_training_data
from modalities import image_captions
from unpaired_rosetta.embeddings import storage_root
from unpaired_rosetta.randomness import Randomness, SEEDS
from unpaired_rosetta.results import result_path, run_id, save_result
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

from .local_mapping import ReliabilityLocalWP
from .robust_refine import GeometryWeightedWP

METHODS = (
    "Original", "O-1k", "D0", "D1", "D2", "D3", "D4", "D5",
    "E0", "E1", "E2", "E3", "E4", "E5", "DE-fixed", "DE-gated", "DE-inverse",
)
GLOBAL_ONLY = {"Original", "O-1k", "D0", "D1", "D2", "D3", "D4", "D5"}


@dataclass(frozen=True)
class GeometryRun:
    method: str
    seed: int
    dataset: str = "coco_train2014"
    model_x: str = "dinov2_vit-b14@224_mean"
    model_y: str = "mpnet"
    num_clusters: int = 30
    num_restarts: int = 30
    batch_size: int = 10000  # QAP initialization + readout + original full-batch refinement
    num_iterations: int = 100
    refinement_batch_size: int = 1024  # only D and O-1k; never changes initialization/readout
    local_neighbors: int = 20
    geometry_temperature: float = 0.15
    confidence_floor: float = 0.3
    step_size: float = 0.1  # D5 only; other global refinements use 1.0
    local_clusters: int = 8
    local_batch_size: int = 512
    local_iterations: int = 3
    soft_temperature: float = 0.1
    local_alpha: float | None = None  # None = per-method preset; explicit override used for alpha sweep
    global_anchor: float | None = None  # None = per-method preset
    min_effective_samples: float = 32
    train_size: int | None = None
    validation_size: int | None = None
    classification: bool = True


def to_protocol(config: GeometryRun) -> AlignmentRun:
    return AlignmentRun(
        dataset=config.dataset, model_x=config.model_x, model_y=config.model_y,
        seed=config.seed, method="ours", train_size=config.train_size,
        validation_size=config.validation_size,
        classification=tuple(image_captions.CLASSIFICATION_DATASETS) if config.classification else (),
    )


def _global_kwargs(c: GeometryRun, workers: int, threads: int) -> dict:
    return dict(
        num_clusters=c.num_clusters, num_restarts=c.num_restarts,
        batch_size=c.batch_size, num_iterations=c.num_iterations,
        num_workers=workers, num_threads=threads, verbose=True,
    )


def build_aligner(c: GeometryRun, workers: int = 1, threads: int = 1):
    common = _global_kwargs(c, workers, threads)
    if c.method == "Original":
        return WassersteinProcrustes(**common)
    if c.method in GLOBAL_ONLY:
        modes = {"O-1k": "uniform", "D0": "uniform", "D1": "gram", "D2": "topology",
                 "D3": "topology_distance", "D4": "full", "D5": "full"}
        return GeometryWeightedWP(
            **common, refinement_batch_size=c.refinement_batch_size, geometry_mode=modes[c.method],
            local_neighbors=c.local_neighbors, geometry_temperature=c.geometry_temperature,
            confidence_floor=c.confidence_floor, step_size=c.step_size if c.method == "D5" else 1.0,
        )
    if c.method not in METHODS:
        raise ValueError(f"Unknown method {c.method}")
    if c.method == "E0" and c.local_alpha not in (None, 0.0):
        raise ValueError("E0 is the exact alpha=0 baseline; do not override its alpha")
    e_presets = {
        "E0": ("soft", 0.0, 3.0),
        "E1": ("hard", 0.10, 0.0),
        "E2": ("soft", 0.10, 0.0),
        "E3": ("soft", 0.25, 0.0),
        "E4": ("soft", 0.10, 3.0),
        "E5": ("soft", 0.25, 3.0),
    }
    if c.method in e_presets:
        local_gate, alpha, anchor = e_presets[c.method]
    else:
        local_gate, alpha, anchor = "soft", 0.10, 3.0
    return ReliabilityLocalWP(
        **common,
        mode="DE" if c.method.startswith("DE-") else "E",
        local_gate=local_gate,
        alpha_mode={"DE-gated": "reliable", "DE-inverse": "inverse"}.get(c.method, "fixed"),
        local_alpha=alpha if c.local_alpha is None else c.local_alpha,
        num_local_clusters=c.local_clusters,
        soft_temperature=c.soft_temperature,
        global_anchor=anchor if c.global_anchor is None else c.global_anchor,
        local_batch_size=c.local_batch_size, local_iterations=c.local_iterations,
        local_neighbors=c.local_neighbors, geometry_temperature=c.geometry_temperature,
        geometry_mode="full", confidence_floor=c.confidence_floor,
        min_effective_samples=c.min_effective_samples,
    )


def _global_compatible(found: dict, desired: GeometryRun) -> bool:
    old = found.get("config", {})
    if old.get("method") not in ("original", "Original"):
        return False
    for key in (
        "dataset", "model_x", "model_y", "seed", "num_clusters", "num_restarts",
        "batch_size", "num_iterations", "train_size",
    ):
        if old.get(key) != getattr(desired, key):
            return False
    return all(key in found for key in ("mean_x", "mean_y", "weight"))


def find_global_cache(c: GeometryRun, root: Path) -> tuple[dict | None, str | None]:
    """Accept only matching *Original* checkpoints, including old experiment_kang/ICP files."""
    folders = [root / "checkpoints" / "experiment_kang_ICP_geometry" / "global",
               root / "checkpoints" / "experiment_kang_ICP"]
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.pt")):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and _global_compatible(ckpt, c):
                    return ckpt, str(path)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
    return None, None


def _save_global_cache(c: GeometryRun, aligner, root: Path) -> Path:
    folder = root / "checkpoints" / "experiment_kang_ICP_geometry" / "global"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.asdict(c)
    cfg["method"] = "Original"
    path = folder / f"{run_id(replace(c, method='Original'))}.pt"
    torch.save({"config": cfg, "mean_x": aligner.mean_x.cpu(),
                "mean_y": aligner.mean_y.cpu(), "weight": aligner.weight.cpu()}, path)
    return path


def _save_checkpoint(c: GeometryRun, aligner, root: Path, global_source: str | None) -> Path:
    folder = root / "checkpoints" / "experiment_kang_ICP_geometry" / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": dataclasses.asdict(c), "mean_x": aligner.mean_x.cpu(),
        "mean_y": aligner.mean_y.cpu(), "weight": aligner.weight.cpu(),
        "global_source": global_source,
    }
    for name in ("global_weight", "local_centers_x", "local_weights", "local_alphas"):
        value = getattr(aligner, name, None)
        if isinstance(value, torch.Tensor):
            checkpoint[name] = value.cpu()
    if isinstance(aligner, ReliabilityLocalWP):
        checkpoint["local_diagnostics"] = aligner.local_diagnostics
    path = folder / f"{run_id(c)}.pt"
    torch.save(checkpoint, path)
    return path


def run_one(
    c: GeometryRun, root: Path, *, workers: int = 1, threads: int = 1,
    overwrite: bool = False, reuse_global: bool = True,
) -> dict | None:
    outdir = root / "results" / "experiment_kang_ICP_geometry"
    result = result_path(outdir, c)
    if result.exists() and not overwrite:
        print(f"[skip] {c.method} seed={c.seed}: {result}", flush=True)
        return None
    # Recreate Randomness per method/seed exactly as original alignment.py.
    rng = Randomness(c.seed)
    protocol = to_protocol(c)
    x, y, paired_x, paired_y = load_training_data(protocol, rng)
    assert paired_x is None and paired_y is None
    model = build_aligner(c, workers, threads)
    started = time.time()
    global_source = None
    if isinstance(model, ReliabilityLocalWP) and reuse_global:
        cached, global_source = find_global_cache(c, root)
        if cached is not None:
            print(f"[global cache] {global_source}", flush=True)
            model.fit_from_global(x, y, mean_x=cached["mean_x"], mean_y=cached["mean_y"],
                                  weight=cached["weight"], seed=c.seed)
        else:
            model.fit(x, y, None, None, rng)
            global_source = str(_save_global_cache(c, model, root))
    else:
        model.fit(x, y, None, None, rng)
        if c.method == "Original":
            global_source = str(_save_global_cache(c, model, root))
    checkpoint_path = _save_checkpoint(c, model, root, global_source)
    metrics = evaluate(model, protocol)  # EXACT original retrieval + zero-shot classifiers.
    elapsed = time.time() - started
    metrics["fit_and_evaluation_seconds"] = elapsed
    save_result(outdir, c, metrics, elapsed)
    diagnostics = {
        "method": c.method, "seed": c.seed, "global_source": global_source,
        "geometry_last_step": getattr(model, "last_reliability", None),
        "local_clusters": getattr(model, "local_diagnostics", None),
    }
    diag_path = root / "checkpoints" / "experiment_kang_ICP_geometry" / "diagnostics" / f"{run_id(c)}.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({"method": c.method, "seed": c.seed, "metrics": metrics,
                      "checkpoint": str(checkpoint_path)}, ensure_ascii=False), flush=True)
    return metrics


def summary(root: Path) -> None:
    """Aggregate strictly comparable settings; NEVER mix smoke/full datasets."""
    import pandas as pd

    output = root / "results" / "experiment_kang_ICP_geometry"
    records = []
    for file in sorted(output.glob("*.json")):
        result = json.loads(file.read_text())
        records.append({**result["config"], **result["metrics"]})
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output / "all_runs.csv", index=False)
    metric_cols = [name for name in (
        "foscttm", "mean_rank", "CIFAR-10_top1", "CIFAR-100_top5", "ImageNet-100_top5",
    ) if name in frame.columns]
    # All configuration dimensions except seed, including train/validation size,
    # must agree before different seeds can be pooled.
    config_cols = set(GeometryRun.__dataclass_fields__)
    group_cols = [c for c in frame if c in config_cols and c != "seed"]
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        fields = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) == 0:
                continue
            rows.append({**fields, "metric": metric, "n_seeds": len(values),
                         "mean": values.mean(), "std": values.std(ddof=1) if len(values) > 1 else float("nan")})
    pd.DataFrame(rows).to_csv(output / "summary_mean_std.csv", index=False)
    print(f"[summary] {output / 'summary_mean_std.csv'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent D/E/DE geometry-reliability experiments")
    p.add_argument("--methods", nargs="+", default=["Original", "O-1k", "D0", "D4", "D5", "E0", "E4", "E5", "DE-fixed", "DE-gated"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--dataset", default="coco_train2014")
    p.add_argument("--model-x", default="dinov2_vit-b14@224_mean")
    p.add_argument("--model-y", default="mpnet")
    p.add_argument("--num-clusters", type=int, default=30)
    p.add_argument("--num-restarts", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--refinement-batch-size", type=int, default=1024)
    p.add_argument("--local-neighbors", type=int, default=20)
    p.add_argument("--geometry-temperature", type=float, default=0.15)
    p.add_argument("--confidence-floor", type=float, default=0.3)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--local-clusters", type=int, default=8)
    p.add_argument("--local-batch-size", type=int, default=512)
    p.add_argument("--local-iterations", type=int, default=3)
    p.add_argument("--soft-temperature", type=float, default=0.1)
    p.add_argument("--local-alpha", type=float, default=None, help="Override method preset for E/DE")
    p.add_argument("--global-anchor", type=float, default=None, help="Override E/DE preset")
    p.add_argument("--min-effective-samples", type=float, default=32)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--validation-size", type=int, default=None)
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-reuse-global", action="store_true", help="Force refitting original W for every E/DE")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    methods = list(METHODS) if args.methods == ["all"] else args.methods
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"Invalid methods {sorted(unknown)}; options: {METHODS}")
    if args.num_clusters < 2 or args.num_restarts < 1 or args.num_iterations < 0:
        raise SystemExit("Invalid global fit parameters")
    if args.refinement_batch_size < 2 or args.local_batch_size < 2:
        raise SystemExit("Batch sizes must be >= 2")
    configurations = [GeometryRun(
        method=method, seed=seed, dataset=args.dataset, model_x=args.model_x, model_y=args.model_y,
        num_clusters=args.num_clusters, num_restarts=args.num_restarts, batch_size=args.batch_size,
        num_iterations=args.num_iterations, refinement_batch_size=args.refinement_batch_size,
        local_neighbors=args.local_neighbors, geometry_temperature=args.geometry_temperature,
        confidence_floor=args.confidence_floor, step_size=args.step_size,
        local_clusters=args.local_clusters, local_batch_size=args.local_batch_size,
        local_iterations=args.local_iterations, soft_temperature=args.soft_temperature,
        local_alpha=args.local_alpha, global_anchor=args.global_anchor,
        min_effective_samples=args.min_effective_samples,
        train_size=args.train_size, validation_size=args.validation_size,
        classification=not args.skip_classification,
    ) for seed in args.seeds for method in methods]
    for c in configurations:
        if args.dry_run:
            print(json.dumps(dataclasses.asdict(c), ensure_ascii=False))
        else:
            run_one(c, storage_root(), workers=args.workers, threads=args.threads,
                    overwrite=args.overwrite, reuse_global=not args.no_reuse_global)
    if not args.dry_run:
        summary(storage_root())


if __name__ == "__main__":
    main()
