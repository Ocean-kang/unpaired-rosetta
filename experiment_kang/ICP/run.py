"""Run ICP variants on the stored DINOv2-B/14 + MPNet embeddings.

Examples (from the repository root):
    pixi run --frozen python -m experiment_kang.ICP.run --methods original A B C D E F --seeds 734796314
    pixi run --frozen python -m experiment_kang.ICP.run --methods all --seeds 734796314 576165995 2197670066 839703249 2584932063
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modalities import image_captions
from experiments.alignment import AlignmentRun, load_training_data, evaluate
from unpaired_rosetta import settings
from unpaired_rosetta.embeddings import storage_root
from unpaired_rosetta.randomness import SEEDS, Randomness
from unpaired_rosetta.results import result_path, run_id, save_result
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

from experiment_kang.ICP import METHODS
from experiment_kang.ICP.aligners import ALIGNER_CLASSES


@dataclass(frozen=True)
class ICPRun:
    method: str
    dataset: str = "coco_train2014"
    model_x: str = "dinov2_vit-b14@224_mean"
    model_y: str = "mpnet"
    seed: int = 734796314
    num_clusters: int = 30
    num_restarts: int = 30
    batch_size: int = 10_000
    num_iterations: int = 100
    geometry_batch_size: int = 1024
    keep_ratio: float = 0.7
    local_neighbors: int = 16
    graph_weight: float = 0.5
    stage_clusters: tuple[int, ...] = (30, 100, 300)
    local_clusters: int = 8
    local_iterations: int = 3
    local_batch_size: int = 512
    local_blend: float = 0.5
    train_size: int | None = None
    validation_size: int | None = None
    classification: bool = True


def build_aligner(config: ICPRun, workers: int, threads: int):
    common = dict(
        num_clusters=config.num_clusters,
        num_restarts=config.num_restarts,
        batch_size=config.batch_size,
        num_iterations=config.num_iterations,
        num_workers=workers,
        num_threads=threads,
        verbose=True,
    )
    if config.method == "original":
        return WassersteinProcrustes(**common)
    cls = ALIGNER_CLASSES[config.method]
    return cls(
        **common,
        geometry_batch_size=config.geometry_batch_size,
        keep_ratio=config.keep_ratio,
        local_neighbors=config.local_neighbors,
        graph_weight=config.graph_weight,
        stage_clusters=config.stage_clusters,
        local_clusters=config.local_clusters,
        local_iterations=config.local_iterations,
        local_batch_size=config.local_batch_size,
        local_blend=config.local_blend,
    )


def _alignment_run(config: ICPRun) -> AlignmentRun:
    return AlignmentRun(
        dataset=config.dataset,
        model_x=config.model_x,
        model_y=config.model_y,
        seed=config.seed,
        method="ours",
        train_size=config.train_size,
        validation_size=config.validation_size,
        classification=tuple(image_captions.CLASSIFICATION_DATASETS) if config.classification else (),
    )


def run_one(
    config: ICPRun,
    output_dir: Path,
    workers: int = 1,
    threads: int = 1,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    path = result_path(output_dir, config)
    if path.exists() and not overwrite:
        print(f"[skip existing] {path}", flush=True)
        return None

    protocol = _alignment_run(config)
    randomness = Randomness(config.seed)
    # This is the same unpaired split and random stream used by experiments.alignment.run_alignment.
    samples_x, samples_y, paired_x, paired_y = load_training_data(protocol, randomness)
    if paired_x is not None or paired_y is not None:
        raise RuntimeError("Expected a zero-pair experiment; check AlignmentRun.num_pairs.")

    aligner = build_aligner(config, workers, threads)
    started = time.time()
    aligner.fit(samples_x, samples_y, paired_x, paired_y, randomness)
    checkpoint_dir = storage_root() / "checkpoints" / "experiment_kang_ICP"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id(config)}.pt"
    checkpoint = {
        "config": dataclasses.asdict(config),
        "mean_x": aligner.mean_x.detach().cpu(),
        "mean_y": aligner.mean_y.detach().cpu(),
        "weight": aligner.weight.detach().cpu(),
    }
    # E has a local, piecewise map in addition to the global fallback.
    for attribute in ("global_weight", "local_centers_x", "local_weights", "cluster_map"):
        if hasattr(aligner, attribute):
            value = getattr(aligner, attribute)
            checkpoint[attribute] = value.detach().cpu() if hasattr(value, "detach") else value
    torch.save(checkpoint, checkpoint_path)
    metrics = evaluate(aligner, protocol)
    elapsed = time.time() - started
    metrics["fit_and_evaluation_seconds"] = elapsed
    saved = save_result(output_dir, config, metrics, elapsed)
    print(json.dumps({"method": config.method, "seed": config.seed, "metrics": metrics, "saved": str(saved), "checkpoint": str(checkpoint_path)}, indent=2))
    return metrics


def write_summary(output_dir: Path) -> None:
    """Write per-run and mean/std CSV summaries without changing the original result files."""
    import pandas as pd

    records = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            config = record.get("config", {})
            metrics = record.get("metrics", {})
            records.append({**config, **metrics})
        except (OSError, json.JSONDecodeError):
            continue
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "all_runs.csv", index=False)
    metric_columns = [
        col for col in frame.columns
        if col in {"foscttm", "mean_rank", "num_validation_samples"}
        or col.endswith("_top1") or col.endswith("_top5")
    ]
    summaries = []
    for method, group in frame.groupby("method", dropna=False):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            summaries.append({
                "method": method,
                "metric": metric,
                "n": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                "mean_plus_minus_std": f"{values.mean():.6f} ± {(values.std(ddof=1) if values.shape[0] > 1 else 0.0):.6f}",
            })
    pd.DataFrame(summaries).to_csv(output_dir / "summary_mean_std.csv", index=False)
    print(f"[summary] {output_dir / 'summary_mean_std.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICP-inspired Unpaired-Rosetta experiments")
    parser.add_argument("--methods", nargs="+", default=["original", "A", "B", "C", "D", "E", "F"],
                        help="Methods: original A B C D E F, or the single token 'all'.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help=f"Seed(s); default matches the repository's five seeds: {SEEDS}")
    parser.add_argument("--dataset", default="coco_train2014")
    parser.add_argument("--model-x", default="dinov2_vit-b14@224_mean")
    parser.add_argument("--model-y", default="mpnet")
    parser.add_argument("--num-clusters", type=int, default=30)
    parser.add_argument("--num-restarts", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--geometry-batch-size", type=int, default=1024)
    parser.add_argument("--keep-ratio", type=float, default=0.7)
    parser.add_argument("--local-neighbors", type=int, default=16)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--stage-clusters", nargs="+", type=int, default=[30, 100, 300])
    parser.add_argument("--local-clusters", type=int, default=8)
    parser.add_argument("--local-iterations", type=int, default=3)
    parser.add_argument("--local-batch-size", type=int, default=512)
    parser.add_argument("--local-blend", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=None,
                        help="Optional number of training samples per modality; use only for smoke tests/ablations.")
    parser.add_argument("--validation-size", type=int, default=None,
                        help="Optional validation subset. Omit for the full original COCO val split (40,504 items).")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip CIFAR-10/100 and ImageNet-100 zero-shot metrics; retrieval metrics are still evaluated.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for geometric initialization.")
    parser.add_argument("--threads", type=int, default=1, help="Assignment threads used by the original read-out.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory; default: $UNPAIRED_ROSETTA_ROOT/results/experiment_kang_ICP.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute result files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without loading embeddings or running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = list(METHODS) if args.methods == ["all"] else args.methods
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods {unknown}; choose from {METHODS} or use --methods all.")
    if args.num_clusters < 2 or args.num_restarts < 1 or args.num_iterations < 1:
        raise SystemExit("num-clusters >= 2, num-restarts >= 1, and num-iterations >= 1 are required.")
    output_dir = args.outdir or (storage_root() / settings.results_folder_name() / "experiment_kang_ICP")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ICPRun(
            method=method,
            dataset=args.dataset,
            model_x=args.model_x,
            model_y=args.model_y,
            seed=seed,
            num_clusters=args.num_clusters,
            num_restarts=args.num_restarts,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
            geometry_batch_size=args.geometry_batch_size,
            keep_ratio=args.keep_ratio,
            local_neighbors=args.local_neighbors,
            graph_weight=args.graph_weight,
            stage_clusters=tuple(args.stage_clusters),
            local_clusters=args.local_clusters,
            local_iterations=args.local_iterations,
            local_batch_size=args.local_batch_size,
            local_blend=args.local_blend,
            train_size=args.train_size,
            validation_size=args.validation_size,
            classification=not args.skip_classification,
        )
        for method in methods for seed in args.seeds
    ]
    print(f"output_dir={output_dir}")
    print(f"runs={len(configs)}; storage_root={storage_root()}")
    if args.dry_run:
        for config in configs:
            print(dataclasses.asdict(config))
        return
    for config in configs:
        run_one(config, output_dir, args.workers, args.threads, args.overwrite)
    write_summary(output_dir)


if __name__ == "__main__":
    main()
