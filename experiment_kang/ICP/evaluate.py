"""Evaluate a saved ICP alignment checkpoint with the repository's original metrics.

Example:
    pixi run --frozen python -m experiment_kang.ICP.evaluate --checkpoint storage/checkpoints/experiment_kang_ICP/<id>.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from experiment_kang.ICP.run import ICPRun, _alignment_run, build_aligner
from experiments.alignment import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ICP alignment with official Unpaired-Rosetta metrics")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint saved by experiment_kang.ICP.run")
    parser.add_argument("--validation-size", type=int, default=None,
                        help="Optional override; omit to use the validation size stored in the checkpoint config.")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Only compute retrieval metrics; default also evaluates CIFAR-10/100 and ImageNet-100.")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["config"])
    if isinstance(config_values.get("stage_clusters"), list):
        config_values["stage_clusters"] = tuple(config_values["stage_clusters"])
    if args.validation_size is not None:
        config_values["validation_size"] = args.validation_size
    if args.skip_classification:
        config_values["classification"] = False
    config = ICPRun(**config_values)
    # The number of workers/threads affects only initialization/read-out speed, not evaluation.
    aligner = build_aligner(config, workers=1, threads=1)
    aligner.mean_x = checkpoint["mean_x"]
    aligner.mean_y = checkpoint["mean_y"]
    aligner.weight = checkpoint["weight"]
    for attribute in ("global_weight", "local_centers_x", "local_weights", "cluster_map"):
        if attribute in checkpoint:
            setattr(aligner, attribute, checkpoint[attribute])
    protocol = _alignment_run(config)
    metrics = evaluate(aligner, protocol)
    print(json.dumps({"checkpoint": str(args.checkpoint), "config": dataclasses.asdict(config), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
