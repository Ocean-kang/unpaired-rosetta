"""Re-evaluate a saved D/E/DE checkpoint with the paper's *unaltered* metrics."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from experiments.alignment import evaluate

from .run import GeometryRun, build_aligner, to_protocol


def evaluate_checkpoint(path: Path, *, validation_size: int | None = None,
                        skip_classification: bool = False) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    fields = dict(data["config"])
    if validation_size is not None:
        fields["validation_size"] = validation_size
    if skip_classification:
        fields["classification"] = False
    c = GeometryRun(**fields)
    model = build_aligner(c)
    for name in ("mean_x", "mean_y", "weight", "global_weight", "local_centers_x", "local_weights", "local_alphas"):
        if name in data:
            setattr(model, name, data[name])
    return {"config": dataclasses.asdict(c), "metrics": evaluate(model, to_protocol(c))}


def main() -> None:
    p = argparse.ArgumentParser(description="Official FOSCTTM, mean rank, zero-shot accuracy")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--validation-size", type=int, default=None)
    p.add_argument("--skip-classification", action="store_true")
    args = p.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, validation_size=args.validation_size,
                                         skip_classification=args.skip_classification), indent=2))


if __name__ == "__main__":
    main()
