"""D: confidence-aware *global* refinement, reusing official initialization/readout.

Only ``WassersteinProcrustes.refinement_step`` is replaced.  'uniform' with
step_size=1 and matching refinement batch is identical to original refinement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from unpaired_rosetta.assignment import batched_hungarian_matching
from unpaired_rosetta.linalg import polar
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes, procrustes

from .local_graph import local_graph_scores, reliability_weights


class GeometryWeightedWP(WassersteinProcrustes):
    def __init__(
        self,
        *,
        refinement_batch_size: int = 1024,
        geometry_mode: str = "full",
        local_neighbors: int = 20,
        geometry_temperature: float = 0.15,
        confidence_floor: float = 0.3,
        step_size: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if refinement_batch_size < 2:
            raise ValueError("refinement_batch_size must be >= 2")
        if not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0,1]")
        self.refinement_batch_size = refinement_batch_size
        self.geometry_mode = geometry_mode
        self.local_neighbors = local_neighbors
        self.geometry_temperature = geometry_temperature
        self.confidence_floor = confidence_floor
        self.step_size = step_size
        self.last_reliability: dict[str, float] = {}

    @torch.inference_mode()
    def refinement_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> Tensor:
        # Exact original random subsets, assignment and Procrustes for the
        # uniform control. Initial QAP/readout *retain* the 10k batch size.
        if paired_x is not None or paired_y is not None:
            raise ValueError("This experiment supports strictly unpaired training only")
        batch_x = samples_x[randomness.subset(len(samples_x), self.refinement_batch_size)]
        batch_y = samples_y[randomness.subset(len(samples_y), self.refinement_batch_size)]
        rows, cols = batched_hungarian_matching(batch_x @ weight, batch_y, max(len(batch_x), len(batch_y)))
        matched_x, matched_y = batch_x[rows], batch_y[cols]
        if self.geometry_mode == "uniform":
            candidate = procrustes(matched_x, matched_y, None, None)
        else:
            diag = local_graph_scores(
                matched_x, matched_y, weight=weight, k=self.local_neighbors,
                temperature=self.geometry_temperature, mode=self.geometry_mode,
            )
            self.last_reliability = {name: float(value.mean()) for name, value in diag.items()}
            pair_weights = reliability_weights(diag["reliability"], floor=self.confidence_floor)
            candidate = polar((matched_x * pair_weights[:, None]).T @ matched_y)
        if self.step_size == 1.0:
            return candidate
        return polar((1 - self.step_size) * weight + self.step_size * candidate)
