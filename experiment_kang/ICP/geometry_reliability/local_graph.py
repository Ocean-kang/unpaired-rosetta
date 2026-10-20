"""Unsupervised geometry reliability of *Hungarian-aligned pseudo-pairs*.

All indices refer to a matched batch after Hungarian assignment, not to the
original COCO image/caption ids. No ground-truth correspondences are accessed.
"""

from __future__ import annotations

import torch
from torch import Tensor


MODES = ("uniform", "similarity", "gram", "topology", "topology_distance", "full")


def local_graph_scores(
    matched_x: Tensor,
    matched_y: Tensor,
    *,
    weight: Tensor | None = None,
    k: int = 20,
    temperature: float = 0.15,
    mode: str = "full",
) -> dict[str, Tensor]:
    """Return soft per-match scores in [0,1] and interpretable components.

    The Hungarian assignment establishes a *temporary* common row index. Its
    consistency is then verified against the neighborhood structures in X/Y.
    Neighborhood overlap is corrected for its random-set expectation k/(n-1).
    Distances are compared on both X and Y kNN neighborhoods, symmetrically.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown geometry mode {mode!r}; expected one of {MODES}")
    if matched_x.ndim != 2 or matched_y.ndim != 2 or matched_x.shape[0] != matched_y.shape[0]:
        raise ValueError("Expected row-aligned 2D pseudo-pairs")
    n = matched_x.shape[0]
    if n == 0:
        raise ValueError("Empty pseudo-pairs")
    dtype, device = matched_x.dtype, matched_x.device
    ones = torch.ones(n, dtype=dtype, device=device)
    if n < 3 or mode == "uniform":
        return {"reliability": ones, "similarity": ones, "topology": ones, "distance": ones, "gram": ones}

    # The input is already center-normalized by the original repository.
    gram_x = matched_x @ matched_x.T
    gram_y = matched_y @ matched_y.T
    neighbors = min(max(int(k), 1), n - 1)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    sx = gram_x.masked_fill(eye, -torch.inf)
    sy = gram_y.masked_fill(eye, -torch.inf)
    nx = sx.topk(neighbors, dim=1).indices
    ny = sy.topk(neighbors, dim=1).indices

    # nx/ny index the *same temporary paired-batch index set*.
    hit = (nx[:, :, None] == ny[:, None, :]).any(dim=2).float().mean(dim=1)
    chance = neighbors / (n - 1)
    topology = ((hit - chance) / max(1.0 - chance, 1e-6)).clamp(0, 1).to(dtype)

    dx = (gram_x.gather(1, nx) - gram_y.gather(1, nx)).abs().mean(dim=1)
    dy = (gram_x.gather(1, ny) - gram_y.gather(1, ny)).abs().mean(dim=1)
    distance = torch.exp(-0.5 * (dx + dy) / max(float(temperature), 1e-6)).clamp(0, 1)

    off_x = gram_x.masked_fill(eye, 0)
    off_y = gram_y.masked_fill(eye, 0)
    # Center only non-self similarities; zeroing diagonal after centering removes
    # the artificial self-similarity peak from the cosine correlation estimate.
    mean_x = off_x.sum(dim=1, keepdim=True) / (n - 1)
    mean_y = off_y.sum(dim=1, keepdim=True) / (n - 1)
    cx = (off_x - mean_x).masked_fill(eye, 0)
    cy = (off_y - mean_y).masked_fill(eye, 0)
    corr = (cx * cy).sum(dim=1) / (cx.norm(dim=1) * cy.norm(dim=1)).clamp_min(1e-7)
    gram = ((corr + 1) / 2).clamp(0, 1)

    if weight is None:
        similarity = ones
    else:
        pair_cos = ((matched_x @ weight) * matched_y).sum(dim=1)
        center = pair_cos.median()
        scale = pair_cos.std(unbiased=False).clamp_min(1e-4)
        similarity = torch.sigmoid((pair_cos - center) / scale).clamp(0, 1)

    reliability = {
        "uniform": ones,
        "similarity": similarity,
        "gram": gram,
        "topology": topology,
        "topology_distance": (topology + distance) / 2,
        "full": 0.5 * similarity + 0.25 * topology + 0.25 * distance,
    }[mode]
    return {
        "reliability": reliability.clamp(0, 1),
        "similarity": similarity,
        "topology": topology,
        "distance": distance,
        "gram": gram,
    }


def reliability_weights(reliability: Tensor, floor: float = 0.3) -> Tensor:
    """Distribution-preserving *soft* weights, normalized to mean one; no trimming."""
    if not 0 < floor <= 1:
        raise ValueError("floor must be in (0, 1]")
    values = floor + (1.0 - floor) * reliability.clamp(0, 1)
    return values / values.mean().clamp_min(1e-8)


def effective_sample_size(weights: Tensor) -> float:
    """Effective sample count of nonnegative, possibly unnormalized weights."""
    weights = weights.clamp_min(0)
    return float((weights.sum().square() / weights.square().sum().clamp_min(1e-8)).item())
