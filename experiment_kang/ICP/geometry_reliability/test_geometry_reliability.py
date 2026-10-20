"""Synthetic, fast unit tests; run from the repository root with pytest -q this_file.

These do not need MS COCO embeddings and do not test the MPOpt QAP solver.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

from .local_graph import effective_sample_size, local_graph_scores, reliability_weights
from .local_mapping import ReliabilityLocalWP
from .robust_refine import GeometryWeightedWP


def test_identical_local_geometries_have_perfect_topology_and_distances():
    torch.manual_seed(7)
    x = F.normalize(torch.randn(48, 12), dim=1)
    scores = local_graph_scores(x, x, k=7, mode="full", weight=torch.eye(12))
    assert torch.allclose(scores["topology"], torch.ones(48), atol=1e-6)
    assert torch.allclose(scores["distance"], torch.ones(48), atol=1e-6)
    shuffled = local_graph_scores(x, x[torch.randperm(48)], k=7, mode="topology")
    assert shuffled["topology"].mean() < 0.5


def test_weights_are_positive_normalized_and_effective_count_bounded():
    r = torch.tensor([0.0, 0.1, 0.8, 1.0])
    w = reliability_weights(r, floor=0.3)
    assert torch.all(w > 0)
    assert torch.allclose(w.mean(), torch.tensor(1.0))
    assert 1 <= effective_sample_size(w) <= 4.0001


def test_uniform_small_batch_matches_official_refinement_step():
    torch.manual_seed(3)
    x, y = F.normalize(torch.randn(40, 8), dim=1), F.normalize(torch.randn(40, 8), dim=1)
    w = torch.eye(8)
    official = WassersteinProcrustes(batch_size=24)
    ours = GeometryWeightedWP(batch_size=32, refinement_batch_size=24, geometry_mode="uniform", step_size=1)
    base = official.refinement_step(x, y, w, None, None, Randomness(17))
    candidate = ours.refinement_step(x, y, w, None, None, Randomness(17))
    assert torch.allclose(candidate, base, atol=1e-5)


def test_alpha_zero_is_exact_original_map():
    torch.manual_seed(3)
    x, y = torch.randn(32, 8), torch.randn(32, 8)
    eye = torch.eye(8)
    model = ReliabilityLocalWP(local_alpha=0, num_local_clusters=4, num_iterations=0)
    model.fit_from_global(x, y, mean_x=x.mean(0, keepdim=True),
                          mean_y=y.mean(0, keepdim=True), weight=eye, seed=11)
    expected = WassersteinProcrustes(num_iterations=0)
    expected.mean_x, expected.mean_y, expected.weight = model.mean_x, model.mean_y, eye
    assert torch.equal(model.transform(x), expected.transform(x))
    assert torch.equal(model.similarity(x[:7], y), expected.similarity(x[:7], y))


def test_e_and_de_soft_corrections_finite_and_bounded():
    torch.manual_seed(41)
    x = F.normalize(torch.randn(96, 12), dim=1)
    y = F.normalize(x @ torch.eye(12) + 0.1 * torch.randn_like(x), dim=1)
    for mode, alpha_mode in [("E", "fixed"), ("DE", "fixed"), ("DE", "reliable"), ("DE", "inverse")]:
        model = ReliabilityLocalWP(
            mode=mode, alpha_mode=alpha_mode, local_alpha=0.25,
            num_local_clusters=3, local_batch_size=30, local_neighbors=4,
            local_iterations=1, global_anchor=3.0, num_iterations=0,
        )
        model.fit_from_global(x, y, mean_x=torch.zeros(1, 12),
                              mean_y=torch.zeros(1, 12), weight=torch.eye(12), seed=23)
        projected = model.transform(x[:12])
        assert projected.shape == (12, 12)
        assert torch.isfinite(projected).all()
        assert torch.allclose(projected.norm(dim=1), torch.ones(12), atol=1e-5)
        assert model.local_alphas is not None
        assert torch.all((model.local_alphas >= 0) & (model.local_alphas <= 0.25))


def test_geometry_step_preserves_semiorthogonality():
    torch.manual_seed(77)
    x = F.normalize(torch.randn(50, 12), dim=1)
    y = F.normalize(torch.randn(50, 9), dim=1)
    w = torch.linalg.svd(torch.randn(12, 9), full_matrices=False)
    w = w.U @ w.Vh
    model = GeometryWeightedWP(batch_size=40, refinement_batch_size=24,
                               geometry_mode="full", local_neighbors=5, step_size=0.1)
    new = model.refinement_step(x, y, w, None, None, Randomness(11))
    assert torch.isfinite(new).all()
    assert torch.allclose(new.T @ new, torch.eye(9), atol=1e-4)
