"""ICP-inspired aligners that reuse Unpaired-Rosetta's initialization and evaluation contract.

The original ``unpaired_rosetta`` package is not modified. The base class below mirrors the small orchestration
portion of ``WassersteinProcrustes._fit`` so that every variant can keep the official preprocessing, geometric
initialization, read-out and Procrustes polar factor while replacing only refinement behavior.

All variants are designed for the zero-pair setting. They reject known paired examples rather than silently
ignoring them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor
from tqdm.auto import trange

from unpaired_rosetta.assignment import hungarian_matching
from unpaired_rosetta.geometric_initialization import GeometricInitialization, kmeans_centers
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes


def _weighted_procrustes(matched_x: Tensor, matched_y: Tensor, weights: Tensor | None = None) -> Tensor:
    """Weighted orthogonal Procrustes, with the repository's original polar-factor implementation."""
    if matched_x.shape[0] == 0:
        raise ValueError("Cannot estimate a Procrustes map from zero correspondences.")
    if weights is None:
        cross_covariance = matched_x.T @ matched_y
    else:
        weights = weights.to(device=matched_x.device, dtype=matched_x.dtype).flatten()
        weights = weights.clamp_min(0)
        if not torch.isfinite(weights).all() or weights.sum() <= 1e-12:
            weights = torch.ones_like(weights)
        weights = weights / weights.mean().clamp_min(1e-12)
        cross_covariance = (matched_x * weights[:, None]).T @ matched_y
    return polar(cross_covariance)


def _normalized_weights(values: Tensor, temperature: float = 1.0) -> Tensor:
    """Convert arbitrary confidence scores to positive, mean-one weights without extreme domination."""
    values = values.float()
    if values.numel() <= 1:
        return torch.ones_like(values)
    standardized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    weights = torch.softmax(standardized / max(float(temperature), 1e-4), dim=0) * values.numel()
    weights = weights.clamp(0.05, 20.0)
    return weights / weights.mean().clamp_min(1e-6)


def _trimmed_indices(confidence: Tensor, keep_ratio: float) -> Tensor:
    count = confidence.numel()
    if count <= 2 or keep_ratio >= 0.999:
        return torch.arange(count, device=confidence.device)
    keep = max(2, min(count, int(math.ceil(count * keep_ratio))))
    return torch.topk(confidence, k=keep, largest=True).indices


def _assignment(batch_x: Tensor, batch_y: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Hungarian assignment for one batch; returns paired row indices and the score matrix."""
    scores = (batch_x @ weight) @ batch_y.T
    rows, columns = hungarian_matching(scores)
    return rows.to(batch_x.device), columns.to(batch_y.device), scores


def _local_variance(samples: Tensor, num_neighbors: int) -> Tensor:
    """Per-sample scalar local variance estimated from a small within-modality cosine kNN graph."""
    n = samples.shape[0]
    if n < 3:
        return torch.ones(n, dtype=samples.dtype, device=samples.device)
    k = max(1, min(int(num_neighbors), n - 1))
    similarity = samples @ samples.T
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.topk(k=k, dim=1).indices
    differences = samples[neighbors] - samples[:, None, :]
    return differences.square().mean(dim=(1, 2)).clamp_min(1e-6)


def _structure_confidence(
    batch_x: Tensor,
    batch_y: Tensor,
    weight: Tensor,
    graph_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Match a batch, then score each match by how well its matched-neighborhood relations agree.

    The graph comparison is performed only on a bounded batch (``geometry_batch_size``), avoiding an N x N
    similarity matrix over the complete training set.
    """
    rows, columns, scores = _assignment(batch_x, batch_y, weight)
    matched_x, matched_y = batch_x[rows], batch_y[columns]
    sim_x = matched_x @ matched_x.T
    sim_y = matched_y @ matched_y.T
    sim_x.fill_diagonal_(0)
    sim_y.fill_diagonal_(0)
    profile_x = sim_x - sim_x.mean(dim=1, keepdim=True)
    profile_y = sim_y - sim_y.mean(dim=1, keepdim=True)
    denominator = profile_x.norm(dim=1) * profile_y.norm(dim=1)
    correlation = (profile_x * profile_y).sum(dim=1) / denominator.clamp_min(1e-6)
    pair_scores = scores[rows, columns]
    pair_z = (pair_scores - pair_scores.mean()) / pair_scores.std(unbiased=False).clamp_min(1e-6)
    graph_z = (correlation - correlation.mean()) / correlation.std(unbiased=False).clamp_min(1e-6)
    confidence = pair_z + float(graph_weight) * graph_z
    return rows, columns, confidence, correlation


class ICPBase(WassersteinProcrustes):
    """Shared experiment scaffold; initialization and read-out reuse the original implementation."""

    def __init__(
        self,
        *,
        geometry_batch_size: int = 1024,
        keep_ratio: float = 0.7,
        local_neighbors: int = 16,
        graph_weight: float = 0.5,
        stage_clusters: Sequence[int] = (30, 100, 300),
        local_clusters: int = 8,
        local_iterations: int = 3,
        local_batch_size: int = 512,
        local_blend: float = 0.5,
        **kwargs: int | bool,
    ) -> None:
        super().__init__(**kwargs)
        self.geometry_batch_size = max(8, int(geometry_batch_size))
        self.keep_ratio = min(1.0, max(0.1, float(keep_ratio)))
        self.local_neighbors = max(1, int(local_neighbors))
        self.graph_weight = float(graph_weight)
        self.stage_clusters = tuple(sorted({max(2, int(k)) for k in stage_clusters}))
        self.local_clusters = max(2, int(local_clusters))
        self.local_iterations = max(1, int(local_iterations))
        self.local_batch_size = max(8, int(local_batch_size))
        self.local_blend = min(1.0, max(0.0, float(local_blend)))
        self._scale_centers: list[tuple[int, Tensor, Tensor]] = []

    @torch.inference_mode()
    def _fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None,
        paired_y: Tensor | None,
        randomness: Randomness,
    ) -> "ICPBase":
        if (paired_x is not None and paired_x.shape[0] > 0) or (paired_y is not None and paired_y.shape[0] > 0):
            raise ValueError("experiment_kang/ICP implements the unpaired (zero-known-pairs) setting only.")

        # Keep the original repository's centering and row normalization exactly.
        samples_x, samples_y = samples_x.float(), samples_y.float()
        self.mean_x = samples_x.mean(dim=0, keepdim=True)
        self.mean_y = samples_y.mean(dim=0, keepdim=True)
        samples_x = center_and_normalize(samples_x, self.mean_x)
        samples_y = center_and_normalize(samples_y, self.mean_y)
        self._train_x, self._train_y = samples_x, samples_y

        # Reuse the official geometric initialization and read-out rather than reimplementing them.
        initialization = GeometricInitialization(
            self.num_clusters, self.num_restarts, self.batch_size, self.num_workers
        )
        correspondence = initialization(samples_x, samples_y, None, None, randomness)
        weight = self.readout(samples_x, samples_y, correspondence, None, None, randomness)
        self._prepare_fit(samples_x, samples_y, weight, randomness)
        for iteration in trange(self.num_iterations, desc="ICP refinement", disable=not self.verbose, leave=False):
            weight = self._icp_step(samples_x, samples_y, weight, randomness, iteration)
        self.weight = weight
        self._finalize_fit(samples_x, samples_y, randomness)
        return self

    def _prepare_fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        randomness: Randomness,
    ) -> None:
        """Optional precomputation hook."""

    def _finalize_fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
    ) -> None:
        """Optional post-fit hook."""

    def _draw_batch(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        randomness: Randomness,
        max_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        size = min(self.batch_size, samples_x.shape[0], samples_y.shape[0])
        if max_size is not None:
            size = min(size, int(max_size))
        return (
            samples_x[randomness.subset(samples_x.shape[0], size)],
            samples_y[randomness.subset(samples_y.shape[0], size)],
        )

    def _icp_step(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        weight: Tensor,
        randomness: Randomness,
        iteration: int,
    ) -> Tensor:
        raise NotImplementedError


class RobustICP(ICPBase):
    """A: trim low-confidence Hungarian matches and use confidence-weighted Procrustes."""

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        batch_x, batch_y = self._draw_batch(samples_x, samples_y, randomness)
        rows, columns, scores = _assignment(batch_x, batch_y, weight)
        pair_score = scores[rows, columns]
        if min(scores.shape) > 1:
            row_second = scores.topk(k=2, dim=1).values[:, 1]
            col_second = scores.topk(k=2, dim=0).values[1]
            confidence = pair_score + 0.5 * (
                pair_score - row_second[rows] + pair_score - col_second[columns]
            )
        else:
            confidence = pair_score
        keep = _trimmed_indices(confidence, self.keep_ratio)
        weights = _normalized_weights(confidence[keep], temperature=1.0)
        return _weighted_procrustes(batch_x[rows[keep]], batch_y[columns[keep]], weights)


class MultiScaleICP(ICPBase):
    """B: coarse-to-fine refinement using cluster-centre correspondences and instance-level ICP."""

    def _prepare_fit(self, samples_x, samples_y, weight, randomness):
        fit_size = min(10_000, self.batch_size, samples_x.shape[0], samples_y.shape[0])
        fit_x = samples_x[randomness.subset(samples_x.shape[0], fit_size)]
        fit_y = samples_y[randomness.subset(samples_y.shape[0], fit_size)]
        self._scale_centers = []
        for k_requested in self.stage_clusters:
            k = min(k_requested, fit_x.shape[0], fit_y.shape[0])
            if k < 2:
                continue
            centers_x = kmeans_centers(fit_x, k, randomness.kmeans_seed())
            centers_y = kmeans_centers(fit_y, k, randomness.kmeans_seed())
            self._scale_centers.append((k, F.normalize(centers_x, dim=1), F.normalize(centers_y, dim=1)))
        if not self._scale_centers:
            raise ValueError("No valid stage_clusters for the available training sample size.")

    def _stage(self, iteration: int) -> tuple[int, Tensor, Tensor]:
        progress = iteration / max(self.num_iterations, 1)
        index = min(int(progress * len(self._scale_centers)), len(self._scale_centers) - 1)
        return self._scale_centers[index]

    def _cluster_covariance(self, centers_x: Tensor, centers_y: Tensor, weight: Tensor) -> Tensor:
        scores = (centers_x @ weight) @ centers_y.T
        rows, columns = hungarian_matching(scores)
        rows, columns = rows.to(centers_x.device), columns.to(centers_y.device)
        return (centers_x[rows].T @ centers_y[columns]) / max(len(rows), 1)

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        stage_index = min(int(iteration / max(self.num_iterations, 1) * len(self._scale_centers)), len(self._scale_centers) - 1)
        _, centers_x, centers_y = self._scale_centers[stage_index]
        batch_x, batch_y = self._draw_batch(samples_x, samples_y, randomness)
        rows, columns, _ = _assignment(batch_x, batch_y, weight)
        instance_covariance = (batch_x[rows].T @ batch_y[columns]) / max(len(rows), 1)
        cluster_covariance = self._cluster_covariance(centers_x, centers_y, weight)
        # Trust the coarse structure early; give instance correspondences increasing influence later.
        cluster_weights = torch.linspace(0.7, 0.3, steps=len(self._scale_centers)).tolist()
        cluster_weight = cluster_weights[stage_index]
        return polar((1.0 - cluster_weight) * instance_covariance + cluster_weight * cluster_covariance)


class CovarianceAwareICP(ICPBase):
    """C: GICP-inspired local uncertainty weighting with a scalable scalar covariance approximation.

    Full per-pair 768 x 768 covariance inverses are prohibitively expensive. This implementation estimates local
    covariance from within-modality kNN neighborhoods and uses its trace as a stable scalar uncertainty weight.
    """

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        batch_x, batch_y = self._draw_batch(
            samples_x, samples_y, randomness, max_size=self.geometry_batch_size
        )
        rows, columns, _ = _assignment(batch_x, batch_y, weight)
        variance_x = _local_variance(batch_x, self.local_neighbors)
        variance_y = _local_variance(batch_y, self.local_neighbors)
        precision = 1.0 / (variance_x[rows] + variance_y[columns] + 1e-4)
        precision = precision / precision.median().clamp_min(1e-6)
        precision = precision.clamp(0.2, 5.0)
        return _weighted_procrustes(batch_x[rows], batch_y[columns], precision)


class StructureAwareICP(ICPBase):
    """D: use consistency of matched kNN similarity graphs to weight sample-level pseudo-pairs."""

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        batch_x, batch_y = self._draw_batch(
            samples_x, samples_y, randomness, max_size=self.geometry_batch_size
        )
        rows, columns, confidence, _ = _structure_confidence(batch_x, batch_y, weight, self.graph_weight)
        weights = _normalized_weights(confidence, temperature=1.0)
        return _weighted_procrustes(batch_x[rows], batch_y[columns], weights)


class PiecewiseOrthogonalICP(ICPBase):
    """E: fit a global map, then estimate regularized local orthogonal maps per matched semantic region."""

    def _finalize_fit(self, samples_x, samples_y, randomness):
        self.global_weight = self.weight.clone()
        k = min(self.local_clusters, samples_x.shape[0], samples_y.shape[0])
        if k < 2:
            self.local_centers_x = samples_x[:1]
            self.local_weights = self.weight.unsqueeze(0)
            return
        model_x = KMeans(n_clusters=k, init="k-means++", n_init=3, max_iter=100, random_state=randomness.kmeans_seed())
        model_y = KMeans(n_clusters=k, init="k-means++", n_init=3, max_iter=100, random_state=randomness.kmeans_seed())
        labels_x = torch.as_tensor(model_x.fit_predict(samples_x.numpy()), dtype=torch.long)
        labels_y = torch.as_tensor(model_y.fit_predict(samples_y.numpy()), dtype=torch.long)
        centers_x = torch.as_tensor(model_x.cluster_centers_, dtype=samples_x.dtype)
        centers_y = torch.as_tensor(model_y.cluster_centers_, dtype=samples_y.dtype)
        centers_x_unit = F.normalize(centers_x, dim=1)
        centers_y_unit = F.normalize(centers_y, dim=1)
        cluster_scores = (centers_x_unit @ self.global_weight) @ centers_y_unit.T
        rows, columns = hungarian_matching(cluster_scores)
        cluster_map = {int(x_id): int(y_id) for x_id, y_id in zip(rows.tolist(), columns.tolist())}

        local_weights = [self.global_weight.clone() for _ in range(k)]
        for x_cluster in range(k):
            y_cluster = cluster_map.get(x_cluster, x_cluster)
            ids_x = (labels_x == x_cluster).nonzero(as_tuple=True)[0]
            ids_y = (labels_y == y_cluster).nonzero(as_tuple=True)[0]
            if min(ids_x.numel(), ids_y.numel()) < 2:
                continue
            current = self.global_weight.clone()
            for _ in range(self.local_iterations):
                count = min(self.local_batch_size, ids_x.numel(), ids_y.numel())
                pick_x = ids_x[randomness.subset(ids_x.numel(), count)]
                pick_y = ids_y[randomness.subset(ids_y.numel(), count)]
                local_x, local_y = samples_x[pick_x], samples_y[pick_y]
                match_rows, match_columns, _ = _assignment(local_x, local_y, current)
                candidate = polar(local_x[match_rows].T @ local_y[match_columns])
                current = polar((1.0 - self.local_blend) * self.global_weight + self.local_blend * candidate)
            local_weights[x_cluster] = current

        self.local_centers_x = centers_x
        self.local_weights = torch.stack(local_weights)
        self.cluster_map = cluster_map

    @torch.inference_mode()
    def transform(self, samples_x: Tensor) -> Tensor:
        """Map each query using the local orthogonal map of its nearest visual-space cluster."""
        x = center_and_normalize(samples_x.float(), self.mean_x)
        centers = self.local_centers_x.to(device=x.device, dtype=x.dtype)
        local_weights = self.local_weights.to(device=x.device, dtype=x.dtype)
        # Equivalent to nearest-centroid Euclidean assignment, without materializing torch.cdist.
        labels = (2.0 * (x @ centers.T) - centers.square().sum(dim=1)[None, :]).argmax(dim=1)
        mapped = torch.empty((x.shape[0], local_weights.shape[-1]), dtype=x.dtype, device=x.device)
        for cluster in labels.unique().tolist():
            mask = labels == cluster
            mapped[mask] = x[mask] @ local_weights[cluster]
        return F.normalize(mapped, dim=-1)

    @torch.inference_mode()
    def similarity(self, queries_x: Tensor, keys_y: Tensor) -> Tensor:
        return self.transform(queries_x) @ center_and_normalize(keys_y.float(), self.mean_y).T

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        # The global stage is intentionally unchanged; local maps are fitted only after it converges.
        batch_x, batch_y = self._draw_batch(samples_x, samples_y, randomness)
        rows, columns, _ = _assignment(batch_x, batch_y, weight)
        return polar(batch_x[rows].T @ batch_y[columns])


class HybridICP(MultiScaleICP):
    """F: combined robust + multi-scale + graph-consistent refinement (A + B + D)."""

    def _icp_step(self, samples_x, samples_y, weight, randomness, iteration):
        stage_index = min(int(iteration / max(self.num_iterations, 1) * len(self._scale_centers)), len(self._scale_centers) - 1)
        _, centers_x, centers_y = self._scale_centers[stage_index]
        batch_x, batch_y = self._draw_batch(
            samples_x, samples_y, randomness, max_size=self.geometry_batch_size
        )
        rows, columns, confidence, _ = _structure_confidence(batch_x, batch_y, weight, self.graph_weight)
        keep = _trimmed_indices(confidence, self.keep_ratio)
        weights = _normalized_weights(confidence[keep], temperature=1.0)
        instance_covariance = (
            batch_x[rows[keep]] * weights[:, None]
        ).T @ batch_y[columns[keep]] / max(len(keep), 1)
        cluster_covariance = self._cluster_covariance(centers_x, centers_y, weight)
        cluster_weights = torch.linspace(0.7, 0.3, steps=len(self._scale_centers)).tolist()
        cluster_weight = cluster_weights[stage_index]
        return polar((1.0 - cluster_weight) * instance_covariance + cluster_weight * cluster_covariance)


ALIGNER_CLASSES = {
    "A": RobustICP,
    "B": MultiScaleICP,
    "C": CovarianceAwareICP,
    "D": StructureAwareICP,
    "E": PiecewiseOrthogonalICP,
    "F": HybridICP,
}
