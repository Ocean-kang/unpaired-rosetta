"""E / D+E: conservative local residual maps atop the *official* global map.

Local stages never see paired labels or validation embeddings. ``fit`` calls the
unaltered WassersteinProcrustes fitting pipeline, while ``fit_from_global``
allows reuse of an exactly-matching global checkpoint. Local randomness uses a
separate seed, so the result does not depend on whether global W is cached.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from unpaired_rosetta.assignment import batched_hungarian_matching
from unpaired_rosetta.geometric_initialization import kmeans_centers
from unpaired_rosetta.linalg import center_and_normalize, polar
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

from .local_graph import effective_sample_size, local_graph_scores, reliability_weights


class ReliabilityLocalWP(WassersteinProcrustes):
    """Global orthogonal alignment with locally anchored, gated residuals.

    mode='E': uniform local covariance; gate='fixed'
    mode='DE': geometry-weighted local covariance; gate='fixed', 'reliable',
    or 'inverse' (the latter is an explicit competing hypothesis).
    """

    def __init__(
        self,
        *,
        mode: str = "E",
        local_gate: str = "soft",
        alpha_mode: str = "fixed",
        local_alpha: float = 0.1,
        num_local_clusters: int = 8,
        soft_temperature: float = 0.1,
        global_anchor: float = 3.0,
        local_batch_size: int = 512,
        local_iterations: int = 3,
        local_neighbors: int = 20,
        geometry_temperature: float = 0.15,
        geometry_mode: str = "full",
        confidence_floor: float = 0.3,
        min_effective_samples: float = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in ("E", "DE") or local_gate not in ("hard", "soft"):
            raise ValueError("Expected mode E/DE and local_gate hard/soft")
        if alpha_mode not in ("fixed", "reliable", "inverse"):
            raise ValueError("Expected alpha_mode fixed/reliable/inverse")
        if not 0 <= local_alpha <= 1 or global_anchor < 0 or soft_temperature <= 0:
            raise ValueError("Invalid local blend, global_anchor or soft_temperature")
        self.local_mode = mode
        self.local_gate = local_gate
        self.alpha_mode = alpha_mode
        self.local_alpha = float(local_alpha)
        self.num_local_clusters = max(int(num_local_clusters), 2)
        self.soft_temperature = float(soft_temperature)
        self.global_anchor = float(global_anchor)
        self.local_batch_size = max(2, int(local_batch_size))
        self.local_iterations = max(1, int(local_iterations))
        self.local_neighbors = max(1, int(local_neighbors))
        self.geometry_temperature = float(geometry_temperature)
        self.geometry_mode = geometry_mode
        self.confidence_floor = float(confidence_floor)
        self.min_effective_samples = float(min_effective_samples)
        self.local_centers_x: Tensor | None = None
        self.local_weights: Tensor | None = None
        self.local_alphas: Tensor | None = None
        self.local_diagnostics: list[dict[str, float]] = []
        self.global_weight: Tensor | None = None
        self._local_seed: int = 0

    @torch.inference_mode()
    def fit(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        paired_x: Tensor | None = None,
        paired_y: Tensor | None = None,
        randomness: Randomness | int = 0,
    ) -> "ReliabilityLocalWP":
        if paired_x is not None or paired_y is not None:
            raise ValueError("Known pairs are prohibited in this unpaired experiment")
        self._local_seed = randomness.seed if isinstance(randomness, Randomness) else int(randomness)
        # The original fit includes its own thread controls and all original
        # initialization, readout, and 100 refinement updates.
        super().fit(samples_x, samples_y, None, None, randomness)
        self.global_weight = self.weight.detach().clone()
        self.fit_local(samples_x, samples_y)
        return self

    @torch.inference_mode()
    def fit_from_global(
        self,
        samples_x: Tensor,
        samples_y: Tensor,
        *,
        mean_x: Tensor,
        mean_y: Tensor,
        weight: Tensor,
        seed: int,
    ) -> "ReliabilityLocalWP":
        """Reuse a compatible global map; caller must validate cached settings."""
        if mean_x.shape != (1, samples_x.shape[1]) or mean_y.shape != (1, samples_y.shape[1]):
            raise ValueError("Global checkpoint has incompatible mean dimensions")
        if weight.shape != (samples_x.shape[1], samples_y.shape[1]):
            raise ValueError("Global checkpoint has incompatible W dimensions")
        self._local_seed = seed
        self.mean_x, self.mean_y = mean_x.float().clone(), mean_y.float().clone()
        self.weight = weight.float().clone()
        self.global_weight = self.weight.clone()
        self.fit_local(samples_x, samples_y)
        return self

    @torch.inference_mode()
    def fit_local(self, samples_x: Tensor, samples_y: Tensor) -> None:
        self.local_centers_x = None
        self.local_weights = None
        self.local_alphas = None
        self.local_diagnostics = []
        if self.local_alpha == 0:
            return  # Exactly the original global map, without a needless kmeans fit.
        random = Randomness(self._local_seed ^ 0x6C0CA1)
        x = center_and_normalize(samples_x.float(), self.mean_x)
        y = center_and_normalize(samples_y.float(), self.mean_y)
        k = min(self.num_local_clusters, len(x), len(y))
        if k < 2:
            return
        count = min(len(x), 10000)
        centers = kmeans_centers(x[random.subset(len(x), count)], k, random.kmeans_seed())
        centers = F.normalize(centers, dim=1)
        # One visual partition and a target partition *induced by global W*.
        # No independently permuted clusters / no ground-truth pair indices.
        labels_x = (x @ centers.T).argmax(dim=1)
        labels_y = (y @ F.normalize(centers @ self.weight, dim=1).T).argmax(dim=1)

        local_weights = []
        local_alphas = []
        for cluster in range(k):
            indices_x = (labels_x == cluster).nonzero(as_tuple=True)[0]
            indices_y = (labels_y == cluster).nonzero(as_tuple=True)[0]
            n = min(self.local_batch_size, len(indices_x), len(indices_y))
            diag: dict[str, float] = {"cluster": float(cluster), "num_visual": float(len(indices_x)),
                                      "num_text": float(len(indices_y)), "matched": float(n)}
            if n < max(self.local_neighbors + 2, 4):
                local_weights.append(self.weight.clone())
                local_alphas.append(0.0)
                diag.update(reliability=0.0, effective_samples=0.0, alpha=0.0, drift=0.0)
                self.local_diagnostics.append(diag)
                continue

            local_weight = self.weight.clone()
            reliability_sum = 0.0
            n_eff = 0.0
            for _ in range(self.local_iterations):
                ids_x = indices_x[random.subset(len(indices_x), n)]
                ids_y = indices_y[random.subset(len(indices_y), n)]
                batch_x, batch_y = x[ids_x], y[ids_y]
                rows, cols = batched_hungarian_matching(
                    batch_x @ local_weight, batch_y, max(len(batch_x), len(batch_y))
                )
                matched_x, matched_y = batch_x[rows], batch_y[cols]
                if self.local_mode == "DE":
                    scores = local_graph_scores(
                        matched_x, matched_y, weight=self.weight, mode=self.geometry_mode,
                        k=self.local_neighbors, temperature=self.geometry_temperature,
                    )
                    confidence = scores["reliability"]
                    weights = reliability_weights(confidence, floor=self.confidence_floor)
                else:
                    confidence = torch.ones(n, dtype=x.dtype)
                    weights = torch.ones_like(confidence)
                reliability_sum += float(confidence.mean())
                n_eff += effective_sample_size(weights)
                covariance = (matched_x * weights[:, None]).T @ matched_y / n
                # Normalize both terms: anchor strength does not depend on n.
                covariance = covariance / covariance.norm().clamp_min(1e-10)
                anchor = self.weight / self.weight.norm().clamp_min(1e-10)
                local_weight = polar(covariance + self.global_anchor * anchor)

            reliability = reliability_sum / self.local_iterations
            effective = n_eff / self.local_iterations
            if self.alpha_mode == "reliable":
                gating = reliability
            elif self.alpha_mode == "inverse":
                gating = 1.0 - reliability
            else:
                gating = 1.0
            sample_factor = min(1.0, effective / max(self.min_effective_samples, 1.0))
            alpha = self.local_alpha * gating * sample_factor
            local_weights.append(local_weight)
            local_alphas.append(alpha)
            diag.update(
                reliability=reliability, effective_samples=effective, alpha=alpha,
                drift=float((local_weight - self.weight).norm() / self.weight.norm().clamp_min(1e-8)),
            )
            self.local_diagnostics.append(diag)
        self.local_centers_x = centers
        self.local_weights = torch.stack(local_weights)
        self.local_alphas = torch.tensor(local_alphas, dtype=x.dtype)

    @torch.inference_mode()
    def transform(self, samples_x: Tensor) -> Tensor:
        x = center_and_normalize(samples_x.float(), self.mean_x)
        mapped = x @ self.weight
        if self.local_centers_x is None or self.local_alpha == 0 or self.local_weights is None:
            return mapped  # Original semantics, important alpha=0 sanity check.
        centers = self.local_centers_x.to(device=x.device, dtype=x.dtype)
        alpha = self.local_alphas.to(device=x.device, dtype=x.dtype)
        if self.local_gate == "hard":
            label = (x @ centers.T).argmax(dim=1)
            gates = F.one_hot(label, num_classes=len(centers)).to(x.dtype)
        else:
            gates = torch.softmax((x @ centers.T) / self.soft_temperature, dim=1)
        # Bounded residual: global W remains an explicit exact special case.
        # A mixture of orthogonal maps is not itself orthogonal; preserve global
        # local geometry only approximately and report its diagnostics.
        for k in range(len(centers)):
            coeff = gates[:, k] * alpha[k]
            if not bool((coeff != 0).any()):
                continue
            delta = self.local_weights[k].to(x.device, x.dtype) - self.weight
            mapped = mapped + coeff[:, None] * (x @ delta)
        return F.normalize(mapped, dim=-1)

    @torch.inference_mode()
    def similarity(self, queries_x: Tensor, keys_y: Tensor) -> Tensor:
        return self.transform(queries_x) @ center_and_normalize(keys_y.float(), self.mean_y).T
