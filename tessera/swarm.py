"""
tessera.swarm — Swarm aggregation: many-to-one collective knowledge transfer.

SwarmAggregator enables a distributed training pattern where many small models
contribute knowledge tokens and a single large aggregator model is updated
from the collective. Optionally, the aggregator can broadcast its updated
state back to contributors so their local models improve too.

Pattern:
    1. Many small models train locally on their own data
    2. Each encodes its knowledge to UHS and sends a TesseraToken
    3. SwarmAggregator combines those tokens (mean, weighted, trimmed, median)
    4. Aggregator model decodes the collective hub vector and fine-tunes
    5. (Optional) Aggregator encodes its updated state back as per-contributor tokens

Use cases:
    - Distributed training cooperative (contributors get free model usage)
    - Privacy-preserving federated learning (no raw data leaves sites)
    - Heterogeneous model swarms (different architectures, shared hub)
    - Consortium training (hospitals, edge devices, research labs)
"""

import datetime
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fingerprint import compute_fingerprints
from .uhs import UniversalHubSpace
from .drift import DriftMeasure
from .privacy import DifferentialPrivacy
from .token import TesseraToken, KnowledgeType
from .utils import setup_logging

logger = setup_logging("tessera.swarm")


class AggregationStrategy(Enum):
    """How to combine contributor hub vectors."""

    MEAN = "mean"  # Simple arithmetic mean
    WEIGHTED_MEAN = "weighted_mean"  # Weight by inverse-drift or custom
    TRIMMED_MEAN = "trimmed_mean"  # Drop outlier tails (Byzantine resilience)
    MEDIAN = "median"  # Per-dimension median (most robust)
    ROBUST_WEIGHTED_MEAN = "robust_weighted_mean"  # Huber-style cosine clipping + weighted mean


class SwarmAggregator:
    """
    Orchestrates knowledge aggregation from multiple contributor tokens
    into a single aggregator (large) model.

    The aggregator does NOT need access to contributor models — only their
    tokens (hub vectors). Contributors are distributed and only send compact
    TesseraTokens.

    Usage:
        aggregator = SwarmAggregator(
            aggregator_model=large_model,
            aggregator_id="central-v1",
        )

        # One-way: update aggregator from contributors
        token = aggregator.aggregate(
            contributor_tokens=[token1, token2, token3],
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        # Round-trip: also send updated knowledge back to contributors
        agg_token, broadcast_tokens = aggregator.aggregate_and_broadcast(
            contributor_tokens=[token1, token2, token3],
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
    """

    def __init__(
        self,
        aggregator_model: nn.Module,
        aggregator_id: str = "aggregator",
        device: str = "cpu",
        hub_dim: Optional[int] = None,
        **kwargs: object,
    ):
        self.aggregator = aggregator_model
        self.aggregator_id = aggregator_id
        self.device = device
        self._hub_dim = hub_dim  # optional, for protocol compatibility

        self.agg_uhs = None  # trained during aggregate()

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def aggregate_hub(
        self,
        tokens: List[TesseraToken],
        method: str = "robust_weighted_mean",
    ) -> np.ndarray:
        """
        Protocol-only: combine contributor tokens into one hub vector (no model).
        Delegates to module-level aggregate_tokens for CLI and tests.
        """
        return aggregate_tokens(tokens, method=method)

    def aggregate(
        self,
        contributor_tokens: List[TesseraToken],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN,
        weights: Optional[List[float]] = None,
        trimmed_fraction: float = 0.1,
        uhs_epochs: int = 10,
        finetune_epochs: int = 5,
        finetune_lr: float = 1e-3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ) -> TesseraToken:
        """
        Aggregate many contributor tokens into the aggregator model.

        Args:
            contributor_tokens: List of N TesseraTokens from small models.
            train_dataloader: Data for UHS training and fine-tuning.
            val_dataloader: Data for drift measurement.
            aggregation_strategy: How to combine hub vectors.
            weights: Custom weights for WEIGHTED_MEAN (auto if None).
            trimmed_fraction: Fraction to trim from each tail (TRIMMED_MEAN).
            uhs_epochs: Epochs for aggregator UHS training.
            finetune_epochs: Epochs for aggregator fine-tuning.
            finetune_lr: Learning rate for fine-tuning.
            privacy_epsilon: DP epsilon for the output token.
            privacy_delta: DP delta for the output token.

        Returns:
            TesseraToken representing the aggregated collective knowledge.
        """

        n = len(contributor_tokens)

        # ── Step 1: Validate ──────────────────────────────────────────
        logger.info(f"[Step 1/8] Validating {n} contributor tokens...")
        self._validate_contributor_tokens(contributor_tokens)

        # ── Step 2: Extract hub vectors ───────────────────────────────
        logger.info("[Step 2/8] Extracting hub vectors...")
        hub_matrix, contributor_ids = self._extract_hub_vectors(contributor_tokens)
        logger.info(f"  Hub matrix shape: ({hub_matrix.shape[0]}, {hub_matrix.shape[1]})")

        # ── Step 3: Aggregate ─────────────────────────────────────────
        logger.info(
            f"[Step 3/8] Aggregating hub vectors " f"(strategy={aggregation_strategy.value})..."
        )

        if weights is None and aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN:
            weights = self._auto_weights(contributor_tokens)

        aggregated_hub = self._aggregate_hub_vectors(
            hub_matrix, aggregation_strategy, weights, trimmed_fraction
        )
        logger.info(f"  Aggregated hub norm: {np.linalg.norm(aggregated_hub):.4f}")

        # ── Step 4: Fingerprint aggregator ────────────────────────────
        logger.info("[Step 4/8] Fingerprinting aggregator model...")
        agg_fingerprints = compute_fingerprints(
            self.aggregator, train_dataloader, None, self.device
        )
        agg_layer_names = sorted(agg_fingerprints.keys())
        agg_d = agg_fingerprints[agg_layer_names[0]].d_layer
        logger.info(f"  Aggregator: {len(agg_layer_names)} layers, d_model={agg_d}")

        # ── Step 5: Train aggregator UHS ──────────────────────────────
        logger.info("[Step 5/8] Training aggregator UHS encoder/decoder...")
        agg_acts = self._collect_activations(self.aggregator, train_dataloader, agg_layer_names)
        agg_pooled = self._pool_activations(agg_acts, agg_layer_names)

        self.agg_uhs = UniversalHubSpace(agg_d, device=self.device)
        agg_loader = DataLoader(
            TensorDataset(torch.tensor(agg_pooled, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        )
        self.agg_uhs.train(agg_loader, epochs=uhs_epochs, verbose=True)

        rt_err = self.agg_uhs.round_trip_error(torch.tensor(agg_pooled[:100], dtype=torch.float32))
        logger.info(f"  Aggregator UHS round-trip error: {rt_err:.4f}")

        # ── Step 6: Decode aggregated hub → aggregator space ──────────
        logger.info("[Step 6/8] Decoding aggregated hub into aggregator space...")
        agg_hub_tensor = (
            torch.tensor(aggregated_hub, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Expand to match training data size for fine-tuning targets
        n_samples = agg_pooled.shape[0]
        expanded_hub = agg_hub_tensor.expand(n_samples, -1)
        decoded_targets = self.agg_uhs.decode(expanded_hub).detach()
        logger.info(f"  Decoded targets shape: {decoded_targets.shape}")

        # ── Step 7: Fine-tune aggregator ──────────────────────────────
        logger.info("[Step 7/8] Fine-tuning aggregator on decoded targets...")
        self._finetune_aggregator(
            self.aggregator,
            train_dataloader,
            decoded_targets,
            agg_layer_names,
            finetune_epochs,
            finetune_lr,
        )

        # ── Step 8: Measure drift, package token ──────────────────────
        logger.info("[Step 8/8] Measuring drift and creating token...")

        drift = DriftMeasure(self.aggregator, self.aggregator, self.device).compute(val_dataloader)

        # Compose privacy from contributors
        composed_eps, composed_delta = self._compose_privacy_budgets(contributor_tokens)

        dp = DifferentialPrivacy(privacy_epsilon, privacy_delta)
        private_hub = dp.add_noise(aggregated_hub)

        # Per-contributor drift scores
        contributor_drifts = [t.drift_score for t in contributor_tokens]
        contributor_weights = weights if weights is not None else [1.0 / n] * n

        token = TesseraToken(
            knowledge_type=KnowledgeType.SWARM,
            uhs_vector=private_hub.tolist(),
            modality_weights={"A": 0.70, "W": 0.0, "B": 0.10, "D": 0.10, "S": 0.10},
            correlation_map={},
            lineage_dag={
                "nodes": [
                    {"id": f"c{i}", "type": "contributor", "ref": cid}
                    for i, cid in enumerate(contributor_ids)
                ]
                + [{"id": "agg", "type": "aggregator", "ref": self.aggregator_id}],
                "root": "agg",
            },
            generation=1,
            projection_hints=[],
            privacy_epsilon=privacy_epsilon,
            privacy_delta=privacy_delta,
            drift_score=drift,
            source_model_id=",".join(contributor_ids),
            target_model_id=self.aggregator_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            version="1.0",
            custom_metadata={
                "swarm_mode": "aggregate",
                "aggregation_method": aggregation_strategy.value,
                "contributor_count": n,
                "contributor_ids": contributor_ids,
                "contributor_drift_scores": contributor_drifts,
                "contributor_weights": contributor_weights,
                "composed_epsilon": composed_eps,
                "composed_delta": composed_delta,
                "aggregator_d_model": agg_d,
            },
        )

        logger.info("  Aggregation complete!")
        logger.info(f"  Contributors:  {n}")
        logger.info(f"  Strategy:      {aggregation_strategy.value}")
        logger.info(f"  Drift:         {drift:.6f}")
        logger.info(f"  Privacy:       ε={privacy_epsilon}, δ={privacy_delta}")
        logger.info(f"  Composed:      ε={composed_eps:.4f}, δ={composed_delta:.2e}")

        return token

    def aggregate_and_broadcast(
        self,
        contributor_tokens: List[TesseraToken],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN,
        weights: Optional[List[float]] = None,
        trimmed_fraction: float = 0.1,
        uhs_epochs: int = 10,
        finetune_epochs: int = 5,
        finetune_lr: float = 1e-3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ) -> Tuple[TesseraToken, List[TesseraToken]]:
        """
        Aggregate contributor tokens into aggregator, then broadcast
        the aggregator's updated state back to each contributor.

        Returns:
            (aggregated_token, [broadcast_token_1, ..., broadcast_token_N])
        """

        # Phase 1: Aggregate inbound
        agg_token = self.aggregate(
            contributor_tokens=contributor_tokens,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            aggregation_strategy=aggregation_strategy,
            weights=weights,
            trimmed_fraction=trimmed_fraction,
            uhs_epochs=uhs_epochs,
            finetune_epochs=finetune_epochs,
            finetune_lr=finetune_lr,
            privacy_epsilon=privacy_epsilon,
            privacy_delta=privacy_delta,
        )

        # Phase 2: Encode aggregator's updated state back to hub
        logger.info("[Broadcast] Encoding aggregator's updated state to hub...")

        agg_fingerprints = compute_fingerprints(
            self.aggregator, train_dataloader, None, self.device
        )
        agg_layer_names = sorted(agg_fingerprints.keys())

        updated_acts = self._collect_activations(self.aggregator, train_dataloader, agg_layer_names)
        updated_pooled = self._pool_activations(updated_acts, agg_layer_names)

        # Encode through the aggregator's UHS
        updated_tensor = torch.tensor(updated_pooled, dtype=torch.float32).to(self.device)
        updated_hub = self.agg_uhs.encode(updated_tensor)
        mean_hub = updated_hub.mean(dim=0).cpu().numpy()

        # Apply DP noise to broadcast
        dp = DifferentialPrivacy(privacy_epsilon, privacy_delta)
        broadcast_hub = dp.add_noise(mean_hub)

        # Phase 3: Create one token per contributor
        logger.info(f"[Broadcast] Creating {len(contributor_tokens)} broadcast tokens...")

        broadcast_tokens = []
        for ct in contributor_tokens:
            cid = ct.source_model_id

            bt = TesseraToken(
                knowledge_type=KnowledgeType.SWARM,
                uhs_vector=broadcast_hub.tolist(),
                modality_weights={"A": 0.70, "W": 0.0, "B": 0.10, "D": 0.10, "S": 0.10},
                correlation_map={},
                lineage_dag={
                    "nodes": [
                        {"id": "agg", "type": "aggregator", "ref": self.aggregator_id},
                        {"id": "tgt", "type": "contributor", "ref": cid},
                    ],
                    "root": "agg",
                },
                generation=agg_token.generation + 1,
                projection_hints=[],
                privacy_epsilon=privacy_epsilon,
                privacy_delta=privacy_delta,
                drift_score=agg_token.drift_score,
                source_model_id=self.aggregator_id,
                target_model_id=cid,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                version="1.0",
                custom_metadata={
                    "swarm_mode": "broadcast",
                    "aggregator_id": self.aggregator_id,
                    "transfer_direction": f"aggregator_to_{cid}",
                    "swap_partner": cid,
                    "from_aggregation": True,
                    "contributor_count": len(contributor_tokens),
                },
            )
            broadcast_tokens.append(bt)

        logger.info(f"  Broadcast complete: {len(broadcast_tokens)} tokens created")
        return agg_token, broadcast_tokens

    # ══════════════════════════════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════════════════════════════

    def _validate_contributor_tokens(self, tokens: List[TesseraToken]) -> None:
        """Ensure all contributor tokens are valid for aggregation."""
        if len(tokens) < 2:
            raise ValueError(
                f"Swarm aggregation requires at least 2 contributor tokens, " f"got {len(tokens)}."
            )

        for i, t in enumerate(tokens):
            if len(t.uhs_vector) != 2048:
                raise ValueError(
                    f"Contributor token {i} ({t.source_model_id}) has "
                    f"hub vector of dimension {len(t.uhs_vector)}, expected 2048."
                )

    # ══════════════════════════════════════════════════════════════════════
    #  Hub vector extraction and aggregation
    # ══════════════════════════════════════════════════════════════════════

    def _extract_hub_vectors(self, tokens: List[TesseraToken]) -> Tuple[np.ndarray, List[str]]:
        """Extract N × 2048 hub matrix and contributor IDs."""
        hub_vectors = np.array([t.uhs_vector for t in tokens], dtype=np.float32)
        contributor_ids = [t.source_model_id for t in tokens]
        return hub_vectors, contributor_ids

    def _aggregate_hub_vectors(
        self,
        hub_matrix: np.ndarray,
        strategy: AggregationStrategy,
        weights: Optional[List[float]],
        trimmed_fraction: float,
    ) -> np.ndarray:
        """Combine N hub vectors into one using the specified strategy."""

        if strategy == AggregationStrategy.MEAN:
            agg = self._mean_aggregation(hub_matrix)
        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            if weights is None:
                weights = [1.0 / len(hub_matrix)] * len(hub_matrix)
            agg = self._weighted_mean_aggregation(hub_matrix, weights)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            agg = self._trimmed_mean_aggregation(hub_matrix, trimmed_fraction)
        elif strategy == AggregationStrategy.MEDIAN:
            agg = self._median_aggregation(hub_matrix)
        elif strategy == AggregationStrategy.ROBUST_WEIGHTED_MEAN:
            if weights is None:
                weights = [1.0 / len(hub_matrix)] * len(hub_matrix)
            agg = self._robust_weighted_mean_aggregation(hub_matrix, weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        # L2-renormalise to unit sphere (hub vectors are normalised)
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm

        return agg

    def _mean_aggregation(self, hub_matrix: np.ndarray) -> np.ndarray:
        """Simple arithmetic mean across contributors."""
        return hub_matrix.mean(axis=0)

    def _weighted_mean_aggregation(
        self, hub_matrix: np.ndarray, weights: List[float]
    ) -> np.ndarray:
        """Weighted mean: Σ wᵢ hᵢ / Σ wᵢ."""
        w = np.array(weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        return (hub_matrix * w[:, np.newaxis]).sum(axis=0)

    def _trimmed_mean_aggregation(self, hub_matrix: np.ndarray, fraction: float) -> np.ndarray:
        """
        Per-dimension trimmed mean: sort each dimension, drop top/bottom
        fraction, average the rest. Provides Byzantine resilience.
        """
        n = hub_matrix.shape[0]
        trim_count = max(1, int(n * fraction))

        if n - 2 * trim_count < 1:
            # Not enough samples to trim; fall back to median
            return self._median_aggregation(hub_matrix)

        sorted_matrix = np.sort(hub_matrix, axis=0)
        trimmed = sorted_matrix[trim_count : n - trim_count]
        return trimmed.mean(axis=0)

    def _median_aggregation(self, hub_matrix: np.ndarray) -> np.ndarray:
        """Per-dimension median (most robust aggregation)."""
        return np.median(hub_matrix, axis=0)

    def _robust_weighted_mean_aggregation(
        self,
        hub_matrix: np.ndarray,
        weights: List[float],
        clip_percentile: float = 90.0,
    ) -> np.ndarray:
        """
        Huber-style robust weighted mean with cosine-distance clipping.

        1. Compute per-dimension median as the robust centre.
        2. Measure cosine distance from each contributor to the median.
        3. Clip contributors whose distance exceeds the clip_percentile
           threshold (i.e. zero their weight).
        4. Weighted mean of surviving contributors.

        Falls back to plain median if all contributors are clipped.
        """
        # Robust centre: per-dimension median
        median_hub = np.median(hub_matrix, axis=0)

        # Cosine distances to median
        median_norm = np.linalg.norm(median_hub)
        if median_norm < 1e-10:
            # Degenerate — fall back to weighted mean
            return self._weighted_mean_aggregation(hub_matrix, weights)

        distances = np.zeros(hub_matrix.shape[0])
        for i in range(hub_matrix.shape[0]):
            row_norm = np.linalg.norm(hub_matrix[i])
            if row_norm < 1e-10:
                distances[i] = 1.0  # maximally distant
            else:
                cos_sim = float(np.dot(hub_matrix[i], median_hub) / (row_norm * median_norm))
                cos_sim = max(-1.0, min(1.0, cos_sim))
                distances[i] = (1.0 - cos_sim) / 2.0  # [0, 1]

        # Clip threshold at the given percentile
        threshold = np.percentile(distances, clip_percentile)

        # Zero weights for outliers
        w = np.array(weights, dtype=np.float64)
        for i in range(len(w)):
            if distances[i] > threshold:
                w[i] = 0.0

        if w.sum() < 1e-10:
            # All clipped — fall back to median
            return median_hub

        w = w / w.sum()
        return (hub_matrix * w[:, np.newaxis]).sum(axis=0)

    def _auto_weights(self, tokens: List[TesseraToken]) -> List[float]:
        """
        Auto-compute weights for WEIGHTED_MEAN.

        Strategy: inverse drift weighting — lower drift contributors get
        higher weight (they had better knowledge transfer quality).
        """
        drifts = np.array([t.drift_score for t in tokens], dtype=np.float32)

        # Avoid division by zero: add small epsilon
        inv_drifts = 1.0 / (drifts + 1e-8)
        weights = inv_drifts / inv_drifts.sum()

        return weights.tolist()

    # ══════════════════════════════════════════════════════════════════════
    #  Activation collection and fine-tuning (reused patterns)
    # ══════════════════════════════════════════════════════════════════════

    def _collect_activations(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
    ) -> Dict[str, list]:
        """
        Collect per-layer activations via forward hooks.
        (Same pattern as transfer.py)
        """
        model.eval()
        model.to(self.device)
        activations: Dict[str, list] = {name: [] for name in layer_names}

        hooks = []
        for name, module in model.named_modules():
            if name in layer_names:

                def make_hook(layer_name):
                    def hook_fn(mod, inp, output):
                        out = output[0] if isinstance(output, tuple) else output
                        if out.ndim == 3:
                            out = out.mean(dim=1)
                        activations[layer_name].append(out.detach().cpu())

                    return hook_fn

                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                model(x)

        for h in hooks:
            h.remove()
        return activations

    def _pool_activations(self, acts: Dict[str, list], layer_names: List[str]) -> np.ndarray:
        """Pool activations using middle layer (most informative)."""
        mid_idx = len(layer_names) // 2
        mid_layer = layer_names[mid_idx]

        if mid_layer in acts and len(acts[mid_layer]) > 0:
            pooled = torch.cat(acts[mid_layer], dim=0).numpy()
        else:
            for name in layer_names:
                if name in acts and len(acts[name]) > 0:
                    pooled = torch.cat(acts[name], dim=0).numpy()
                    break
            else:
                raise RuntimeError("No activations collected from any layer")

        return pooled

    def _finetune_aggregator(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        targets: torch.Tensor,
        layer_names: List[str],
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        """
        Fine-tune aggregator to align its activations with decoded targets.
        (Same hook + MSE pattern as transfer.py:_finetune_receiver)
        """
        model.train()
        model.to(self.device)
        targets = targets.to(self.device)

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        mid_layer = layer_names[len(layer_names) // 2]

        for epoch in range(epochs):
            epoch_loss = 0.0
            sample_idx = 0

            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_size = x.shape[0]

                if sample_idx + batch_size > len(targets):
                    break

                batch_targets = targets[sample_idx : sample_idx + batch_size]
                sample_idx += batch_size

                captured = {}

                def capture_hook(module, inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if out.ndim == 3:
                        out = out.mean(dim=1)
                    captured["act"] = out

                hook = None
                for name, module in model.named_modules():
                    if name == mid_layer:
                        hook = module.register_forward_hook(capture_hook)
                        break

                x = x.to(self.device)
                model(x)

                if hook is not None:
                    hook.remove()

                if "act" not in captured:
                    continue

                act = captured["act"]
                tgt = batch_targets

                if act.shape[-1] != tgt.shape[-1]:
                    tgt = F.adaptive_avg_pool1d(tgt.unsqueeze(1), act.shape[-1]).squeeze(1)

                loss = F.mse_loss(act, tgt)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            logger.info(
                f"  Fine-tune epoch {epoch + 1}/{epochs}: " f"alignment_loss={epoch_loss:.4f}"
            )

        model.eval()

    # ══════════════════════════════════════════════════════════════════════
    #  Privacy composition
    # ══════════════════════════════════════════════════════════════════════

    def _compose_privacy_budgets(self, tokens: List[TesseraToken]) -> Tuple[float, float]:
        """
        Compose privacy budgets from N contributors.

        Uses advanced composition theorem for Gaussian mechanism:
            ε_total = √N × max(εᵢ)
            δ_total = Σ δᵢ
        """
        n = len(tokens)
        epsilons = [t.privacy_epsilon for t in tokens]
        deltas = [t.privacy_delta for t in tokens]

        composed_epsilon = math.sqrt(n) * max(epsilons)
        composed_delta = sum(deltas)

        return composed_epsilon, composed_delta


# ══════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ══════════════════════════════════════════════════════════════════════════


def swarm_metadata(
    swarm_round_id: Optional[str] = None,
    contributor_id: Optional[str] = None,
    local_data_fingerprint: Optional[str] = None,
    extra_signals: Optional[Dict[str, object]] = None,
    *,
    round_id: Optional[str] = None,
    quality_signals: Optional[Dict[str, object]] = None,
) -> dict:
    """
    Build the standard custom_metadata dict for a swarm contribution token.

    Every token participating in a swarm round **must** carry these three
    fields in its custom_metadata.  This helper ensures the right keys and
    adds any optional extra signals.

    Accepts positional (round_id, contributor_id, fingerprint, signals_dict)
    or keyword round_id / contributor_id / local_data_fingerprint / quality_signals.
    """
    rid = swarm_round_id if swarm_round_id is not None else round_id
    cid = contributor_id or ""
    fp = local_data_fingerprint or ""
    meta: dict = {
        "swarm_round_id": rid or "",
        "contributor_id": cid,
        "local_data_fingerprint": fp,
    }
    # quality_signals → nested under "quality_signals" key (structured per-metric dict)
    if quality_signals:
        meta["quality_signals"] = dict(quality_signals)
    # extra_signals → merged flat into top-level dict (arbitrary key-value pairs)
    if extra_signals:
        meta.update(extra_signals)
    return meta


def validate_for_swarm(
    token: TesseraToken,
    require_signature: bool = False,
) -> Tuple[bool, str]:
    """
    Quick-check whether a token has the required swarm fields.

    Validates:
        1. custom_metadata contains non-empty 'swarm_round_id'
        2. custom_metadata contains non-empty 'contributor_id'
        3. custom_metadata contains non-empty 'local_data_fingerprint'
        4. uhs_vector is non-empty
        5. (optional) Ed25519 signature is present and valid

    Args:
        token: The token to validate.
        require_signature: If True, token must carry a valid Ed25519 signature.
            Tokens without a signature are rejected. Default False for
            backwards compatibility with unsigned tokens.

    Returns:
        (True, "ok") if valid, (False, reason) if not.
    """
    meta = token.custom_metadata or {}

    if not meta.get("swarm_round_id"):
        return False, "Missing or empty 'swarm_round_id' in custom_metadata."

    if not meta.get("contributor_id"):
        return False, "Missing or empty 'contributor_id' in custom_metadata."

    if not meta.get("local_data_fingerprint"):
        return False, "Missing or empty 'local_data_fingerprint' in custom_metadata."

    if not token.uhs_vector or len(token.uhs_vector) == 0:
        return False, "UHS vector is empty."

    if require_signature:
        from .signing import verify_token_signature

        ok, reason = verify_token_signature(token)
        if not ok:
            return False, f"Signature validation failed: {reason}"

    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════
#  Protocol layer: CLI and token-only helpers (no model training)
# ══════════════════════════════════════════════════════════════════════════

# Metadata keys for swarm tokens (used by CLI and tests)
SWARM_ROUND_ID = "swarm_round_id"
CONTRIBUTOR_ID = "contributor_id"
QUALITY_SIGNALS = "quality_signals"
AGGREGATION_WEIGHT = "aggregation_weight"
UTILITY_SCORE = "utility_score"
BROADCAST_VERSION = "broadcast_version"
LINEAGE_PARENT_ROUNDS = "lineage_parent_rounds"


def score_token(
    token: TesseraToken,
    round_context: Optional[Dict[str, object]] = None,
) -> float:
    """
    Compute utility score for a token in a round context.
    Delegates to credits.compute_utility with quality, novelty, freshness, reliability.
    """
    from . import credits

    ctx = round_context or {}
    meta = token.custom_metadata or {}
    signals = meta.get(QUALITY_SIGNALS) or {}
    drift = float(signals.get("drift", 0.5))
    recon_error = float(signals.get("recon_error", 0.0))
    quality = credits.compute_quality_score(drift, recon_error)
    prior = ctx.get("prior_centroid")
    hub = np.array(token.uhs_vector, dtype=np.float64) if token.uhs_vector else None
    novelty = credits.compute_novelty_score(hub, prior) if hub is not None else 0.5
    freshness = credits.compute_freshness_score(token.timestamp or "")
    ledger = ctx.get("ledger") or ctx.get("contributor_history") or []
    cid = meta.get(CONTRIBUTOR_ID) or token.source_model_id or ""
    reliability = credits.compute_reliability_score(cid, ledger)
    return credits.compute_utility(quality, novelty, freshness, reliability)


def aggregate_tokens(
    tokens: List[TesseraToken],
    method: str = "robust_weighted_mean",
) -> np.ndarray:
    """
    Aggregate accepted tokens' UHS vectors into one hub vector.
    Protocol-only: no model. Uses same strategies as SwarmAggregator.
    method: "mean", "weighted_mean" or "weighted", "median", "robust_weighted_mean".
    Accepts AggregationStrategy enum (e.g. AggregationStrategy.MEAN).
    """
    if not tokens:
        raise ValueError("aggregate_tokens requires at least one token")
    # Allow enum
    if hasattr(method, "value"):
        method = method.value
    method = str(method).lower()
    if method == "weighted":
        method = "weighted_mean"
    hub_matrix = np.array([t.uhs_vector for t in tokens], dtype=np.float32)
    weights = [(t.custom_metadata or {}).get(AGGREGATION_WEIGHT, 1.0) for t in tokens]
    if method == "mean":
        agg = hub_matrix.mean(axis=0)
    elif method == "weighted_mean":
        w = np.array(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones(len(w)) / len(w)
        else:
            w = w / w.sum()
        agg = (hub_matrix * w[:, np.newaxis]).sum(axis=0)
    elif method == "median":
        agg = np.median(hub_matrix, axis=0)
    elif method in ("robust_weighted_mean", "trimmed_mean"):
        # Robust: use weighted mean with cosine-distance clipping
        w = np.array(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones(len(w)) / len(w)
        else:
            w = w / w.sum()
        median_hub = np.median(hub_matrix, axis=0)
        mn = np.linalg.norm(median_hub)
        if mn < 1e-10:
            agg = (hub_matrix * w[:, np.newaxis]).sum(axis=0)
        else:
            dists = np.zeros(hub_matrix.shape[0])
            for i in range(hub_matrix.shape[0]):
                rn = np.linalg.norm(hub_matrix[i])
                if rn < 1e-10:
                    dists[i] = 1.0
                else:
                    cs = np.dot(hub_matrix[i], median_hub) / (rn * mn)
                    dists[i] = (1.0 - max(-1.0, min(1.0, cs))) / 2.0
            thresh = np.percentile(dists, 90.0)
            for i in range(len(w)):
                if dists[i] > thresh:
                    w[i] = 0.0
            if w.sum() < 1e-10:
                agg = median_hub.copy()
            else:
                w = w / w.sum()
                agg = (hub_matrix * w[:, np.newaxis]).sum(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    # L2-renorm for mean/weighted/median; leave robust_weighted_mean un-normalised for compatibility
    if method in ("mean", "weighted_mean", "median"):
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm
    return agg.astype(np.float32)


def compute_credits(
    contributor_id: str,
    utility_scores: List[float],
    caps: Optional[Dict[str, float]] = None,
) -> float:
    """Sum of utility scores for the round, optionally capped."""
    caps = caps or {}
    total = sum(utility_scores)
    max_per = caps.get("max_credits_per_day")
    if max_per is not None and total > max_per:
        return float(max_per)
    return total


def submit(token_path: str, contributor_id: str) -> Tuple[bool, str]:
    """Validate and submit a contributor token. Returns (success, message)."""
    from pathlib import Path
    from .binary import TBFSerializer

    path = Path(token_path)
    if not path.exists():
        return False, f"Token file not found: {token_path}"
    try:
        token = TBFSerializer.load(path)
    except Exception as e:
        return False, f"Failed to load token: {e}"
    meta = token.custom_metadata or {}
    if CONTRIBUTOR_ID not in meta:
        token.custom_metadata = {**meta, CONTRIBUTOR_ID: contributor_id}
    ok, reason = validate_for_swarm(token)
    if not ok:
        return False, f"Policy rejected: {reason}"
    return True, "Token accepted for submission (ingress is out-of-band in v1)"


def aggregate(
    round_id: str,
    token_paths: List[str],
) -> Optional[np.ndarray]:
    """Load tokens for round, aggregate to one hub vector. Returns None if below min contributors."""
    from pathlib import Path
    from .binary import TBFSerializer
    from . import policy

    tokens = []
    for p in token_paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            t = TBFSerializer.load(path)
            if (t.custom_metadata or {}).get(SWARM_ROUND_ID) == round_id:
                tokens.append(t)
        except Exception:
            continue
    if len(tokens) < policy.MIN_ACCEPTED_CONTRIBUTORS:
        return None
    return aggregate_tokens(tokens, method="robust_weighted_mean")


def broadcast(
    round_id: str,
    hub_vector: np.ndarray,
    broadcast_version: str,
) -> TesseraToken:
    """Build a broadcast token (central → contributors) for the round."""
    return TesseraToken(
        knowledge_type=KnowledgeType.SWARM,
        uhs_vector=hub_vector.tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={},
        source_model_id="swarm_central",
        target_model_id=None,
        version="1.0",
        custom_metadata={
            BROADCAST_VERSION: broadcast_version,
            SWARM_ROUND_ID: round_id,
            LINEAGE_PARENT_ROUNDS: [round_id],
        },
    )


def score(round_id: str, token_paths: List[str]) -> Dict[str, float]:
    """Score each token for a round; return contributor_id -> utility_score."""
    from pathlib import Path
    from .binary import TBFSerializer

    ctx = {"round_id": round_id}
    out = {}
    for p in token_paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            t = TBFSerializer.load(path)
            if (t.custom_metadata or {}).get(SWARM_ROUND_ID) != round_id:
                continue
            cid = (t.custom_metadata or {}).get(CONTRIBUTOR_ID, path.stem)
            u = score_token(t, ctx)
            meta = t.custom_metadata or {}
            meta[UTILITY_SCORE] = u
            meta[AGGREGATION_WEIGHT] = max(0.0, u)
            t.custom_metadata = meta
            out[cid] = u
        except Exception:
            continue
    return out
