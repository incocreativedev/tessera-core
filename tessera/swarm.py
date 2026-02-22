"""
tessera.swarm — Round-trip swarm orchestration.

Two layers:
  - Protocol layer: submit(), aggregate_tokens(), broadcast(), score_token(),
    credits, policy gating (no model training). Use for coordination and CLI.
  - ML engine: SwarmAggregator — fingerprints aggregator model, trains UHS,
    aggregates tokens, fine-tunes central model, re-encodes broadcast token.

See docs/swarm-roundtrip-architecture.md for the full spec.
v1: Pilot domain Ag/Mining; central operator; quality-weighted credits.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union

import numpy as np

from .token import TesseraToken, KnowledgeType
from . import policy
from . import credits


class AggregationStrategy(Enum):
    """Strategy for combining contributor hub vectors into one."""

    MEAN = "mean"  # Simple mean, then L2 renorm
    WEIGHTED = "weighted"  # Utility-weighted mean, L2 renorm
    TRIMMED = "trimmed"  # Drop farthest by cosine, then mean, L2 renorm
    MEDIAN = "median"  # Coordinate-wise median, L2 renorm
    ROBUST_WEIGHTED_MEAN = "robust_weighted_mean"  # v1 default: Huber-style clip by cosine to median


# Swarm custom_metadata keys (stored in token.custom_metadata)
SWARM_ROUND_ID = "swarm_round_id"
CONTRIBUTOR_ID = "contributor_id"
LOCAL_DATA_FINGERPRINT = "local_data_fingerprint"
QUALITY_SIGNALS = "quality_signals"
AGGREGATION_WEIGHT = "aggregation_weight"
UTILITY_SCORE = "utility_score"
BROADCAST_VERSION = "broadcast_version"
LINEAGE_PARENT_ROUNDS = "lineage_parent_rounds"


def swarm_metadata(
    round_id: str,
    contributor_id: str,
    local_data_fingerprint: str = "",
    quality_signals: Optional[Dict[str, float]] = None,
) -> dict:
    """Build custom_metadata dict for a contributor token."""
    return {
        SWARM_ROUND_ID: round_id,
        CONTRIBUTOR_ID: contributor_id,
        LOCAL_DATA_FINGERPRINT: local_data_fingerprint or "",
        QUALITY_SIGNALS: quality_signals or {},
        AGGREGATION_WEIGHT: 0.0,  # set by score_token
        UTILITY_SCORE: 0.0,
        BROADCAST_VERSION: "",
        LINEAGE_PARENT_ROUNDS: [],
    }


def validate_for_swarm(token: TesseraToken) -> tuple[bool, str]:
    """
    Run policy checks on a token for swarm acceptance.
    Returns (accepted, reason).
    """
    return policy.accept_token(token)


def score_token(token: TesseraToken, round_context: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute utility score for a token in a round context.
    round_context can include: prior_centroid, contributor_history, round_ts.
    """
    return credits.utility_score(token, round_context or {})


def aggregate_tokens(
    tokens: List[TesseraToken],
    method: Union[str, AggregationStrategy] = "robust_weighted_mean",
) -> np.ndarray:
    """
    Aggregate accepted tokens' UHS vectors into one hub vector.
    method: "mean", "weighted", "trimmed", "median", "robust_weighted_mean" (v1 default),
    or AggregationStrategy enum. All strategies L2-renormalise the result.
    """
    if not tokens:
        raise ValueError("aggregate_tokens requires at least one token")
    vectors = np.array([t.uhs_vector for t in tokens], dtype=np.float64)
    weights = np.array(
        [t.custom_metadata.get(AGGREGATION_WEIGHT, 1.0) for t in tokens],
        dtype=np.float64,
    )
    try:
        strategy = method if isinstance(method, AggregationStrategy) else AggregationStrategy(method)
    except ValueError:
        raise ValueError(f"Unknown aggregation method: {method}") from None
    if strategy == AggregationStrategy.ROBUST_WEIGHTED_MEAN:
        return _robust_weighted_mean(vectors, weights)
    if strategy == AggregationStrategy.MEAN:
        out = _aggregate_mean(vectors)
    elif strategy == AggregationStrategy.WEIGHTED:
        out = _aggregate_weighted(vectors, weights)
    elif strategy == AggregationStrategy.TRIMMED:
        out = _aggregate_trimmed(vectors, weights)
    elif strategy == AggregationStrategy.MEDIAN:
        out = _aggregate_median(vectors)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return _l2_renorm(out)


def _l2_renorm(vec: np.ndarray) -> np.ndarray:
    """L2-normalise to unit vector."""
    n = np.linalg.norm(vec)
    if n <= 0:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


def _aggregate_mean(vectors: np.ndarray) -> np.ndarray:
    return np.mean(vectors, axis=0)


def _aggregate_weighted(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()
    return np.average(vectors, axis=0, weights=weights)


def _aggregate_trimmed(vectors: np.ndarray, weights: np.ndarray, trim_frac: float = 0.2) -> np.ndarray:
    """Drop fraction of vectors farthest from weighted median (by cosine), then mean."""
    if len(vectors) <= 2 or trim_frac <= 0:
        return _aggregate_weighted(vectors, weights)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit = vectors / norms
    median_vec = np.average(unit, axis=0, weights=np.maximum(weights, 0.0))
    median_vec = median_vec / (np.linalg.norm(median_vec) or 1.0)
    cos_sim = unit @ median_vec
    keep = int(max(1, (1 - trim_frac) * len(vectors)))
    idx = np.argsort(-cos_sim)[:keep]
    return np.mean(vectors[idx], axis=0)


def _aggregate_median(vectors: np.ndarray) -> np.ndarray:
    """Coordinate-wise median."""
    return np.median(vectors, axis=0)


def _robust_weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean with clipping of outliers by cosine distance to median."""
    # Normalize weights
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()
    # L2-normalize vectors for cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit = vectors / norms
    # Median vector (weighted median approximated by centroid of weighted sample)
    median_vec = np.average(unit, axis=0, weights=weights)
    median_vec = median_vec / (np.linalg.norm(median_vec) or 1.0)
    # Cosine distance = 1 - cos_sim; clip contributions with high distance
    cos_sim = unit @ median_vec
    # Huber-style: downweight outliers (low cos_sim)
    clip_threshold = 0.5  # cos_sim below this gets reduced weight
    adjusted = np.where(cos_sim >= clip_threshold, weights, weights * (cos_sim + 1e-6))
    adjusted = adjusted / adjusted.sum()
    out = np.average(vectors, axis=0, weights=adjusted)
    return out.astype(np.float32)


def compute_credits(
    contributor_id: str,
    utility_scores: List[float],
    caps: Optional[Dict[str, float]] = None,
) -> float:
    """Compute credits for a contributor from their accepted utility scores in a round."""
    return credits.compute_credits(contributor_id, utility_scores, caps or {})


def submit(token_path: str, contributor_id: str) -> tuple[bool, str]:
    """
    Validate and submit a contributor token to central ingress.
    Returns (success, message). In v1 this is local validation only;
    actual ingress is out-of-band (e.g. API upload).
    """
    from .binary import TBFSerializer
    from pathlib import Path
    path = Path(token_path)
    if not path.exists():
        return False, f"Token file not found: {token_path}"
    try:
        token = TBFSerializer.load(path)
    except Exception as e:
        return False, f"Failed to load token: {e}"
    if CONTRIBUTOR_ID not in token.custom_metadata:
        token.custom_metadata = token.custom_metadata or {}
        token.custom_metadata[CONTRIBUTOR_ID] = contributor_id
    accepted, reason = validate_for_swarm(token)
    if not accepted:
        return False, f"Policy rejected: {reason}"
    return True, "Token accepted for submission (ingress is out-of-band in v1)"


def aggregate(round_id: str, token_paths: List[str]) -> Optional[np.ndarray]:
    """
    Load tokens for a round (after scoring/acceptance), aggregate to one hub vector.
    Returns aggregated vector or None if below minimum contributors.
    """
    from .binary import TBFSerializer
    from pathlib import Path
    tokens = []
    for p in token_paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            t = TBFSerializer.load(path)
            if t.custom_metadata.get(SWARM_ROUND_ID) == round_id:
                tokens.append(t)
        except Exception:
            continue
    if len(tokens) < policy.MIN_ACCEPTED_CONTRIBUTORS:
        return None
    return aggregate_tokens(tokens, method="robust_weighted_mean")


def broadcast(round_id: str, hub_vector: np.ndarray, broadcast_version: str) -> TesseraToken:
    """
    Build a broadcast token (large model → contributors) for a round.
    Caller must attach the hub_vector from aggregate and set version.
    """
    from .token import KnowledgeType
    token = TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
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
    return token


def score(round_id: str, token_paths: List[str]) -> Dict[str, float]:
    """
    Score each token for a round; return contributor_id -> utility_score.
    """
    from .binary import TBFSerializer
    from pathlib import Path
    round_context = {"round_id": round_id}
    out = {}
    for p in token_paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            t = TBFSerializer.load(path)
            if t.custom_metadata.get(SWARM_ROUND_ID) != round_id:
                continue
            cid = t.custom_metadata.get(CONTRIBUTOR_ID, path.stem)
            u = score_token(t, round_context)
            t.custom_metadata[UTILITY_SCORE] = u
            t.custom_metadata[AGGREGATION_WEIGHT] = max(0.0, u)
            out[cid] = u
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# ML engine: SwarmAggregator — model training pipeline (optional, requires torch)
# -----------------------------------------------------------------------------


class SwarmAggregator:
    """
    Central (large) model aggregator: aggregates contributor tokens, fine-tunes
    the aggregator model on decoded hub targets, and produces a broadcast token.

    Use with the protocol layer: policy.accept_token / score_token gate tokens
    before passing them to aggregate() or aggregate_and_broadcast().
    """

    def __init__(
        self,
        aggregator_model: Any,
        aggregator_id: str = "swarm_central",
        device: str = "cpu",
        hub_dim: int = 2048,
        d_model: Optional[int] = None,
    ):
        """
        Args:
            aggregator_model: nn.Module (e.g. transformer) to be updated from swarm.
            aggregator_id: Identifier for broadcast token source_model_id.
            device: Torch device.
            hub_dim: UHS dimension (must match contributor tokens).
            d_model: Override aggregator activation dim; if None, inferred from fingerprint.
        """
        self.aggregator_model = aggregator_model
        self.aggregator_id = aggregator_id
        self.device = device
        self.hub_dim = hub_dim
        self.d_model_override = d_model
        self._agg_uhs: Optional[Any] = None
        self._agg_layer_names: Optional[List[str]] = None

    def aggregate(
        self,
        tokens: List[TesseraToken],
        strategy: Union[str, AggregationStrategy] = "robust_weighted_mean",
    ) -> np.ndarray:
        """Combine contributor tokens into one hub vector (protocol aggregation)."""
        return aggregate_tokens(tokens, method=strategy)

    def aggregate_and_broadcast(
        self,
        tokens: List[TesseraToken],
        train_dataloader: Any,
        round_id: str,
        broadcast_version: Optional[str] = None,
        uhs_epochs: int = 10,
        finetune_epochs: int = 5,
        finetune_lr: float = 1e-3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        use_privacy_composition: bool = True,
    ) -> TesseraToken:
        """
        Aggregate tokens → decode into aggregator space → fine-tune aggregator
        → re-encode updated activations as broadcast token (KnowledgeType.SWARM).

        Privacy: if use_privacy_composition, effective epsilon = epsilon / sqrt(N)
        for N contributors (composition over contributors).
        """
        import datetime
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from .fingerprint import compute_fingerprints
        from .uhs import UniversalHubSpace
        from .privacy import DifferentialPrivacy

        N = len(tokens)
        if N == 0:
            raise ValueError("aggregate_and_broadcast requires at least one token")

        hub_vec = self.aggregate(tokens, strategy=AggregationStrategy.ROBUST_WEIGHTED_MEAN)
        hub_vec = np.asarray(hub_vec, dtype=np.float64)

        # Resolve aggregator layer names and d_model
        if self._agg_layer_names is None or self._agg_uhs is None:
            fps = compute_fingerprints(
                self.aggregator_model, train_dataloader, None, self.device
            )
            self._agg_layer_names = sorted(fps.keys())
            d_agg = self.d_model_override or fps[self._agg_layer_names[0]].d_layer
            self._agg_uhs = UniversalHubSpace(d_agg, hub_dim=self.hub_dim, device=self.device)
            # Train UHS on aggregator activations
            acts = self._collect_activations(
                self.aggregator_model, train_dataloader, self._agg_layer_names
            )
            pooled = self._pool_activations(acts, self._agg_layer_names)
            act_loader = DataLoader(
                TensorDataset(torch.tensor(pooled, dtype=torch.float32)),
                batch_size=32,
                shuffle=True,
            )
            self._agg_uhs.train(act_loader, epochs=uhs_epochs, verbose=False)

        d_agg = self._agg_uhs.d_model
        mid_layer = self._agg_layer_names[len(self._agg_layer_names) // 2]

        # Decode aggregate hub into aggregator space (single target vector)
        hub_t = torch.tensor(hub_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        target_vec = self._agg_uhs.decode(hub_t).squeeze(0)

        # Fine-tune aggregator so its activations move toward target
        self._finetune_toward_target(
            train_dataloader, target_vec, mid_layer, epochs=finetune_epochs, lr=finetune_lr
        )

        # Re-encode updated aggregator activations → broadcast hub vector
        acts_new = self._collect_activations(
            self.aggregator_model, train_dataloader, self._agg_layer_names
        )
        pooled_new = self._pool_activations(acts_new, self._agg_layer_names)
        pooled_t = torch.tensor(pooled_new, dtype=torch.float32).to(self.device)
        hub_encoded = self._agg_uhs.encode(pooled_t)
        broadcast_hub = hub_encoded.mean(dim=0).cpu().numpy()

        # Privacy: √N composition — spend epsilon/sqrt(N) on broadcast
        eps = privacy_epsilon / math.sqrt(N) if use_privacy_composition else privacy_epsilon
        dp = DifferentialPrivacy(eps, privacy_delta)
        broadcast_hub = dp.add_noise(broadcast_hub)

        version = broadcast_version or f"round-{round_id}"
        token = TesseraToken(
            knowledge_type=KnowledgeType.SWARM,
            uhs_vector=broadcast_hub.tolist(),
            modality_weights={"A": 1.0},
            correlation_map={},
            lineage_dag={},
            source_model_id=self.aggregator_id,
            target_model_id=None,
            version="1.0",
            custom_metadata={
                BROADCAST_VERSION: version,
                SWARM_ROUND_ID: round_id,
                LINEAGE_PARENT_ROUNDS: [round_id],
            },
        )
        return token

    def _collect_activations(
        self,
        model: Any,
        dataloader: Any,
        layer_names: List[str],
    ) -> Dict[str, list]:
        from collections import defaultdict
        import torch
        activations: Dict[str, list] = defaultdict(list)
        hooks = []

        def make_hook(name):
            def fn(_module, _inp, output):
                out = output[0] if isinstance(output, tuple) else output
                out = out.detach().cpu()
                if out.ndim == 3:
                    out = out.mean(dim=1)
                activations[name].append(out)
            return fn

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x.to(self.device))

        for h in hooks:
            h.remove()
        return activations

    def _pool_activations(self, acts: Dict[str, list], layer_names: List[str]) -> np.ndarray:
        import torch
        mid_idx = len(layer_names) // 2
        mid_layer = layer_names[mid_idx]
        if mid_layer in acts and len(acts[mid_layer]) > 0:
            return torch.cat(acts[mid_layer], dim=0).numpy()
        for name in layer_names:
            if name in acts and len(acts[name]) > 0:
                return torch.cat(acts[name], dim=0).numpy()
        raise RuntimeError("No activations collected")

    def _finetune_toward_target(
        self,
        dataloader: Any,
        target_vec: Any,
        mid_layer: str,
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> None:
        import torch
        import torch.nn.functional as F
        self.aggregator_model.train()
        self.aggregator_model.to(self.device)
        target_vec = target_vec.to(self.device)
        optimiser = torch.optim.Adam(self.aggregator_model.parameters(), lr=lr)
        for _ in range(epochs):
            sample_idx = 0
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                batch_size = x.shape[0]
                captured = {}

                def capture_hook(_m, _i, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if out.ndim == 3:
                        out = out.mean(dim=1)
                    captured["act"] = out

                hook = None
                for name, module in self.aggregator_model.named_modules():
                    if name == mid_layer:
                        hook = module.register_forward_hook(capture_hook)
                        break
                self.aggregator_model(x)
                if hook:
                    hook.remove()
                if "act" not in captured:
                    continue
                act = captured["act"]
                tgt = target_vec.unsqueeze(0).expand(act.size(0), -1)
                if act.shape[-1] != tgt.shape[-1]:
                    tgt = F.adaptive_avg_pool1d(
                        tgt.unsqueeze(1), act.shape[-1]
                    ).squeeze(1)
                loss = F.mse_loss(act, tgt)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                sample_idx += batch_size
        self.aggregator_model.eval()
