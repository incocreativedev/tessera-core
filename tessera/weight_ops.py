"""
tessera.weight_ops — Weight extraction, SVD compression, and chunked UHS encoding.

Weight matrices are too large to encode directly through the hub space
(a single Linear layer's weight might be 768 × 3072 = 2.4 M floats).
This module provides:

    1. Weight extraction from named parameters (Linear, Conv2d)
    2. SVD compression to low-rank approximations
    3. Chunking compressed weights into hub_dim-sized vectors
    4. Encoding/decoding those chunks through UHS
    5. Reassembling decoded chunks back into weight matrices
    6. Weight statistics for drift measurement

Key design: The UHS encoder L2-normalises its output, destroying magnitude.
To preserve it we store a per-chunk scale factor alongside the chunk data and
re-apply it after decoding (the same idea as INT8 scale/zero-point in binary.py).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import setup_logging

logger = setup_logging("tessera.weight_ops")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WeightSnapshot:
    """
    Captured weight matrix stored in SVD-compressed form.

    Stores the truncated SVD components (U, S, Vt) plus original shape
    information so the weight can be decoded into a different architecture's
    layer dimensions.
    """

    layer_name: str
    original_shape: Tuple[int, ...]  # e.g. (768, 3072) for a Linear layer
    rank: int  # SVD truncation rank used
    U: np.ndarray  # (rows, rank)
    S: np.ndarray  # (rank,)
    Vt: np.ndarray  # (rank, cols)

    @property
    def compressed_size(self) -> int:
        """Number of floats in the compressed representation."""
        return self.U.size + self.S.size + self.Vt.size

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the approximate weight matrix from SVD components."""
        return (self.U * self.S[None, :]) @ self.Vt  # (rows, cols)


@dataclass
class WeightStats:
    """Per-layer weight distribution statistics for drift measurement."""

    layer_name: str
    mean: float
    std: float
    frobenius_norm: float
    spectral_norm: float  # largest singular value
    effective_rank: float  # (Σσ)² / Σσ²
    shape: Tuple[int, ...]


@dataclass
class ChunkMeta:
    """Metadata for one hub-dim-sized chunk of a compressed weight."""

    layer_name: str
    chunk_idx: int
    total_chunks: int
    scale: float  # pre-norm L2 magnitude; restored after decode
    flat_offset: int  # offset into the flat [U; S; Vt] array
    flat_length: int  # number of floats this chunk covers (≤ hub_dim)
    # Reconstruction info
    original_shape: Tuple[int, ...]
    rank: int
    u_size: int  # U.size
    s_size: int  # S.size
    vt_size: int  # Vt.size


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def _auto_detect_weight_layers(model: nn.Module) -> List[str]:
    """Return names of all Linear and Conv2d modules in traversal order."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name:
            layers.append(name)
    return layers


def extract_weights(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract weight matrices from named parameters.

    Targets the `.weight` attribute of Linear and Conv2d modules.
    Conv2d weights (out, in, kH, kW) are reshaped to (out, in*kH*kW).

    Args:
        model: PyTorch model.
        layer_names: Module names to extract from (auto-detect if None).

    Returns:
        {layer_name: 2D numpy array of shape (out_features, in_features)}.
    """
    target = layer_names or _auto_detect_weight_layers(model)
    modules = dict(model.named_modules())

    weights: Dict[str, np.ndarray] = {}
    for name in target:
        if name not in modules:
            logger.warning(f"Layer '{name}' not found in model; skipping.")
            continue
        module = modules[name]
        if not hasattr(module, "weight") or module.weight is None:
            logger.warning(f"Layer '{name}' has no weight parameter; skipping.")
            continue

        w = module.weight.detach().cpu().float().numpy()
        if w.ndim > 2:
            # Conv2d: (out, in, kH, kW) → (out, in*kH*kW)
            w = w.reshape(w.shape[0], -1)
        elif w.ndim == 1:
            w = w.reshape(1, -1)

        weights[name] = w
        logger.debug(f"  Extracted '{name}': {w.shape}")

    return weights


# ---------------------------------------------------------------------------
# SVD compression
# ---------------------------------------------------------------------------


def svd_compress(
    weight: np.ndarray,
    layer_name: str = "",
    rank: Optional[int] = None,
    energy_threshold: float = 0.95,
) -> WeightSnapshot:
    """
    Compress a 2D weight matrix via truncated SVD.

    If rank is None, automatically selects the minimum rank that preserves
    at least `energy_threshold` fraction of the total spectral energy
    (sum of squared singular values).

    Args:
        weight: 2D float array of shape (rows, cols).
        layer_name: Identifier for logging / snapshot metadata.
        rank: Explicit truncation rank. If None, use energy_threshold.
        energy_threshold: Fraction of spectral energy to retain (0.0–1.0).

    Returns:
        WeightSnapshot containing U, S, Vt and reconstruction metadata.
    """
    assert weight.ndim == 2, f"Expected 2D weight, got shape {weight.shape}"
    rows, cols = weight.shape

    # Full SVD (economy mode: U is rows×min(rows,cols))
    U_full, S_full, Vt_full = np.linalg.svd(weight.astype(np.float64), full_matrices=False)

    if rank is None:
        # Auto-select rank by energy threshold
        energy = S_full**2
        cumulative = np.cumsum(energy) / (energy.sum() + 1e-12)
        rank = int(np.searchsorted(cumulative, energy_threshold)) + 1

    rank = max(1, min(rank, len(S_full)))

    U = U_full[:, :rank].astype(np.float32)
    S = S_full[:rank].astype(np.float32)
    Vt = Vt_full[:rank, :].astype(np.float32)

    logger.debug(
        f"  SVD compress '{layer_name}': {weight.shape} → rank={rank} "
        f"({100 * (S**2).sum() / ((S_full**2).sum() + 1e-12):.1f}% energy)"
    )

    return WeightSnapshot(
        layer_name=layer_name,
        original_shape=weight.shape,
        rank=rank,
        U=U,
        S=S,
        Vt=Vt,
    )


# ---------------------------------------------------------------------------
# Chunking for hub encoding
# ---------------------------------------------------------------------------


def chunk_for_hub(
    snapshot: WeightSnapshot,
    hub_dim: int,
) -> Tuple[List[np.ndarray], List[ChunkMeta]]:
    """
    Flatten SVD components and split into hub_dim-sized vectors.

    Strategy:
        1. Concatenate [U.flatten(), S, Vt.flatten()] into a 1D array.
        2. Split into chunks of exactly hub_dim (last chunk zero-padded).
        3. Each chunk stores its L2 scale so magnitude survives L2-norm in UHS.

    Args:
        snapshot: SVD-compressed weight.
        hub_dim: Target chunk size (must match UHS hub_dim).

    Returns:
        (chunks, metadata) where each chunk is a (hub_dim,) float32 array
        and metadata records how to reassemble them.
    """
    flat = np.concatenate(
        [snapshot.U.flatten(), snapshot.S, snapshot.Vt.flatten()], axis=0
    ).astype(np.float32)

    u_size = snapshot.U.size
    s_size = snapshot.S.size
    vt_size = snapshot.Vt.size
    total = len(flat)

    # Pad to multiple of hub_dim
    n_chunks = max(1, int(np.ceil(total / hub_dim)))
    padded_len = n_chunks * hub_dim
    flat_padded = np.zeros(padded_len, dtype=np.float32)
    flat_padded[:total] = flat

    chunks: List[np.ndarray] = []
    metas: List[ChunkMeta] = []

    for i in range(n_chunks):
        start = i * hub_dim
        chunk = flat_padded[start : start + hub_dim].copy()
        scale = float(np.linalg.norm(chunk))
        metas.append(
            ChunkMeta(
                layer_name=snapshot.layer_name,
                chunk_idx=i,
                total_chunks=n_chunks,
                scale=scale,
                flat_offset=start,
                flat_length=min(hub_dim, total - start) if start < total else 0,
                original_shape=snapshot.original_shape,
                rank=snapshot.rank,
                u_size=u_size,
                s_size=s_size,
                vt_size=vt_size,
            )
        )
        chunks.append(chunk)

    return chunks, metas


# ---------------------------------------------------------------------------
# Hub encoding / decoding of weight chunks
# ---------------------------------------------------------------------------


def encode_weight_chunks(
    chunks: List[np.ndarray],
    uhs,  # UniversalHubSpace — avoid circular import with string annotation
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Encode weight chunks through the UHS encoder.

    Stores the pre-normalisation L2 scale of each chunk so magnitude can be
    restored after decoding (the UHS encoder L2-normalises its output).

    Args:
        chunks: List of (hub_dim,) float32 arrays.
        uhs: Trained UniversalHubSpace instance.

    Returns:
        (encoded_chunks, scales) — encoded hub vectors and their pre-norm scales.
    """
    encoded: List[np.ndarray] = []
    scales: List[float] = []

    if not chunks:
        return encoded, scales

    # Batch encode for efficiency
    batch = torch.tensor(np.stack(chunks, axis=0), dtype=torch.float32)
    with torch.no_grad():
        hub = uhs.encoder(batch.to(uhs.device))  # (N, hub_dim) — L2-normalised

    hub_np = hub.cpu().numpy()
    for i, chunk in enumerate(chunks):
        scale = float(np.linalg.norm(chunk))
        scales.append(scale)
        encoded.append(hub_np[i])

    return encoded, scales


def decode_and_reassemble(
    hub_vectors: List[np.ndarray],
    scales: List[float],
    metas: List[ChunkMeta],
    uhs,  # UniversalHubSpace
    target_shape: Tuple[int, ...],
    target_rank: int,
) -> np.ndarray:
    """
    Decode hub vectors and reassemble into a weight matrix of target_shape.

    Steps:
        1. Decode each hub vector through the UHS decoder.
        2. Re-scale decoded chunk by stored pre-norm scale.
        3. Concatenate chunks into flat array, strip padding.
        4. Unpack U, S, Vt; reshape to target_rank and target_shape dimensions.
        5. Reconstruct weight matrix: W ≈ U @ diag(S) @ Vt.

    Args:
        hub_vectors: Encoded hub vectors (one per chunk).
        scales: Pre-norm scale per chunk (from encode_weight_chunks).
        metas: ChunkMeta list (from chunk_for_hub).
        uhs: Trained UniversalHubSpace for the receiver.
        target_shape: (rows, cols) of the target layer weight.
        target_rank: Rank to use when unpacking SVD components.

    Returns:
        Reconstructed 2D weight matrix of shape target_shape.
    """
    if not hub_vectors or not metas:
        rows, cols = target_shape
        return np.zeros((rows, cols), dtype=np.float32)

    hub_dim = uhs.hub_dim
    hub_tensor = torch.tensor(np.stack(hub_vectors, axis=0), dtype=torch.float32)

    with torch.no_grad():
        decoded = uhs.decoder(hub_tensor.to(uhs.device))  # (N, hub_dim)

    decoded_np = decoded.cpu().numpy()

    # Re-scale and concatenate
    flat_parts = []
    for i, (vec, scale) in enumerate(zip(decoded_np, scales)):
        rescaled = vec * scale  # restore magnitude
        flat_parts.append(rescaled)

    flat = np.concatenate(flat_parts, axis=0)

    # Determine expected flat size from meta
    meta0 = metas[0]
    rank = min(target_rank, meta0.rank)
    rows, cols = target_shape
    u_size = rows * rank
    s_size = rank
    vt_size = rank * cols
    needed = u_size + s_size + vt_size

    flat = flat[:needed]  # strip padding

    if len(flat) < needed:
        # Pad with zeros if decode produced less (shouldn't happen in practice)
        flat = np.pad(flat, (0, needed - len(flat)))

    # Unpack SVD components
    U = flat[:u_size].reshape(rows, rank).astype(np.float32)
    S = flat[u_size : u_size + s_size].astype(np.float32)
    Vt = flat[u_size + s_size :].reshape(rank, cols).astype(np.float32)

    # Ensure S is non-negative (magnitude may have flipped sign)
    S = np.abs(S)

    # Reconstruct weight
    W = (U * S[None, :]) @ Vt  # (rows, cols)
    return W


# ---------------------------------------------------------------------------
# Receiver weight initialisation
# ---------------------------------------------------------------------------


def initialize_receiver_weights(
    receiver: nn.Module,
    decoded_weights: Dict[str, np.ndarray],
):
    """
    Load decoded weight matrices into the receiver model's parameters.

    For dimension mismatches between decoded and actual parameter shapes,
    applies zero-padding or centre-crop so the load never fails.

    Args:
        receiver: Target model whose weights will be updated.
        decoded_weights: {layer_name: 2D numpy array} to load.
    """
    modules = dict(receiver.named_modules())

    with torch.no_grad():
        for name, W_decoded in decoded_weights.items():
            if name not in modules:
                logger.warning(f"Layer '{name}' not found in receiver; skipping.")
                continue

            module = modules[name]
            if not hasattr(module, "weight") or module.weight is None:
                logger.warning(f"Layer '{name}' has no weight; skipping.")
                continue

            param = module.weight
            target_shape = param.shape

            # Flatten Conv2d target to 2D for assignment
            if param.ndim > 2:
                rows, cols = target_shape[0], int(np.prod(target_shape[1:]))
            else:
                rows, cols = target_shape

            # Adapt decoded shape to target shape via crop / pad
            W = _adapt_weight(W_decoded, (rows, cols))

            if param.ndim > 2:
                W = W.reshape(target_shape)

            param.copy_(torch.tensor(W, dtype=param.dtype))
            logger.debug(f"  Initialised '{name}': {tuple(target_shape)}")


def _adapt_weight(W: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """
    Adapt a 2D weight matrix to target shape via cropping and/or zero-padding.
    """
    src_rows, src_cols = W.shape
    tgt_rows, tgt_cols = target

    # Rows
    if src_rows >= tgt_rows:
        W = W[:tgt_rows, :]
    else:
        pad = np.zeros((tgt_rows - src_rows, W.shape[1]), dtype=W.dtype)
        W = np.concatenate([W, pad], axis=0)

    # Cols
    if src_cols >= tgt_cols:
        W = W[:, :tgt_cols]
    else:
        pad = np.zeros((W.shape[0], tgt_cols - src_cols), dtype=W.dtype)
        W = np.concatenate([W, pad], axis=1)

    return W


# ---------------------------------------------------------------------------
# Weight statistics for drift measurement
# ---------------------------------------------------------------------------


def compute_weight_stats(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, WeightStats]:
    """
    Compute per-layer weight distribution statistics.

    Used by WeightDriftMeasure to compare transmitter vs. receiver weights
    without needing a dataloader.

    Args:
        model: PyTorch model.
        layer_names: Layers to measure (auto-detect if None).

    Returns:
        {layer_name: WeightStats}
    """
    weights = extract_weights(model, layer_names)
    stats: Dict[str, WeightStats] = {}

    for name, W in weights.items():
        singular_values = np.linalg.svd(W, compute_uv=False)
        sv_sum = singular_values.sum()
        sv_sq_sum = (singular_values**2).sum()
        effective_rank = (sv_sum**2) / (sv_sq_sum + 1e-12)

        stats[name] = WeightStats(
            layer_name=name,
            mean=float(W.mean()),
            std=float(W.std()),
            frobenius_norm=float(np.linalg.norm(W, "fro")),
            spectral_norm=float(singular_values[0]) if len(singular_values) > 0 else 0.0,
            effective_rank=float(effective_rank),
            shape=W.shape,
        )

    return stats
