"""
tessera.fingerprint — Activation fingerprinting for anchor characterisation.

Fingerprinting computes per-layer activation statistics by running a reference
dataset through a model in inference mode. The resulting fingerprint captures
the statistical geometry of each layer's activation space:

    - Mean activation vector (centroid)
    - Variance vector (per-dimension spread)
    - Top-k PCA components (principal directions)
    - Intrinsic dimensionality via participation ratio

These fingerprints enable:
    1. Compatibility scoring between models (cosine similarity of fingerprints)
    2. CKA-based layer correspondence for cross-architecture transfer
    3. UHS encoder/decoder training targets
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class LayerFingerprint:
    """
    Statistical fingerprint of a single layer's activation space.

    Computed over a reference dataset (TSRD in production, any dataset in demo).
    """

    layer_name: str
    layer_idx: int

    # Per-dimension statistics — shape: (d_layer,)
    mean: np.ndarray
    variance: np.ndarray

    # Principal components — eigenvectors of the covariance matrix
    pca_top5: np.ndarray        # shape: (5, d_layer)
    pca_top5_ev: np.ndarray     # shape: (5,) — eigenvalues
    pca_top10: np.ndarray       # shape: (10, d_layer)
    pca_top10_ev: np.ndarray    # shape: (10,) — eigenvalues

    # Intrinsic dimensionality: (Σλ)² / Σλ²
    intrinsic_dim: float

    # Metadata
    d_layer: int
    token_count: int

    def cosine_similarity(self, other: "LayerFingerprint") -> float:
        """
        Compute cosine similarity between two fingerprints using mean vectors.

        This is the primary compatibility metric in the Tessera protocol.
        """
        dot = np.dot(self.mean, other.mean)
        norm_a = np.linalg.norm(self.mean)
        norm_b = np.linalg.norm(other.mean)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


class ActivationFingerprint:
    """
    Hook-based activation collector and fingerprint computer.

    Usage:
        fp = ActivationFingerprint(model, target_layers=["layers.0", "layers.1"])
        fingerprints = fp.collect(dataloader, device="cpu")
        # fingerprints: {layer_name: LayerFingerprint, ...}

    The collector registers forward hooks on named modules, stores activations
    during inference, then computes statistics over the collected data.
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None):
        """
        Args:
            model: PyTorch model to fingerprint.
            target_layers: Module names to hook. If None, auto-detects
                           transformer blocks / linear layers.
        """
        self.model = model
        self.target_layers = target_layers or self._auto_detect_layers()
        self._hooks: list = []
        self._activations: Dict[str, list] = defaultdict(list)

    def _auto_detect_layers(self) -> List[str]:
        """
        Auto-detect hookable layers (transformer encoder layers or top-level children).
        """
        layers = []
        for name, module in self.model.named_modules():
            # Match common transformer block patterns
            if isinstance(module, nn.TransformerEncoderLayer):
                layers.append(name)
            elif isinstance(module, nn.TransformerDecoderLayer):
                layers.append(name)
        # Fallback: use direct children if no transformer layers found
        if not layers:
            for name, module in self.model.named_children():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    for sub_name, sub_module in module.named_children():
                        layers.append(f"{name}.{sub_name}")
                elif not isinstance(module, (nn.Embedding, nn.Linear)):
                    layers.append(name)
        return layers

    def _register_hooks(self):
        """Attach forward hooks to target layers."""
        self._activations.clear()
        self._hooks.clear()

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Create a hook function that stores activations for a named layer."""
        def hook_fn(module, input, output):
            # Handle various output types
            if isinstance(output, tuple):
                act = output[0]
            elif isinstance(output, torch.Tensor):
                act = output
            else:
                return  # Skip unsupported output types

            # Flatten to (N_tokens, d_layer)
            act = act.detach().cpu()
            if act.ndim == 3:
                # (batch, seq_len, d) → (batch*seq_len, d)
                act = act.reshape(-1, act.shape[-1])
            elif act.ndim == 2:
                pass  # Already (batch, d)
            else:
                act = act.reshape(-1, act.shape[-1])

            self._activations[layer_name].append(act)

        return hook_fn

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def collect(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> Dict[str, LayerFingerprint]:
        """
        Run inference and compute per-layer fingerprints.

        Args:
            dataloader: Reference dataset loader.
            device: Compute device.

        Returns:
            Dictionary mapping layer names to LayerFingerprint objects.
        """
        self.model.eval()
        self.model.to(device)
        self._register_hooks()

        # Collect activations
        with torch.no_grad():
            for batch in dataloader:
                # Handle (input, label) tuples and plain tensors
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                self.model(x)

        self._remove_hooks()

        # Compute fingerprints
        fingerprints = {}
        for idx, layer_name in enumerate(self.target_layers):
            if layer_name not in self._activations:
                continue
            if len(self._activations[layer_name]) == 0:
                continue

            H = torch.cat(self._activations[layer_name], dim=0).float().numpy()
            fingerprints[layer_name] = self._compute_stats(H, layer_name, idx)

        self._activations.clear()
        return fingerprints

    def _compute_stats(
        self, H: np.ndarray, layer_name: str, layer_idx: int
    ) -> LayerFingerprint:
        """
        Compute statistics from collected activations.

        Args:
            H: Activation matrix, shape (N, d).
            layer_name: Name of the layer.
            layer_idx: Index in target_layers list.

        Returns:
            LayerFingerprint with all statistics populated.
        """
        N, d = H.shape

        # Mean and variance
        mean = H.mean(axis=0)
        variance = H.var(axis=0)

        # Covariance and PCA via eigendecomposition
        H_centered = H - mean
        cov = (H_centered.T @ H_centered) / max(N - 1, 1)

        # Use scipy for stable eigendecomposition of symmetric matrix
        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(cov)
        except ImportError:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Top-k PCA
        k5 = min(5, d)
        k10 = min(10, d)
        pca_top5 = eigenvectors[:, :k5].T       # (k5, d)
        pca_top5_ev = eigenvalues[:k5]
        pca_top10 = eigenvectors[:, :k10].T      # (k10, d)
        pca_top10_ev = eigenvalues[:k10]

        # Intrinsic dimensionality: participation ratio
        ev_sum = eigenvalues.sum()
        ev_sq_sum = (eigenvalues ** 2).sum()
        intrinsic_dim = (ev_sum ** 2) / ev_sq_sum if ev_sq_sum > 0 else 0.0

        return LayerFingerprint(
            layer_name=layer_name,
            layer_idx=layer_idx,
            mean=mean,
            variance=variance,
            pca_top5=pca_top5,
            pca_top5_ev=pca_top5_ev,
            pca_top10=pca_top10,
            pca_top10_ev=pca_top10_ev,
            intrinsic_dim=float(intrinsic_dim),
            d_layer=d,
            token_count=N,
        )


def compute_fingerprints(
    model: nn.Module,
    dataloader: DataLoader,
    target_layers: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, LayerFingerprint]:
    """
    Convenience function: compute activation fingerprints for a model.

    Args:
        model: PyTorch model.
        dataloader: Reference dataset.
        target_layers: Layers to fingerprint (auto-detected if None).
        device: Compute device.

    Returns:
        {layer_name: LayerFingerprint, ...}
    """
    fp = ActivationFingerprint(model, target_layers)
    return fp.collect(dataloader, device)
