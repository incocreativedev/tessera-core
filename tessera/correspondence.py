"""
tessera.correspondence — CKA-based layer matching for cross-architecture transfer.

When transmitter and receiver have different numbers of layers (e.g. 12 vs 24),
we need to determine which tx layer corresponds to which rx layer. Linear CKA
computes a similarity score between every (tx_layer, rx_layer) pair on shared
reference data, then greedy or optimal (Hungarian) assignment finds the best
one-to-one correspondence.

Usage:
    corr = LayerCorrespondence(transmitter, receiver, device="cpu")
    matches = corr.compute(dataloader)
    # matches: [(tx_layer, rx_layer, cka_score), ...]
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import setup_logging

logger = setup_logging("tessera.correspondence")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear Centered Kernel Alignment between two activation matrices.

    Uses the efficient HSIC formulation that avoids constructing the full
    (N, N) Gram matrices, making it O(N * d) rather than O(N²).

    Args:
        X: (N, d_x) activation matrix from one layer.
        Y: (N, d_y) activation matrix from another layer.

    Returns:
        CKA similarity in [0, 1]. 1.0 = identical representations, 0.0 = unrelated.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples; got {X.shape[0]} vs {Y.shape[0]}")

    # Center the activations (remove mean)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    n = X.shape[0]

    # Efficient HSIC via Frobenius norms:
    #   HSIC(K, L) ≈ ||Y.T @ X||_F^2 / n^2  for linear kernels
    YtX = Y.T @ X  # (d_y, d_x)
    hsic_xy = np.sum(YtX**2) / (n**2)

    XtX = X.T @ X  # (d_x, d_x)
    hsic_xx = np.sum(XtX**2) / (n**2)

    YtY = Y.T @ Y  # (d_y, d_y)
    hsic_yy = np.sum(YtY**2) / (n**2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def _auto_detect_weight_layers(model: nn.Module) -> List[str]:
    """
    Auto-detect hookable layers (Linear and Conv2d) for weight correspondence.

    Mirrors the fallback logic in fingerprint._auto_detect_layers() but targets
    weight-bearing modules directly.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name:
            layers.append(name)
    return layers


def _auto_detect_activation_layers(model: nn.Module) -> List[str]:
    """
    Auto-detect hookable layers for activation-based CKA (same logic as fingerprint.py).
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            layers.append(name)
        elif isinstance(module, nn.TransformerDecoderLayer):
            layers.append(name)

    if not layers:
        for name, module in model.named_children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                for sub_name, _ in module.named_children():
                    layers.append(f"{name}.{sub_name}")
            elif not isinstance(module, (nn.Embedding, nn.Linear)):
                layers.append(name)
    return layers


def collect_layer_activations(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Collect activations from specified layers via forward hooks.

    Uses the same hook pattern as tessera.transfer.ModeATransfer._collect_activations().

    Args:
        model: PyTorch model.
        dataloader: Reference data (inputs only; labels are ignored).
        layer_names: Module names to hook (must match model.named_modules()).
        device: Compute device.

    Returns:
        {layer_name: (N, d_layer) numpy array} — one row per token/sample.
    """
    activations: Dict[str, list] = defaultdict(list)
    hooks = []

    def make_hook(name: str):
        def fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach().cpu()
            if out.ndim == 3:
                # (batch, seq, d) → pool over sequence → (batch, d)
                out = out.mean(dim=1)
            elif out.ndim > 2:
                out = out.reshape(out.shape[0], -1)
            activations[name].append(out.numpy())

        return fn

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))

    for h in hooks:
        h.remove()

    result = {}
    for name in layer_names:
        if activations[name]:
            result[name] = np.concatenate(activations[name], axis=0).astype(np.float32)
    return result


def compute_cka_matrix(
    tx_acts: Dict[str, np.ndarray],
    rx_acts: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute the full CKA similarity matrix between all tx and rx layers.

    Args:
        tx_acts: {layer_name: (N, d)} activations for the transmitter.
        rx_acts: {layer_name: (N, d)} activations for the receiver.

    Returns:
        (similarity_matrix, tx_layer_names, rx_layer_names)
        similarity_matrix has shape (len(tx_layers), len(rx_layers)).
    """
    tx_names = sorted(tx_acts.keys())
    rx_names = sorted(rx_acts.keys())

    matrix = np.zeros((len(tx_names), len(rx_names)), dtype=np.float32)
    for i, tx_name in enumerate(tx_names):
        for j, rx_name in enumerate(rx_names):
            X = tx_acts[tx_name]
            Y = rx_acts[rx_name]
            # Truncate to shared number of samples
            n = min(len(X), len(Y))
            matrix[i, j] = linear_cka(X[:n], Y[:n])

    return matrix, tx_names, rx_names


def match_layers(
    cka_matrix: np.ndarray,
    tx_layer_names: List[str],
    rx_layer_names: List[str],
    strategy: str = "greedy",
) -> List[Tuple[str, str, float]]:
    """
    Find best layer correspondences from the CKA similarity matrix.

    Args:
        cka_matrix: (n_tx, n_rx) matrix of CKA similarity scores.
        tx_layer_names: Transmitter layer names (row labels).
        rx_layer_names: Receiver layer names (column labels).
        strategy: "greedy" selects argmax per tx row (fast);
                  "hungarian" finds the globally optimal one-to-one assignment
                  via scipy.optimize.linear_sum_assignment (slower but better).

    Returns:
        List of (tx_layer, rx_layer, cka_score) tuples sorted by tx layer order.
    """
    if strategy == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment

            # Hungarian minimises cost; negate for maximisation
            row_ind, col_ind = linear_sum_assignment(-cka_matrix)
            matches = []
            for r, c in zip(row_ind, col_ind):
                matches.append((tx_layer_names[r], rx_layer_names[c], float(cka_matrix[r, c])))
            matches.sort(key=lambda t: tx_layer_names.index(t[0]))
            return matches
        except ImportError:
            logger.warning("scipy not available; falling back to greedy matching.")
            strategy = "greedy"

    # Greedy: for each tx layer pick the best rx layer (with replacement allowed)
    matches = []
    for i, tx_name in enumerate(tx_layer_names):
        j = int(np.argmax(cka_matrix[i]))
        matches.append((tx_name, rx_layer_names[j], float(cka_matrix[i, j])))
    return matches


class LayerCorrespondence:
    """
    High-level API: compute CKA-based layer correspondence between two models.

    Example:
        corr = LayerCorrespondence(transmitter, receiver, device="cpu")
        matches = corr.compute(dataloader)
        for tx_layer, rx_layer, score in matches:
            print(f"{tx_layer} → {rx_layer}  (CKA={score:.3f})")
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        device: str = "cpu",
    ):
        """
        Args:
            transmitter: Trained source model.
            receiver: Target model.
            device: Compute device.
        """
        self.transmitter = transmitter
        self.receiver = receiver
        self.device = device

    def compute(
        self,
        dataloader: DataLoader,
        tx_layers: Optional[List[str]] = None,
        rx_layers: Optional[List[str]] = None,
        strategy: str = "greedy",
    ) -> List[Tuple[str, str, float]]:
        """
        Compute CKA-based layer correspondence.

        Args:
            dataloader: Shared reference data (must produce inputs both models accept).
            tx_layers: Transmitter layer names to consider (auto-detect if None).
            rx_layers: Receiver layer names to consider (auto-detect if None).
            strategy: "greedy" or "hungarian".

        Returns:
            List of (tx_layer, rx_layer, cka_score) tuples.
        """
        tx_layer_names = tx_layers or _auto_detect_activation_layers(self.transmitter)
        rx_layer_names = rx_layers or _auto_detect_activation_layers(self.receiver)

        logger.info(
            f"Computing CKA correspondence: {len(tx_layer_names)} tx layers × "
            f"{len(rx_layer_names)} rx layers"
        )

        tx_acts = collect_layer_activations(
            self.transmitter, dataloader, tx_layer_names, self.device
        )
        rx_acts = collect_layer_activations(
            self.receiver, dataloader, rx_layer_names, self.device
        )

        if not tx_acts or not rx_acts:
            logger.warning("No activations collected — returning empty correspondence.")
            return []

        matrix, tx_names, rx_names = compute_cka_matrix(tx_acts, rx_acts)
        matches = match_layers(matrix, tx_names, rx_names, strategy=strategy)

        for tx_name, rx_name, score in matches:
            logger.info(f"  {tx_name} → {rx_name}  (CKA={score:.3f})")

        return matches

    def cka_matrix(
        self,
        dataloader: DataLoader,
        tx_layers: Optional[List[str]] = None,
        rx_layers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Return the raw CKA similarity matrix for visualisation or debugging.

        Returns:
            (matrix, tx_layer_names, rx_layer_names)
        """
        tx_layer_names = tx_layers or _auto_detect_activation_layers(self.transmitter)
        rx_layer_names = rx_layers or _auto_detect_activation_layers(self.receiver)

        tx_acts = collect_layer_activations(
            self.transmitter, dataloader, tx_layer_names, self.device
        )
        rx_acts = collect_layer_activations(
            self.receiver, dataloader, rx_layer_names, self.device
        )

        return compute_cka_matrix(tx_acts, rx_acts)
