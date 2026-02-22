"""
tessera.drift — Transfer fidelity measurement via KL-divergence.

Drift D quantifies how much knowledge is lost during transfer.
Defined as the KL divergence of activation distributions on a shared
reference dataset:

    D(Alice → Bob) = D_KL( P_Alice(y|R) || P_Bob(y|R) )

Interpretation:
    D = 0       → Perfect transfer
    D < τ_accept → Acceptable transfer
    D > τ_accept → Triggers retransmission with adjusted projection parameters

This module fits Gaussian distributions to per-layer activations and
computes the closed-form KL divergence between them.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import setup_logging

logger = setup_logging("tessera.drift")


class DriftMeasure:
    """
    Measures activation-based fidelity between transmitter and receiver.

    Fits Gaussian N(μ, Σ) to each model's per-layer activations on shared
    reference data, then computes the KL divergence between them.
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        device: str = "cpu",
    ):
        self.transmitter = transmitter
        self.receiver = receiver
        self.device = device

    def compute(
        self,
        dataloader: DataLoader,
        target_layers_tx: Optional[list] = None,
        target_layers_rx: Optional[list] = None,
    ) -> float:
        """
        Compute average KL-divergence across matched layers.

        Args:
            dataloader: Shared reference data.
            target_layers_tx: Transmitter layers to measure (auto-detect if None).
            target_layers_rx: Receiver layers to measure (auto-detect if None).

        Returns:
            Average KL-divergence (lower = better fidelity). 0.0 = perfect.
        """
        logger.info("  Collecting transmitter activation statistics...")
        tx_stats = self._collect_statistics(self.transmitter, dataloader, target_layers_tx)

        logger.info("  Collecting receiver activation statistics...")
        rx_stats = self._collect_statistics(self.receiver, dataloader, target_layers_rx)

        if not tx_stats or not rx_stats:
            logger.warning("  No matching layers found for drift measurement.")
            return 0.0

        # Match layers by index (simplest approach for different architectures)
        tx_layers = sorted(tx_stats.keys())
        rx_layers = sorted(rx_stats.keys())
        n_pairs = min(len(tx_layers), len(rx_layers))

        kl_divs = []
        for i in range(n_pairs):
            tx_name = tx_layers[i]
            rx_name = rx_layers[i]

            mu_p, sigma_p = tx_stats[tx_name]
            mu_q, sigma_q = rx_stats[rx_name]

            # Dimensions must match for KL — use diagonal approximation if not
            if len(mu_p) != len(mu_q):
                # Truncate to smaller dimension for comparison
                d = min(len(mu_p), len(mu_q))
                mu_p, sigma_p = mu_p[:d], sigma_p[:d]
                mu_q, sigma_q = mu_q[:d], sigma_q[:d]

            kl = self._kl_diagonal_gaussian(mu_p, sigma_p, mu_q, sigma_q)
            kl_divs.append(kl)

        avg_kl = float(np.mean(kl_divs)) if kl_divs else 0.0
        logger.info(f"  Drift: {avg_kl:.6f} (across {len(kl_divs)} layer pairs)")
        return avg_kl

    def _collect_statistics(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        target_layers: Optional[list] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Collect per-layer activation mean and variance (diagonal covariance).

        Returns:
            {layer_name: (mean, variance)} where each is shape (d_layer,).
        """
        activations: Dict[str, list] = defaultdict(list)
        hooks = []

        # Determine which modules to hook
        if target_layers is None:
            target_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    target_layers.append(name)
            # Fallback to ModuleList children
            if not target_layers:
                for name, module in model.named_children():
                    if isinstance(module, nn.ModuleList):
                        for sub_name, _ in module.named_children():
                            target_layers.append(f"{name}.{sub_name}")

        def make_hook(layer_name):
            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                act = act.detach().cpu()
                if act.ndim >= 2:
                    act = act.reshape(-1, act.shape[-1])
                activations[layer_name].append(act.numpy())

            return hook_fn

        for name, module in model.named_modules():
            if name in target_layers:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                model(x.to(self.device))

        for h in hooks:
            h.remove()

        # Compute stats
        stats = {}
        for name, acts in activations.items():
            H = np.concatenate(acts, axis=0).astype(np.float64)
            mean = H.mean(axis=0)
            var = H.var(axis=0) + 1e-8  # Stabilise
            stats[name] = (mean, var)

        return stats

    @staticmethod
    def _kl_diagonal_gaussian(
        mu_p: np.ndarray,
        var_p: np.ndarray,
        mu_q: np.ndarray,
        var_q: np.ndarray,
    ) -> float:
        """
        KL divergence KL(P || Q) for diagonal Gaussians.

        KL = 0.5 × Σ_i [ var_p_i / var_q_i + (μ_q_i - μ_p_i)² / var_q_i
                          - 1 + log(var_q_i / var_p_i) ]

        Uses diagonal covariance for computational efficiency (full covariance
        is O(d²) in memory and O(d³) to invert).
        """
        d = len(mu_p)
        var_p = np.maximum(var_p, 1e-8)
        var_q = np.maximum(var_q, 1e-8)

        term1 = np.sum(var_p / var_q)
        term2 = np.sum((mu_q - mu_p) ** 2 / var_q)
        term3 = np.sum(np.log(var_q / var_p))

        kl = 0.5 * (term1 + term2 - d + term3)
        return max(0.0, float(kl))


class WeightDriftMeasure:
    """
    Measures weight-space fidelity between transmitter and receiver.

    Unlike DriftMeasure (which needs a dataloader), this class computes
    drift directly from model parameters — no forward pass required.

    Compares per-layer weight statistics (mean, std, Frobenius norm,
    spectral norm, effective rank) between matched layers and returns
    a composite drift score (lower = better).
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        device: str = "cpu",
    ):
        self.transmitter = transmitter
        self.receiver = receiver
        self.device = device

    def compute(
        self,
        tx_layers: Optional[list] = None,
        rx_layers: Optional[list] = None,
        correspondences: Optional[List[Tuple[str, str]]] = None,
    ) -> float:
        """
        Compute average weight-distribution drift across matched layers.

        Args:
            tx_layers: Transmitter layer names (auto-detect if None).
            rx_layers: Receiver layer names (auto-detect if None).
            correspondences: Pre-computed [(tx_layer, rx_layer)] pairs.
                             If provided, overrides tx_layers/rx_layers.

        Returns:
            Composite drift score (lower = better). 0.0 = identical statistics.
        """
        from .weight_ops import compute_weight_stats

        if correspondences:
            tx_names = [p[0] for p in correspondences]
            rx_names = [p[1] for p in correspondences]
            tx_stats = compute_weight_stats(self.transmitter, tx_names)
            rx_stats = compute_weight_stats(self.receiver, rx_names)
            pairs = [(p[0], p[1]) for p in correspondences]
        else:
            tx_stats = compute_weight_stats(self.transmitter, tx_layers)
            rx_stats = compute_weight_stats(self.receiver, rx_layers)
            tx_keys = sorted(tx_stats.keys())
            rx_keys = sorted(rx_stats.keys())
            n = min(len(tx_keys), len(rx_keys))
            pairs = list(zip(tx_keys[:n], rx_keys[:n]))

        if not pairs:
            logger.warning("No layer pairs for weight drift measurement.")
            return 0.0

        scores = []
        for tx_name, rx_name in pairs:
            if tx_name not in tx_stats or rx_name not in rx_stats:
                continue
            d = self._weight_distance(tx_stats[tx_name], rx_stats[rx_name])
            scores.append(d)

        drift = float(np.mean(scores)) if scores else 0.0
        logger.info(f"  Weight drift: {drift:.6f} (across {len(scores)} layer pairs)")
        return drift

    @staticmethod
    def _weight_distance(tx_stat, rx_stat) -> float:
        """
        Normalised distance between weight statistics of two layers.

        Combines:
          - Relative mean difference:     |μ_tx - μ_rx| / (|μ_tx| + ε)
          - Relative std difference:      |σ_tx - σ_rx| / (σ_tx + ε)
          - Spectral norm ratio deviation: |1 - sn_rx/sn_tx|
          - Effective rank ratio deviation: |1 - er_rx/er_tx|
        """
        eps = 1e-8

        mean_diff = abs(tx_stat.mean - rx_stat.mean) / (abs(tx_stat.mean) + eps)
        std_diff = abs(tx_stat.std - rx_stat.std) / (tx_stat.std + eps)
        sn_ratio = abs(1.0 - rx_stat.spectral_norm / (tx_stat.spectral_norm + eps))
        er_ratio = abs(1.0 - rx_stat.effective_rank / (tx_stat.effective_rank + eps))

        return float(np.mean([mean_diff, std_diff, sn_ratio, er_ratio]))
