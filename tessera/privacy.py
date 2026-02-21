"""
tessera.privacy — Differential privacy via the Gaussian mechanism.

Each Tessera token carries a privacy budget (ε, δ). Before serialisation,
Gaussian noise calibrated to this budget is added to the knowledge vector,
providing formal differential privacy guarantees.

The sensitivity Δf is bounded by the L2 norm of the knowledge representation.
Cumulative budget tracking across multi-token transmissions uses the moments
accountant (not implemented in this reference; single-token budgets only).
"""

import math

import numpy as np


class DifferentialPrivacy:
    """
    Gaussian mechanism for (ε, δ)-differential privacy.

    Adds noise N(0, σ²I) where σ is calibrated from the privacy budget.

    Args:
        epsilon: Privacy budget (smaller = more private). Use float('inf') for no noise.
        delta: Failure probability.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive (use float('inf') for no noise)")
        if not (0 < delta < 1):
            raise ValueError("Delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.sigma = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        """
        Calibrate noise standard deviation from (ε, δ).

        Standard Gaussian mechanism formula:
            σ ≥ Δf × √(2 ln(1.25/δ)) / ε

        We assume unit sensitivity (Δf = 1) for L2-normalised hub vectors.
        """
        if math.isinf(self.epsilon):
            return 0.0
        return math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, vector: np.ndarray) -> np.ndarray:
        """
        Add calibrated Gaussian noise to a vector.

        Args:
            vector: 1-D or 2-D numpy array.

        Returns:
            Noisy copy of the vector (same shape).
        """
        if self.sigma == 0.0:
            return vector.copy()
        noise = np.random.normal(0.0, self.sigma, size=vector.shape)
        return vector + noise

    def add_noise_tensor(self, tensor) -> "torch.Tensor":
        """
        Add calibrated Gaussian noise to a PyTorch tensor.

        Args:
            tensor: Any-shape torch.Tensor.

        Returns:
            Noisy copy on the same device.
        """
        import torch

        if self.sigma == 0.0:
            return tensor.clone()
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

    def __repr__(self) -> str:
        return f"DifferentialPrivacy(ε={self.epsilon}, δ={self.delta}, σ={self.sigma:.6f})"
