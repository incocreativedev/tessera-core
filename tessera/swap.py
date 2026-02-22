"""
tessera.swap — SWAP projection for bidirectional hub-space transfer.

A SWAP projection simultaneously maps both models into shared hub space
and optimises the mapping to minimise bidirectional round-trip error.
This is useful when both transmitter and receiver are expected to act as
future transmitters (e.g. swarm contributors that also receive broadcasts).

Status: Planned for v0.2.0. This stub exposes the public API surface so that
        imports in __init__.py resolve.
"""

from typing import Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from .uhs import UniversalHubSpace
from .utils import setup_logging

logger = setup_logging("tessera.swap")


class SWAPProjection:
    """
    Bidirectional SWAP projection — planned for v0.2.0.

    Jointly optimises encoder/decoder pairs for two models so that
    A → hub → B and B → hub → A are both minimised simultaneously.
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        device: str = "cpu",
        hub_dim: int = 2048,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.device = device
        self.hub_dim = hub_dim
        self.uhs_a: Optional[UniversalHubSpace] = None
        self.uhs_b: Optional[UniversalHubSpace] = None

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
    ) -> None:
        """Fit bidirectional projection (not yet implemented)."""
        raise NotImplementedError(
            "SWAPProjection is planned for v0.2.0. "
            "Use ModeATransfer or ModeWTransfer for one-directional transfer."
        )

    def project_a_to_b(self, dataloader: DataLoader) -> None:
        """Project model A knowledge into model B (not yet implemented)."""
        raise NotImplementedError("SWAPProjection is planned for v0.2.0.")

    def project_b_to_a(self, dataloader: DataLoader) -> None:
        """Project model B knowledge into model A (not yet implemented)."""
        raise NotImplementedError("SWAPProjection is planned for v0.2.0.")
