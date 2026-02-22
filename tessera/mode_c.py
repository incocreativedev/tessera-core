"""
tessera.mode_c — Mode C (Compressed) knowledge transfer orchestrator.

Mode C applies aggressive quantisation and structured pruning to produce
compact tokens optimised for bandwidth-constrained edge deployment.

Status: Planned for v0.2.0. This stub exposes the public API surface so that
        imports in __init__.py resolve. Calling execute() raises NotImplementedError.
"""

from typing import List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from .token import TesseraToken
from .utils import setup_logging

logger = setup_logging("tessera.mode_c")


class ModeCTransfer:
    """
    Mode C (Compressed) knowledge transfer — planned for v0.2.0.

    Produces highly compressed tokens (INT8 / structured pruning) suitable
    for transmission to resource-constrained receivers.
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        transmitter_id: str = "transmitter",
        receiver_id: str = "receiver",
        device: str = "cpu",
        hub_dim: int = 2048,
    ):
        self.transmitter = transmitter
        self.receiver = receiver
        self.transmitter_id = transmitter_id
        self.receiver_id = receiver_id
        self.device = device
        self.hub_dim = hub_dim

    def execute(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tx_layers: Optional[List[str]] = None,
        rx_layers: Optional[List[str]] = None,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ) -> TesseraToken:
        """Execute Mode C transfer (not yet implemented)."""
        raise NotImplementedError(
            "Mode C (Compressed) transfer is planned for v0.2.0. "
            "Use ModeATransfer or ModeWTransfer in the meantime."
        )
