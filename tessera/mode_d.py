"""
tessera.mode_d — Mode D (Dataset) knowledge transfer orchestrator.

Mode D packages knowledge as synthetic dataset distillations rather than
model-internal representations, enabling transfer to receivers that cannot
run forward passes on the transmitter's architecture.

Status: Planned for v0.2.0. This stub exposes the public API surface so that
        imports in __init__.py resolve. Calling execute() raises NotImplementedError.
"""

from typing import List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from .token import TesseraToken
from .utils import setup_logging

logger = setup_logging("tessera.mode_d")


class ModeDTransfer:
    """
    Mode D (Dataset) knowledge transfer — planned for v0.2.0.

    Distils transmitter knowledge into synthetic datasets or soft-label
    corpora that the receiver can train on independently.
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
        """Execute Mode D transfer (not yet implemented)."""
        raise NotImplementedError(
            "Mode D (Dataset) transfer is planned for v0.2.0. "
            "Use ModeATransfer or ModeWTransfer in the meantime."
        )
