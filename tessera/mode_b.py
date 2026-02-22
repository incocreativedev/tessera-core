"""
tessera.mode_b — Mode B (Behavioural) knowledge transfer orchestrator.

Mode B matches model outputs / logit distributions rather than internal
activations or weights, making it suitable for black-box knowledge distillation
where internal layer access is unavailable.

Status: Planned for v0.2.0. This stub exposes the public API surface so that
        imports in __init__.py resolve. Calling execute() raises NotImplementedError.
"""

import datetime
from typing import List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from .token import KnowledgeType, TesseraToken
from .utils import setup_logging

logger = setup_logging("tessera.mode_b")


class ModeBTransfer:
    """
    Mode B (Behavioural) knowledge transfer — planned for v0.2.0.

    Transfers knowledge by aligning the output distributions (logits / softmax)
    of the transmitter and receiver on shared reference data.
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
        """Execute Mode B transfer (not yet implemented)."""
        raise NotImplementedError(
            "Mode B (Behavioural) transfer is planned for v0.2.0. "
            "Use ModeATransfer for activation-based transfer or "
            "ModeWTransfer for weight-based transfer."
        )
