"""
tessera.registry — Local file-based anchor model registry.

The Anchor Registry stores characterised models and their pre-trained UHS
encoder/decoder pairs. In production, this would be a hosted service
(HuggingFace Hub, S3, etc.). This reference implementation uses a local
directory structure:

    ~/.tessera/
    ├── registry.json           # Index of all registered anchors
    └── anchors/
        ├── model_a/
        │   ├── config.json     # Anchor metadata
        │   ├── encoder.pt      # UHS encoder weights
        │   └── decoder.pt      # UHS decoder weights
        └── model_b/
            └── ...
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .uhs import EncoderMLP, DecoderMLP
from .utils import setup_logging

logger = setup_logging("tessera.registry")


class AnchorRegistry:
    """
    Local registry for anchor models and their UHS pairs.

    Anchors are identified by unique string IDs (e.g. "llama3-8b",
    "simple_tx_128d"). Each anchor stores its encoder, decoder,
    model dimension, and optional metadata.
    """

    def __init__(self, registry_dir: Optional[str] = None):
        """
        Args:
            registry_dir: Root directory for the registry.
                          Defaults to ~/.tessera/
        """
        if registry_dir is None:
            registry_dir = str(Path.home() / ".tessera")

        self.root = Path(registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.anchors_dir = self.root / "anchors"
        self.anchors_dir.mkdir(exist_ok=True)
        self.index_path = self.root / "registry.json"

        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Load the registry index from disk."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"version": "1.0", "anchors": {}}

    def _save_index(self):
        """Persist the registry index."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register(
        self,
        anchor_id: str,
        d_model: int,
        encoder: EncoderMLP,
        decoder: DecoderMLP,
        metadata: Optional[dict] = None,
    ):
        """
        Register an anchor with its pre-trained UHS pair.

        Args:
            anchor_id: Unique identifier (e.g. "simple_tx_128d").
            d_model: Model embedding dimension.
            encoder: Trained EncoderMLP.
            decoder: Trained DecoderMLP.
            metadata: Optional metadata dict.
        """
        anchor_dir = self.anchors_dir / anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(encoder.state_dict(), anchor_dir / "encoder.pt")
        torch.save(decoder.state_dict(), anchor_dir / "decoder.pt")

        # Save config
        config = {
            "anchor_id": anchor_id,
            "d_model": d_model,
            "hub_dim": encoder.hub_dim,
            **(metadata or {}),
        }
        with open(anchor_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Update index
        self._index["anchors"][anchor_id] = {
            "d_model": d_model,
            "hub_dim": encoder.hub_dim,
            "path": str(anchor_dir),
        }
        self._save_index()
        logger.info(f"  Registered anchor: {anchor_id} (d={d_model})")

    def load(
        self,
        anchor_id: str,
        device: str = "cpu",
    ) -> Tuple[EncoderMLP, DecoderMLP]:
        """
        Load a registered anchor's encoder/decoder pair.

        Args:
            anchor_id: The anchor to load.
            device: Device to load weights onto.

        Returns:
            (encoder, decoder) tuple.

        Raises:
            KeyError: If anchor_id is not registered.
        """
        if anchor_id not in self._index["anchors"]:
            raise KeyError(
                f"Anchor '{anchor_id}' not found in registry. " f"Available: {self.list()}"
            )

        info = self._index["anchors"][anchor_id]
        anchor_dir = Path(info["path"])
        d_model = info["d_model"]
        hub_dim = info.get("hub_dim", 2048)

        encoder = EncoderMLP(d_model, hub_dim).to(device)
        decoder = DecoderMLP(d_model, hub_dim).to(device)

        encoder.load_state_dict(
            torch.load(anchor_dir / "encoder.pt", map_location=device, weights_only=True)
        )
        decoder.load_state_dict(
            torch.load(anchor_dir / "decoder.pt", map_location=device, weights_only=True)
        )

        return encoder, decoder

    def list(self) -> List[str]:
        """List all registered anchor IDs."""
        return list(self._index["anchors"].keys())

    def info(self, anchor_id: str) -> dict:
        """Get metadata for a registered anchor."""
        if anchor_id not in self._index["anchors"]:
            raise KeyError(f"Anchor '{anchor_id}' not found")
        return self._index["anchors"][anchor_id]

    def __contains__(self, anchor_id: str) -> bool:
        return anchor_id in self._index["anchors"]

    def __repr__(self) -> str:
        n = len(self._index["anchors"])
        return f"AnchorRegistry(root='{self.root}', anchors={n})"
