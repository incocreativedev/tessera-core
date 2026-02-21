"""
tessera.token — Self-describing Tessera token and serialisation.

A TesseraToken is the atomic unit of knowledge transfer. It carries:
    - The knowledge payload (as a UHS vector)
    - Activation-based metadata (modality weighting, correlation, gates)
    - Lineage (DAG of model ancestry)
    - Privacy parameters (ε, δ budget)
    - Fidelity estimate (drift score)
    - Authentication (source/target IDs, version, timestamp)

Tokens are serialised as SafeTensors (binary payload) + JSON (metadata).
"""

import json
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np


class KnowledgeType(Enum):
    """Type of knowledge encoded in this token."""

    ACTIVATION = "ACTIVATION"  # Mode A: activation fingerprints (primary)
    WEIGHT = "WEIGHT"  # Mode W: raw parameters (same-anchor only)
    DATASET = "DATASET"  # Mode D: training data subsets + curriculum
    BEHAVIOUR = "BEHAVIOUR"  # Mode B: decision boundaries, policies, CoT


@dataclass
class TesseraToken:
    """
    Self-describing token carrying knowledge for transfer.

    Every field maps to the Tessera v1.0 specification token structure.
    """

    # --- Core payload ---
    knowledge_type: KnowledgeType
    uhs_vector: List[float]  # 2048-dim hub-space vector

    # --- Activation-based fields ---
    modality_weights: Dict[str, float]  # {basis: amplitude} e.g. {"A": 0.85, "W": 0.12, "B": 0.03}
    correlation_map: Dict[str, float]  # {token_id: mutual_information}

    # --- Lineage ---
    lineage_dag: dict  # DAG of model ancestry
    generation: int = 1  # Transfer generation count

    # --- Gates ---
    projection_hints: List[dict] = field(default_factory=list)

    # --- Privacy & fidelity ---
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    drift_score: float = 0.0  # KL-divergence (lower = better)

    # --- Provenance ---
    source_model_id: str = ""
    target_model_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    version: str = "1.0"

    # --- Extensible metadata ---
    custom_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "knowledge_type": self.knowledge_type.value,
            "uhs_vector": self.uhs_vector,
            "modality_weights": self.modality_weights,
            "correlation_map": self.correlation_map,
            "lineage_dag": self.lineage_dag,
            "generation": self.generation,
            "projection_hints": self.projection_hints,
            "privacy_epsilon": self.privacy_epsilon,
            "privacy_delta": self.privacy_delta,
            "drift_score": self.drift_score,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TesseraToken":
        """Reconstruct a token from a dictionary."""
        d = data.copy()
        d["knowledge_type"] = KnowledgeType(d["knowledge_type"])
        return cls(**d)


class TokenSerializer:
    """
    Handles token persistence using SafeTensors (binary) + JSON (metadata).

    File layout for a token saved as "transfer.safetensors":
        transfer.safetensors  — UHS vector as float32 tensor
        transfer.json         — Full token metadata
    """

    @staticmethod
    def save_token(token: TesseraToken, filepath: str):
        """
        Save a token to disk.

        Args:
            filepath: Path ending in .safetensors (JSON sidecar auto-named).
        """
        import torch
        from safetensors.torch import save_file

        filepath = Path(filepath)

        # Binary payload
        uhs_tensor = torch.tensor(token.uhs_vector, dtype=torch.float32)
        save_file({"uhs_vector": uhs_tensor}, str(filepath))

        # Metadata sidecar
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(token.to_dict(), f, indent=2)

    @staticmethod
    def load_token(filepath: str) -> TesseraToken:
        """
        Load a token from disk.

        Args:
            filepath: Path to the .safetensors file.

        Returns:
            Reconstructed TesseraToken.
        """
        from safetensors.torch import load_file

        filepath = Path(filepath)

        # Load binary
        tensors = load_file(str(filepath))
        uhs_vector = tensors["uhs_vector"].tolist()

        # Load metadata
        meta_path = filepath.with_suffix(".json")
        with open(meta_path) as f:
            data = json.load(f)

        data["uhs_vector"] = uhs_vector
        return TesseraToken.from_dict(data)
