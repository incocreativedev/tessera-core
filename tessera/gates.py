"""
tessera.gates — Projection types for knowledge transformation.

Projections define categories of transformation applied during cross-architecture
knowledge transfer. Each projection type maps to a learned projection matrix.

Projection types (from the Tessera specification):
    H (Orthogonal)   — Fan-out: specialist → generalist
    CONDITIONAL      — Conditional: gated integration
    P (Scaling)      — Contextual re-weighting
    R (Reshape)      — Dimensionality adaptation (low-rank SVD)
    SWAP             — Bidirectional exchange via UHS
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class ProjectionType(Enum):
    """Projection types for knowledge transformation."""

    ORTHOGONAL = "H"  # Fan-out: dense 1-to-N projection
    CONDITIONAL = "CONDITIONAL"  # Conditional: gated by fidelity threshold τ
    SCALING = "P"  # Diagonal scaling (activation magnitude alignment)
    RESHAPE = "R"  # Low-rank SVD projection (dimensionality change)
    SWAP = "SWAP"  # Bidirectional exchange through UHS


@dataclass
class ProjectionHint:
    """
    Advisory hint about which projection(s) may improve transfer fidelity.

    Projection hints are included in Tessera tokens as non-binding suggestions.
    The receiver decides whether and how to apply them.
    """

    projection_type: ProjectionType
    strength: float  # 0.0–1.0, importance weight
    target_layers: List[str]  # Layer names this projection applies to

    def to_dict(self) -> dict:
        return {
            "projection": self.projection_type.value,
            "strength": self.strength,
            "target_layers": self.target_layers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectionHint":
        return cls(
            projection_type=ProjectionType(data["projection"]),
            strength=data["strength"],
            target_layers=data["target_layers"],
        )
