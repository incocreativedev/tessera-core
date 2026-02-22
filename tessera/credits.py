"""
tessera.credits — Utility scoring and credit ledger for swarm contributors.

Implements quality-weighted credits per the Codex v1 spec:
    utility = 0.35*quality + 0.25*novelty + 0.20*freshness + 0.20*reliability

Contributors earn credits proportional to their utility scores. Credits
map to free-usage tiers (rolling 30-day window).

Abuse guard: per-contributor max 15% of total round weight (enforced by
policy.py, not here).
"""

import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── v1 utility weights ───────────────────────────────────────────────────

UTILITY_WEIGHTS: Dict[str, float] = {
    "quality": 0.35,
    "novelty": 0.25,
    "freshness": 0.20,
    "reliability": 0.20,
}
"""Default weights for the four utility components."""


# ── Credit ledger entry ──────────────────────────────────────────────────


@dataclass
class CreditEntry:
    """Single credit ledger record for one contributor in one round."""

    contributor_id: str
    swarm_round_id: str
    utility_score: float
    quality_score: float
    novelty_score: float
    freshness_score: float
    reliability_score: float
    credits_awarded: float
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "contributor_id": self.contributor_id,
            "swarm_round_id": self.swarm_round_id,
            "utility_score": self.utility_score,
            "quality_score": self.quality_score,
            "novelty_score": self.novelty_score,
            "freshness_score": self.freshness_score,
            "reliability_score": self.reliability_score,
            "credits_awarded": self.credits_awarded,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CreditEntry":
        """Reconstruct from a dictionary."""
        return cls(**data)


# ── Scoring functions ────────────────────────────────────────────────────


def compute_quality_score(drift: float, recon_error: float = 0.0) -> float:
    """
    Quality from inverse drift and reconstruction error.

    quality = 1.0 / (1.0 + drift + recon_error)

    Lower drift/error → higher quality. Returns float in [0, 1].
    """
    return 1.0 / (1.0 + abs(drift) + abs(recon_error))


def compute_novelty_score(
    token_hub: np.ndarray,
    prior_centroid: Optional[np.ndarray],
) -> float:
    """
    Novelty from cosine distance to prior round centroid.

    If no prior centroid exists, returns 1.0 (full novelty — first round).
    Returns float in [0, 1].
    """
    if prior_centroid is None:
        return 1.0

    token_hub = np.asarray(token_hub, dtype=np.float64)
    prior_centroid = np.asarray(prior_centroid, dtype=np.float64)

    norm_t = np.linalg.norm(token_hub)
    norm_c = np.linalg.norm(prior_centroid)

    if norm_t < 1e-10 or norm_c < 1e-10:
        return 1.0

    cosine_sim = float(np.dot(token_hub, prior_centroid) / (norm_t * norm_c))
    # Clamp to [-1, 1] for numerical safety
    cosine_sim = max(-1.0, min(1.0, cosine_sim))

    # Cosine distance in [0, 1]: 0 = identical, 1 = orthogonal
    return (1.0 - cosine_sim) / 2.0


def compute_freshness_score(
    token_timestamp: str,
    window_hours: int = 24,
) -> float:
    """
    Freshness from recency within a time window.

    freshness = exp(-age_hours / window_hours) if age < window_hours else 0.0

    Full credit for very recent tokens; exponential decay over the window.
    Returns float in [0, 1].
    """
    try:
        ts = datetime.datetime.fromisoformat(token_timestamp)
    except (ValueError, TypeError):
        return 0.0

    # Ensure timezone-aware
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)

    now = datetime.datetime.now(datetime.timezone.utc)
    age_hours = (now - ts).total_seconds() / 3600.0

    if age_hours < 0:
        # Future timestamp — treat as fresh
        return 1.0

    if age_hours > window_hours:
        return 0.0

    return math.exp(-age_hours / window_hours)


def compute_reliability_score(
    contributor_id: str,
    ledger: list,
) -> float:
    """
    Reliability from contributor's historical acceptance ratio.

    reliability = accepted_rounds / total_rounds_participated

    If contributor has no history, returns 0.5 (neutral prior).
    Accepts either a list of CreditEntry objects or a list of dicts.
    Returns float in [0, 1].
    """
    entries = []
    for entry in ledger:
        if isinstance(entry, CreditEntry):
            if entry.contributor_id == contributor_id:
                entries.append(entry)
        elif isinstance(entry, dict):
            if entry.get("contributor_id") == contributor_id:
                entries.append(entry)

    if not entries:
        return 0.5  # Neutral prior for new contributors

    # All entries in the ledger are "accepted" (you only get a credit entry
    # if your token was accepted). So reliability = 1.0 if all rounds
    # resulted in credit. For a more nuanced version, compare against
    # total submission attempts — but we only track accepted here.
    return 1.0


def compute_utility(
    quality: float,
    novelty: float,
    freshness: float,
    reliability: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Composite utility score: weighted sum of four components.

    Default: 0.35*quality + 0.25*novelty + 0.20*freshness + 0.20*reliability

    All inputs and output in [0, 1].
    """
    w = weights or UTILITY_WEIGHTS
    utility = (
        w["quality"] * quality
        + w["novelty"] * novelty
        + w["freshness"] * freshness
        + w["reliability"] * reliability
    )
    return max(0.0, min(1.0, utility))


# ── Credits ledger ───────────────────────────────────────────────────────


class CreditsLedger:
    """
    Append-only ledger of contributor credits and utility.

    Supports:
        - Recording credit entries
        - Querying total credits per contributor
        - Rolling 30-day credit windows (for free-usage tiers)
        - JSON serialisation/deserialisation

    Usage:
        ledger = CreditsLedger()
        entry = ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_042",
            utility_score=0.82,
            quality_score=0.90,
            novelty_score=0.75,
            freshness_score=0.95,
            reliability_score=1.0,
        )
        print(ledger.rolling_30_day_credits("site_a"))
    """

    def __init__(self):
        self.entries: List[CreditEntry] = []

    def record_credit(
        self,
        contributor_id: str,
        swarm_round_id: str,
        utility_score: float,
        quality_score: float,
        novelty_score: float,
        freshness_score: float,
        reliability_score: float,
        max_credits_per_entry: Optional[float] = None,
    ) -> CreditEntry:
        """
        Create and store a credit entry.

        credits_awarded = utility_score (capped if max_credits_per_entry set).
        """
        credits = utility_score
        if max_credits_per_entry is not None:
            credits = min(credits, max_credits_per_entry)

        entry = CreditEntry(
            contributor_id=contributor_id,
            swarm_round_id=swarm_round_id,
            utility_score=utility_score,
            quality_score=quality_score,
            novelty_score=novelty_score,
            freshness_score=freshness_score,
            reliability_score=reliability_score,
            credits_awarded=credits,
        )
        self.entries.append(entry)
        return entry

    def get_contributor_credits(self, contributor_id: str) -> float:
        """Sum of all credits ever awarded to this contributor."""
        return sum(e.credits_awarded for e in self.entries if e.contributor_id == contributor_id)

    def rolling_30_day_credits(self, contributor_id: str) -> float:
        """Sum of credits from entries in the last 30 days."""
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

        total = 0.0
        for e in self.entries:
            if e.contributor_id != contributor_id:
                continue

            try:
                ts = datetime.datetime.fromisoformat(e.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                if ts >= cutoff:
                    total += e.credits_awarded
            except (ValueError, TypeError):
                continue

        return total

    def to_list(self) -> List[dict]:
        """Serialise ledger to list of dicts."""
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, data: List[dict]) -> "CreditsLedger":
        """Deserialise ledger from list of dicts."""
        ledger = cls()
        ledger.entries = [CreditEntry.from_dict(d) for d in data]
        return ledger

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"CreditsLedger(entries={len(self.entries)})"
