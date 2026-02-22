"""
tessera.policy — Governance and acceptance rules for swarm rounds.

Defines the policy layer that gates which tokens are accepted into a
swarm aggregation round and validates round-level constraints (minimum
contributors, per-contributor weight caps).

v1 defaults (Ag/Mining pilot):
    - Minimum 5 unique contributors per round
    - No single contributor may exceed 15% of total round weight
    - All tokens must carry swarm metadata (round ID, contributor ID,
      local data fingerprint)
    - Valid privacy fields (ε > 0, 0 < δ < 1)
    - Non-empty UHS vector
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .token import TesseraToken

# ── v1 constants ──────────────────────────────────────────────────────────

MIN_ACCEPTED_CONTRIBUTORS: int = 5
"""Minimum unique contributors required to run an aggregation round."""

MAX_CONTRIBUTOR_WEIGHT_FRACTION: float = 0.15
"""Maximum fraction of round weight any single contributor may hold."""


# ── Token-level acceptance ────────────────────────────────────────────────

def accept_token(token: TesseraToken) -> Tuple[bool, str]:
    """
    Validate a single token for swarm round acceptance.

    Checks (in order):
        1. custom_metadata contains 'swarm_round_id' (non-empty str)
        2. custom_metadata contains 'contributor_id' (non-empty str)
        3. custom_metadata contains 'local_data_fingerprint' (non-empty str)
        4. privacy_epsilon > 0
        5. privacy_delta in (0, 1)
        6. uhs_vector is non-empty

    Returns:
        (True, "") if accepted, (False, reason) if rejected.
    """
    meta = token.custom_metadata or {}

    # Required metadata fields
    if not meta.get("swarm_round_id"):
        return False, "Missing or empty 'swarm_round_id' in custom_metadata."

    if not meta.get("contributor_id"):
        return False, "Missing or empty 'contributor_id' in custom_metadata."

    if not meta.get("local_data_fingerprint"):
        return False, "Missing or empty 'local_data_fingerprint' in custom_metadata."

    # Privacy fields
    if token.privacy_epsilon <= 0:
        return False, (
            f"Invalid privacy_epsilon={token.privacy_epsilon}; must be > 0."
        )

    if not (0 < token.privacy_delta < 1):
        return False, (
            f"Invalid privacy_delta={token.privacy_delta}; must be in (0, 1)."
        )

    # UHS vector
    if not token.uhs_vector or len(token.uhs_vector) == 0:
        return False, "UHS vector is empty."

    return True, ""


# ── Round-level acceptance ────────────────────────────────────────────────

def check_round_acceptance(
    tokens: List[TesseraToken],
    min_contributors: int = MIN_ACCEPTED_CONTRIBUTORS,
    max_weight_fraction: float = MAX_CONTRIBUTOR_WEIGHT_FRACTION,
) -> Tuple[bool, str]:
    """
    Validate a collection of tokens for a complete swarm round.

    Checks:
        1. Every token passes accept_token() individually.
        2. At least min_contributors unique contributor_ids.
        3. No single contributor holds > max_weight_fraction of total tokens.

    Returns:
        (True, "") if round is valid, (False, reason) if not.
    """
    if not tokens:
        return False, "No tokens provided."

    # Individual validation
    for i, t in enumerate(tokens):
        ok, reason = accept_token(t)
        if not ok:
            return False, f"Token {i} rejected: {reason}"

    # Count unique contributors
    contributor_ids = [
        t.custom_metadata.get("contributor_id", "") for t in tokens
    ]
    unique_contributors = set(contributor_ids)

    if len(unique_contributors) < min_contributors:
        return False, (
            f"Only {len(unique_contributors)} unique contributors; "
            f"minimum is {min_contributors}."
        )

    # Weight cap: each contributor's token count / total tokens
    n_total = len(tokens)
    from collections import Counter
    counts = Counter(contributor_ids)
    for cid, count in counts.items():
        fraction = count / n_total
        if fraction > max_weight_fraction:
            return False, (
                f"Contributor '{cid}' holds {fraction:.1%} of round weight "
                f"({count}/{n_total} tokens); cap is {max_weight_fraction:.0%}."
            )

    return True, ""


# ── Round policy encapsulation ────────────────────────────────────────────

@dataclass
class RoundPolicy:
    """
    Encapsulates governance rules for a single swarm round.

    Can be customised per-round to override defaults (e.g. for pilot
    rounds with fewer contributors or stricter weight caps).

    Usage:
        policy = RoundPolicy(round_id="round_042", min_contributors=3)
        ok, reason = policy.validate_round(tokens)
    """

    round_id: str
    min_contributors: int = MIN_ACCEPTED_CONTRIBUTORS
    max_weight_per_contributor: float = MAX_CONTRIBUTOR_WEIGHT_FRACTION
    allowed_contributors: Optional[Set[str]] = field(default=None)

    def can_accept_token(self, token: TesseraToken) -> Tuple[bool, str]:
        """Check if a single token is acceptable under this policy."""
        ok, reason = accept_token(token)
        if not ok:
            return ok, reason

        # Optional allowlist
        if self.allowed_contributors is not None:
            cid = token.custom_metadata.get("contributor_id", "")
            if cid not in self.allowed_contributors:
                return False, (
                    f"Contributor '{cid}' not in allowed set for "
                    f"round {self.round_id}."
                )

        return True, ""

    def validate_round(
        self, tokens: List[TesseraToken]
    ) -> Tuple[bool, str]:
        """Validate a full round against this policy."""
        return check_round_acceptance(
            tokens,
            min_contributors=self.min_contributors,
            max_weight_fraction=self.max_weight_per_contributor,
        )
