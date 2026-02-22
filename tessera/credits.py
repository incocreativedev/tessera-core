"""
tessera.credits — Utility scoring and credit ledger for swarm (v1).

Formula: utility = 0.35*quality + 0.25*novelty + 0.20*freshness + 0.20*reliability.
Per-contributor cap 15% of round weight; rolling 30-day credits.
See docs/swarm-roundtrip-architecture.md.
"""

import time
from typing import Dict, List, Any

from .token import TesseraToken

QUALITY_SIGNALS = "quality_signals"
UTILITY_SCORE = "utility_score"
AGGREGATION_WEIGHT = "aggregation_weight"


# Weights for utility (v1)
W_QUALITY = 0.35
W_NOVELTY = 0.25
W_FRESHNESS = 0.20
W_RELIABILITY = 0.20

# Freshness: full credit within 24h (seconds)
FRESHNESS_FULL_WINDOW = 24 * 3600


def _quality_component(signals: Dict[str, float]) -> float:
    """From inverse drift / reconstruction error; clamp to [0,1]."""
    drift = signals.get("drift", 1.0)
    recon_error = signals.get("recon_error", 1.0)
    # Lower drift and recon_error = higher quality
    q = 1.0 / (1.0 + float(drift) + float(recon_error))
    return max(0.0, min(1.0, q))


def _novelty_component(signals: Dict[str, float]) -> float:
    """Distance to prior round centroid; spec says novelty from distance."""
    n = signals.get("novelty", 0.5)
    if isinstance(n, (int, float)):
        return max(0.0, min(1.0, float(n)))
    return 0.5


def _freshness_component(token: TesseraToken) -> float:
    """Recency window decay; full credit <= 24h."""
    try:
        ts = token.timestamp
        if isinstance(ts, (int, float)):
            t = float(ts)
        else:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            t = dt.timestamp()
    except Exception:
        return 0.0
    now = time.time()
    age = now - t
    if age <= 0:
        return 1.0
    if age >= FRESHNESS_FULL_WINDOW:
        return 0.0
    return 1.0 - (age / FRESHNESS_FULL_WINDOW)


def _reliability_component(round_context: Dict[str, Any]) -> float:
    """Contributor historical acceptance ratio from round_context."""
    hist = round_context.get("contributor_history") or {}
    cid = round_context.get("contributor_id")
    if not cid or cid not in hist:
        return 0.5  # neutral if unknown
    h = hist[cid]
    accepted = h.get("accepted", 0)
    total = h.get("total", 1)
    if total <= 0:
        return 0.5
    return max(0.0, min(1.0, accepted / total))


def utility_score(token: TesseraToken, round_context: Dict[str, Any]) -> float:
    """
    utility = 0.35*quality + 0.25*novelty + 0.20*freshness + 0.20*reliability.
    """
    meta = token.custom_metadata or {}
    signals = meta.get(QUALITY_SIGNALS) or {}
    quality = _quality_component(signals)
    novelty = _novelty_component(signals)
    freshness = _freshness_component(token)
    ctx = dict(round_context)
    ctx.setdefault("contributor_id", meta.get("contributor_id"))
    reliability = _reliability_component(ctx)
    u = W_QUALITY * quality + W_NOVELTY * novelty + W_FRESHNESS * freshness + W_RELIABILITY * reliability
    return max(0.0, min(1.0, u))


def compute_credits(
    contributor_id: str,
    utility_scores: List[float],
    caps: Dict[str, float],
) -> float:
    """
    Daily credits for contributor = sum of accepted utility scores.
    caps can include max_fraction (per-contributor cap) and total_round_weight
    for normalizing; here we just sum and optionally cap by a max_credits_per_day.
    """
    total = sum(utility_scores)
    max_per_day = caps.get("max_credits_per_day")
    if max_per_day is not None and total > max_per_day:
        return float(max_per_day)
    return total


def rolling_30_day_credits(ledger: List[Dict[str, Any]]) -> float:
    """
    Sum credits from ledger entries within last 30 days.
    Each entry: { "contributor_id", "credits", "ts" } (ts in seconds).
    """
    from datetime import timedelta
    now = time.time()
    cutoff = now - timedelta(days=30).total_seconds()
    return sum(e.get("credits", 0) for e in ledger if e.get("ts", 0) >= cutoff)
