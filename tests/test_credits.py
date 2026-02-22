"""Tests for tessera.credits — utility scoring and credit assignment."""

import numpy as np
from tessera.token import TesseraToken, KnowledgeType
from tessera.swarm import swarm_metadata
from tessera import credits


def make_token_for_credits(quality_signals=None):
    """Token with swarm metadata for credit scoring."""
    if quality_signals is None:
        quality_signals = {"drift": 0.1, "recon_error": 0.05, "novelty": 0.6, "freshness": 1.0}
    meta = swarm_metadata("R1", "c1", "fp", quality_signals)
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=np.random.randn(64).astype(np.float32).tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={},
        source_model_id="contributor",
        target_model_id=None,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        custom_metadata=meta,
    )


class TestUtilityScore:
    def test_utility_in_bounds(self):
        token = make_token_for_credits()
        u = credits.utility_score(token, {"round_id": "R1"})
        assert 0 <= u <= 1

    def test_utility_deterministic(self):
        token = make_token_for_credits()
        u1 = credits.utility_score(token, {"round_id": "R1"})
        u2 = credits.utility_score(token, {"round_id": "R1"})
        assert abs(u1 - u2) < 1e-9  # freshness uses time.time(); float order can differ

    def test_quality_lower_drift_higher_utility(self):
        low_drift = make_token_for_credits({"drift": 0.01, "recon_error": 0.01, "novelty": 0.5, "freshness": 1.0})
        high_drift = make_token_for_credits({"drift": 2.0, "recon_error": 2.0, "novelty": 0.5, "freshness": 1.0})
        u_low = credits.utility_score(low_drift, {})
        u_high = credits.utility_score(high_drift, {})
        assert u_low > u_high

    def test_reliability_from_history(self):
        token = make_token_for_credits()
        ctx = {"contributor_id": "c1", "contributor_history": {"c1": {"accepted": 8, "total": 10}}}
        u = credits.utility_score(token, ctx)
        assert u >= 0 and u <= 1


class TestComputeCredits:
    def test_sum_utility_scores(self):
        c = credits.compute_credits("c1", [0.5, 0.3, 0.2], {})
        assert c == 1.0

    def test_cap_max_credits_per_day(self):
        c = credits.compute_credits("c1", [0.5, 0.5, 0.5], {"max_credits_per_day": 1.0})
        assert c == 1.0

    def test_no_cap(self):
        c = credits.compute_credits("c1", [0.2, 0.3], {})
        assert c == 0.5


class TestRolling30DayCredits:
    def test_empty_ledger(self):
        assert credits.rolling_30_day_credits([]) == 0

    def test_sum_within_window(self):
        import time
        now = time.time()
        ledger = [
            {"contributor_id": "c1", "credits": 1.0, "ts": now - 10},
            {"contributor_id": "c1", "credits": 2.0, "ts": now - 86400},
        ]
        assert credits.rolling_30_day_credits(ledger) == 3.0

    def test_old_entries_excluded(self):
        import time
        from datetime import timedelta
        now = time.time()
        cutoff = now - timedelta(days=31).total_seconds()
        ledger = [
            {"contributor_id": "c1", "credits": 100.0, "ts": cutoff},
        ]
        assert credits.rolling_30_day_credits(ledger) == 0
