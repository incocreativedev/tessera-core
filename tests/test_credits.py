"""
Tests for tessera.credits — utility scoring and credit ledger.
"""

import datetime
import pytest
import numpy as np

from tessera.credits import (
    UTILITY_WEIGHTS,
    CreditEntry,
    CreditsLedger,
    compute_quality_score,
    compute_novelty_score,
    compute_freshness_score,
    compute_reliability_score,
    compute_utility,
)


# ── Quality ─────────────────────────────────────────────────────────────


class TestQualityScore:
    def test_deterministic(self):
        assert compute_quality_score(0.1, 0.05) == compute_quality_score(0.1, 0.05)

    def test_in_range(self):
        score = compute_quality_score(0.5, 0.3)
        assert 0.0 <= score <= 1.0

    def test_inverse_relationship(self):
        high = compute_quality_score(0.0, 0.0)
        low = compute_quality_score(1.0, 1.0)
        assert high > low

    def test_zero_drift_and_error_is_one(self):
        assert compute_quality_score(0.0, 0.0) == 1.0


# ── Novelty ─────────────────────────────────────────────────────────────


class TestNoveltyScore:
    def test_no_prior_centroid_returns_one(self):
        hub = np.random.randn(64)
        assert compute_novelty_score(hub, None) == 1.0

    def test_identical_vectors_returns_zero(self):
        hub = np.ones(64)
        score = compute_novelty_score(hub, hub.copy())
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_returns_half(self):
        a = np.zeros(64)
        b = np.zeros(64)
        a[0] = 1.0
        b[1] = 1.0
        score = compute_novelty_score(a, b)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_in_range(self):
        hub = np.random.randn(64)
        centroid = np.random.randn(64)
        score = compute_novelty_score(hub, centroid)
        assert 0.0 <= score <= 1.0


# ── Freshness ───────────────────────────────────────────────────────────


class TestFreshnessScore:
    def test_recent_is_high(self):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        score = compute_freshness_score(now, window_hours=24)
        assert score > 0.9

    def test_old_is_zero(self):
        old = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)
        ).isoformat()
        score = compute_freshness_score(old, window_hours=24)
        assert score == 0.0

    def test_invalid_timestamp(self):
        assert compute_freshness_score("not-a-date") == 0.0

    def test_in_range(self):
        ts = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=6)
        ).isoformat()
        score = compute_freshness_score(ts, window_hours=24)
        assert 0.0 <= score <= 1.0


# ── Reliability ─────────────────────────────────────────────────────────


class TestReliabilityScore:
    def test_no_history_returns_half(self):
        assert compute_reliability_score("new_site", []) == 0.5

    def test_with_history_returns_one(self):
        entry = CreditEntry(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.8,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
            credits_awarded=0.8,
        )
        assert compute_reliability_score("site_a", [entry]) == 1.0

    def test_ignores_other_contributors(self):
        entry = CreditEntry(
            contributor_id="site_b",
            swarm_round_id="round_001",
            utility_score=0.8,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
            credits_awarded=0.8,
        )
        assert compute_reliability_score("site_a", [entry]) == 0.5


# ── Utility ─────────────────────────────────────────────────────────────


class TestComputeUtility:
    def test_deterministic(self):
        u1 = compute_utility(0.9, 0.7, 0.8, 1.0)
        u2 = compute_utility(0.9, 0.7, 0.8, 1.0)
        assert u1 == u2

    def test_in_range(self):
        u = compute_utility(0.5, 0.5, 0.5, 0.5)
        assert 0.0 <= u <= 1.0

    def test_all_ones_is_one(self):
        assert compute_utility(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros_is_zero(self):
        assert compute_utility(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_weighted_correctly(self):
        # Only quality = 1.0, rest zero → utility should be 0.35
        u = compute_utility(1.0, 0.0, 0.0, 0.0)
        assert u == pytest.approx(UTILITY_WEIGHTS["quality"])


# ── CreditEntry ─────────────────────────────────────────────────────────


class TestCreditEntry:
    def test_to_dict_roundtrip(self):
        entry = CreditEntry(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.82,
            quality_score=0.90,
            novelty_score=0.75,
            freshness_score=0.95,
            reliability_score=1.0,
            credits_awarded=0.82,
        )
        d = entry.to_dict()
        restored = CreditEntry.from_dict(d)
        assert restored.contributor_id == entry.contributor_id
        assert restored.credits_awarded == entry.credits_awarded


# ── CreditsLedger ───────────────────────────────────────────────────────


class TestCreditsLedger:
    def test_record_and_sum(self):
        ledger = CreditsLedger()
        ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.8,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
        )
        ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_002",
            utility_score=0.5,
            quality_score=0.6,
            novelty_score=0.4,
            freshness_score=0.5,
            reliability_score=1.0,
        )
        total = ledger.get_contributor_credits("site_a")
        assert total == pytest.approx(1.3)

    def test_rolling_30_day(self):
        ledger = CreditsLedger()
        _ = ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.8,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
        )
        rolling = ledger.rolling_30_day_credits("site_a")
        assert rolling == pytest.approx(0.8)

    def test_serialise_roundtrip(self):
        ledger = CreditsLedger()
        ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.8,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
        )
        data = ledger.to_list()
        restored = CreditsLedger.from_list(data)
        assert len(restored) == 1
        assert restored.get_contributor_credits("site_a") == pytest.approx(0.8)

    def test_max_credits_per_entry(self):
        ledger = CreditsLedger()
        entry = ledger.record_credit(
            contributor_id="site_a",
            swarm_round_id="round_001",
            utility_score=0.95,
            quality_score=0.9,
            novelty_score=0.7,
            freshness_score=0.95,
            reliability_score=1.0,
            max_credits_per_entry=0.5,
        )
        assert entry.credits_awarded == 0.5

    def test_len_and_repr(self):
        ledger = CreditsLedger()
        assert len(ledger) == 0
        assert "0" in repr(ledger)
