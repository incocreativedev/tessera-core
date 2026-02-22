"""Tests for tessera.swarm — Swarm aggregation (many-to-one)."""

import math
import torch
import pytest
from tessera.swarm import (
    SwarmAggregator,
    AggregationStrategy,
    swarm_metadata,
    validate_for_swarm,
)
from tessera.transfer import ModeATransfer
from tessera.token import TesseraToken, KnowledgeType, TokenSerializer
from tests.conftest import SmallTransformer


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def contributor_tokens(train_loader, val_loader):
    """
    Create 3 realistic contributor tokens by running ModeATransfer
    on 3 different small models. Each token has a real 2048-dim UHS vector.
    """
    tokens = []
    for i in range(3):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)

        # Lightly train TX
        opt = torch.optim.Adam(tx.parameters(), lr=1e-3)
        tx.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(tx(bx), by)
            loss.backward()
            opt.step()
        tx.eval()

        transfer = ModeATransfer(tx, rx, f"site_{i}", f"rx_{i}")
        token = transfer.execute(
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )
        tokens.append(token)
    return tokens


@pytest.fixture
def large_model():
    """A larger aggregator model."""
    return SmallTransformer(d_model=128, num_layers=3)


@pytest.fixture
def same_size_model():
    """Aggregator with same d_model as contributors."""
    return SmallTransformer(d_model=64, num_layers=2)


# ── Aggregation tests ────────────────────────────────────────────────────


class TestAggregation:
    def test_aggregate_returns_swarm_token(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """aggregate() returns a single SWARM token."""
        agg = SwarmAggregator(same_size_model, "central")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert isinstance(token, TesseraToken)
        assert token.knowledge_type == KnowledgeType.SWARM
        assert len(token.uhs_vector) == 2048
        assert token.target_model_id == "central"

    def test_mean_strategy(self, contributor_tokens, same_size_model, train_loader, val_loader):
        """MEAN strategy produces valid token."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.MEAN,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["aggregation_method"] == "mean"

    def test_weighted_mean_strategy(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """WEIGHTED_MEAN with auto-weights (inverse drift)."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.WEIGHTED_MEAN,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["aggregation_method"] == "weighted_mean"
        assert len(token.custom_metadata["contributor_weights"]) == 3
        # Weights should sum to ~1.0
        w_sum = sum(token.custom_metadata["contributor_weights"])
        assert abs(w_sum - 1.0) < 1e-4

    def test_trimmed_mean_strategy(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """TRIMMED_MEAN strategy."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.TRIMMED_MEAN,
            trimmed_fraction=0.1,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["aggregation_method"] == "trimmed_mean"

    def test_median_strategy(self, contributor_tokens, same_size_model, train_loader, val_loader):
        """MEDIAN strategy."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.MEDIAN,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["aggregation_method"] == "median"

    def test_metadata_includes_contributor_info(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Aggregated token metadata has contributor IDs, count, drifts."""
        agg = SwarmAggregator(same_size_model, "central")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        meta = token.custom_metadata
        assert meta["contributor_count"] == 3
        assert len(meta["contributor_ids"]) == 3
        assert len(meta["contributor_drift_scores"]) == 3
        assert meta["swarm_mode"] == "aggregate"

    def test_custom_weights(self, contributor_tokens, same_size_model, train_loader, val_loader):
        """Custom weights are used and stored in metadata."""
        agg = SwarmAggregator(same_size_model, "c")
        custom_w = [0.5, 0.3, 0.2]
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.WEIGHTED_MEAN,
            weights=custom_w,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["contributor_weights"] == custom_w

    def test_cross_architecture_aggregation(
        self, contributor_tokens, large_model, train_loader, val_loader
    ):
        """Aggregator with different d_model than contributors works."""
        agg = SwarmAggregator(large_model, "big_central")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert isinstance(token, TesseraToken)
        assert token.custom_metadata["aggregator_d_model"] == 128


# ── Broadcast tests ──────────────────────────────────────────────────────


class TestBroadcast:
    def test_returns_pair(self, contributor_tokens, same_size_model, train_loader, val_loader):
        """aggregate_and_broadcast returns (token, list)."""
        agg = SwarmAggregator(same_size_model, "central")
        agg_token, bcast = agg.aggregate_and_broadcast(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert isinstance(agg_token, TesseraToken)
        assert isinstance(bcast, list)
        assert len(bcast) == 3

    def test_broadcast_targets_contributors(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Each broadcast token targets its specific contributor."""
        agg = SwarmAggregator(same_size_model, "central")
        _, bcast = agg.aggregate_and_broadcast(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        expected_ids = {t.source_model_id for t in contributor_tokens}
        actual_ids = {bt.target_model_id for bt in bcast}
        assert actual_ids == expected_ids

    def test_broadcast_source_is_aggregator(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """All broadcast tokens have source = aggregator_id."""
        agg = SwarmAggregator(same_size_model, "central")
        _, bcast = agg.aggregate_and_broadcast(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        for bt in bcast:
            assert bt.source_model_id == "central"
            assert bt.custom_metadata["swarm_mode"] == "broadcast"

    def test_broadcast_tokens_have_valid_hub(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Broadcast tokens carry 2048-dim UHS vectors."""
        agg = SwarmAggregator(same_size_model, "central")
        _, bcast = agg.aggregate_and_broadcast(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        for bt in bcast:
            assert len(bt.uhs_vector) == 2048
            assert bt.knowledge_type == KnowledgeType.SWARM


# ── Privacy composition tests ────────────────────────────────────────────


class TestPrivacyComposition:
    def test_composed_epsilon_formula(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """ε_total = √N × max(εᵢ)."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        n = len(contributor_tokens)
        max_eps = max(t.privacy_epsilon for t in contributor_tokens)
        expected_eps = math.sqrt(n) * max_eps

        assert abs(token.custom_metadata["composed_epsilon"] - expected_eps) < 1e-6

    def test_composed_delta_sums(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """δ_total = Σ δᵢ."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        expected_delta = sum(t.privacy_delta for t in contributor_tokens)
        assert abs(token.custom_metadata["composed_delta"] - expected_delta) < 1e-10


# ── Validation tests ─────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_empty_list(self, same_size_model, train_loader, val_loader):
        """Raises ValueError for < 2 tokens."""
        agg = SwarmAggregator(same_size_model, "c")
        with pytest.raises(ValueError, match="at least 2"):
            agg.aggregate([], train_loader, val_loader)

    def test_rejects_single_token(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Raises ValueError for only 1 token."""
        agg = SwarmAggregator(same_size_model, "c")
        with pytest.raises(ValueError, match="at least 2"):
            agg.aggregate([contributor_tokens[0]], train_loader, val_loader)

    def test_rejects_wrong_hub_dim(self, same_size_model, train_loader, val_loader):
        """Raises ValueError if hub vector != 2048-dim."""
        bad_token = TesseraToken(
            knowledge_type=KnowledgeType.ACTIVATION,
            uhs_vector=[0.0] * 512,  # wrong dim
            modality_weights={"A": 1.0},
            correlation_map={},
            lineage_dag={"nodes": [], "root": "x"},
            source_model_id="bad",
        )
        ok_token = TesseraToken(
            knowledge_type=KnowledgeType.ACTIVATION,
            uhs_vector=[0.0] * 2048,
            modality_weights={"A": 1.0},
            correlation_map={},
            lineage_dag={"nodes": [], "root": "x"},
            source_model_id="ok",
        )

        agg = SwarmAggregator(same_size_model, "c")
        with pytest.raises(ValueError, match="dimension"):
            agg.aggregate([bad_token, ok_token], train_loader, val_loader)


# ── Edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_two_contributors(self, contributor_tokens, same_size_model, train_loader, val_loader):
        """Works with minimum (2) contributors."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens[:2],
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["contributor_count"] == 2

    def test_drift_non_negative(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Drift score is non-negative."""
        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.drift_score >= 0

    def test_lineage_dag_has_all_contributors(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """Lineage DAG includes all contributor nodes + aggregator."""
        agg = SwarmAggregator(same_size_model, "central")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        dag = token.lineage_dag
        assert dag["root"] == "agg"
        # 3 contributors + 1 aggregator
        assert len(dag["nodes"]) == 4

    def test_serialise_aggregated_token(
        self, contributor_tokens, same_size_model, train_loader, val_loader, tmp_dir
    ):
        """Aggregated token can be saved and loaded."""
        import os

        agg = SwarmAggregator(same_size_model, "c")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        path = os.path.join(tmp_dir, "swarm.safetensors")
        TokenSerializer.save_token(token, path)
        loaded = TokenSerializer.load_token(path)

        assert loaded.knowledge_type == KnowledgeType.SWARM
        assert loaded.custom_metadata["contributor_count"] == 3
        assert len(loaded.uhs_vector) == 2048


# ── Robust weighted mean tests ──────────────────────────────────────────


class TestRobustWeightedMean:
    def test_robust_produces_valid_token(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """ROBUST_WEIGHTED_MEAN strategy produces a valid SWARM token."""
        agg = SwarmAggregator(same_size_model, "central")
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.ROBUST_WEIGHTED_MEAN,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.knowledge_type == KnowledgeType.SWARM
        assert len(token.uhs_vector) == 2048
        assert token.custom_metadata["aggregation_method"] == "robust_weighted_mean"

    def test_robust_with_custom_weights(
        self, contributor_tokens, same_size_model, train_loader, val_loader
    ):
        """ROBUST_WEIGHTED_MEAN works with explicit weights."""
        agg = SwarmAggregator(same_size_model, "central")
        n = len(contributor_tokens)
        weights = [1.0 / n] * n
        token = agg.aggregate(
            contributor_tokens,
            train_loader,
            val_loader,
            aggregation_strategy=AggregationStrategy.ROBUST_WEIGHTED_MEAN,
            weights=weights,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert token.custom_metadata["contributor_count"] == n


# ── Swarm metadata helper tests ─────────────────────────────────────────


class TestSwarmMetadataHelper:
    def test_required_fields(self):
        """swarm_metadata() returns the three mandatory fields."""
        meta = swarm_metadata("round_042", "site_a", "sha256:abc")
        assert meta["swarm_round_id"] == "round_042"
        assert meta["contributor_id"] == "site_a"
        assert meta["local_data_fingerprint"] == "sha256:abc"

    def test_extra_signals(self):
        """Extra signals are merged into the metadata dict."""
        meta = swarm_metadata(
            "round_042",
            "site_a",
            "sha256:abc",
            extra_signals={"quality_hint": 0.9, "region": "us-east"},
        )
        assert meta["quality_hint"] == 0.9
        assert meta["region"] == "us-east"

    def test_no_extra_signals(self):
        """Without extra_signals, dict has exactly 3 keys."""
        meta = swarm_metadata("round_042", "site_a", "sha256:abc")
        assert len(meta) == 3


# ── validate_for_swarm tests ────────────────────────────────────────────


class TestValidateForSwarm:
    def _make_valid_token(self) -> TesseraToken:
        return TesseraToken(
            knowledge_type=KnowledgeType.SWARM,
            uhs_vector=[0.1] * 2048,
            modality_weights={"A": 1.0},
            correlation_map={},
            lineage_dag={"nodes": [], "root": "x"},
            source_model_id="site_a",
            custom_metadata=swarm_metadata("round_001", "site_a", "sha256:abc"),
        )

    def test_valid_token_passes(self):
        token = self._make_valid_token()
        ok, reason = validate_for_swarm(token)
        assert ok is True
        assert reason == "ok"

    def test_missing_round_id_fails(self):
        token = self._make_valid_token()
        del token.custom_metadata["swarm_round_id"]
        ok, reason = validate_for_swarm(token)
        assert ok is False
        assert "swarm_round_id" in reason

    def test_missing_contributor_id_fails(self):
        token = self._make_valid_token()
        del token.custom_metadata["contributor_id"]
        ok, reason = validate_for_swarm(token)
        assert ok is False
        assert "contributor_id" in reason

    def test_missing_fingerprint_fails(self):
        token = self._make_valid_token()
        del token.custom_metadata["local_data_fingerprint"]
        ok, reason = validate_for_swarm(token)
        assert ok is False
        assert "local_data_fingerprint" in reason

    def test_empty_uhs_fails(self):
        token = self._make_valid_token()
        token.uhs_vector = []
        ok, reason = validate_for_swarm(token)
        assert ok is False
        assert "UHS" in reason
