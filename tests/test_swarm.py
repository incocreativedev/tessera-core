"""Tests for tessera.swarm — round-trip orchestration (submit, validate, aggregate, broadcast, score)."""

import os
import numpy as np
import pytest
from tessera.token import TesseraToken, KnowledgeType
from tessera.swarm import (
    swarm_metadata,
    validate_for_swarm,
    score_token,
    aggregate_tokens,
    submit,
    aggregate,
    broadcast,
    score,
    AggregationStrategy,
    SwarmAggregator,
    SWARM_ROUND_ID,
    CONTRIBUTOR_ID,
    QUALITY_SIGNALS,
    AGGREGATION_WEIGHT,
)
from tessera import policy
from tessera.binary import TBFSerializer


def make_swarm_token(round_id="R1", contributor_id="c1", dim=64):
    """Token with valid swarm custom_metadata for policy acceptance."""
    vec = np.random.randn(dim).astype(np.float32).tolist()
    meta = swarm_metadata(
        round_id=round_id,
        contributor_id=contributor_id,
        local_data_fingerprint="abc123",
        quality_signals={"drift": 0.1, "recon_error": 0.05, "novelty": 0.7, "freshness": 1.0},
    )
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=vec,
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={},
        source_model_id="contributor",
        target_model_id=None,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        custom_metadata=meta,
    )


class TestSwarmMetadata:
    def test_swarm_metadata_keys(self):
        meta = swarm_metadata("R1", "c1", "fp", quality_signals={"drift": 0.1})
        assert meta[SWARM_ROUND_ID] == "R1"
        assert meta[CONTRIBUTOR_ID] == "c1"
        assert meta["local_data_fingerprint"] == "fp"
        assert meta[QUALITY_SIGNALS]["drift"] == 0.1

    def test_validate_accepts_valid_token(self):
        token = make_swarm_token()
        accepted, reason = validate_for_swarm(token)
        assert accepted is True
        assert reason == "ok"

    def test_validate_rejects_missing_round_id(self):
        token = make_swarm_token()
        token.custom_metadata.pop(SWARM_ROUND_ID)
        accepted, reason = validate_for_swarm(token)
        assert accepted is False
        assert "swarm_round_id" in reason

    def test_validate_rejects_missing_contributor_id(self):
        token = make_swarm_token()
        token.custom_metadata.pop(CONTRIBUTOR_ID)
        accepted, reason = validate_for_swarm(token)
        assert accepted is False
        assert "contributor_id" in reason

    def test_validate_rejects_missing_fingerprint_key(self):
        token = make_swarm_token()
        token.custom_metadata.pop("local_data_fingerprint")
        accepted, reason = validate_for_swarm(token)
        assert accepted is False
        assert "local_data_fingerprint" in reason


class TestAggregation:
    def test_aggregate_single_token(self):
        token = make_swarm_token(dim=32)
        token.custom_metadata[AGGREGATION_WEIGHT] = 1.0
        vec = aggregate_tokens([token], method="robust_weighted_mean")
        assert vec.shape == (32,)
        np.testing.assert_allclose(vec, np.array(token.uhs_vector), atol=1e-5)

    def test_aggregate_multiple_tokens(self):
        tokens = [make_swarm_token(contributor_id=f"c{i}", dim=16) for i in range(5)]
        for t in tokens:
            t.custom_metadata[AGGREGATION_WEIGHT] = 1.0
        vec = aggregate_tokens(tokens, method="robust_weighted_mean")
        assert vec.shape == (16,)

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError, match="at least one token"):
            aggregate_tokens([], method="robust_weighted_mean")

    def test_aggregate_unknown_method_raises(self):
        token = make_swarm_token(dim=8)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_tokens([token], method="invalid")

    def test_aggregate_mean_l2_renorm(self):
        tokens = [make_swarm_token(contributor_id=f"c{i}", dim=16) for i in range(3)]
        for t in tokens:
            t.custom_metadata[AGGREGATION_WEIGHT] = 1.0
        vec = aggregate_tokens(tokens, method=AggregationStrategy.MEAN)
        assert vec.shape == (16,)
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)

    def test_aggregate_median_l2_renorm(self):
        tokens = [make_swarm_token(contributor_id=f"c{i}", dim=16) for i in range(3)]
        vec = aggregate_tokens(tokens, method="median")
        assert vec.shape == (16,)
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)

    def test_aggregate_weighted_l2_renorm(self):
        tokens = [make_swarm_token(contributor_id=f"c{i}", dim=16) for i in range(3)]
        for i, t in enumerate(tokens):
            t.custom_metadata[AGGREGATION_WEIGHT] = 0.5 + i * 0.25
        vec = aggregate_tokens(tokens, method="weighted")
        assert vec.shape == (16,)
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)


class TestBroadcast:
    def test_broadcast_token_structure(self):
        hub = np.random.randn(32).astype(np.float32)
        token = broadcast("R1", hub, "v1")
        assert token.source_model_id == "swarm_central"
        assert token.custom_metadata["broadcast_version"] == "v1"
        assert token.custom_metadata[SWARM_ROUND_ID] == "R1"
        assert len(token.uhs_vector) == 32


class TestScore:
    def test_score_returns_float(self):
        token = make_swarm_token()
        u = score_token(token, {"round_id": "R1"})
        assert isinstance(u, float)
        assert 0 <= u <= 1

    def test_score_deterministic_same_token(self):
        token = make_swarm_token()
        u1 = score_token(token, {"round_id": "R1"})
        u2 = score_token(token, {"round_id": "R1"})
        assert abs(u1 - u2) < 1e-9  # freshness uses time.time(); float order can differ


class TestSubmit:
    def test_submit_nonexistent_path(self):
        ok, msg = submit("/nonexistent/path.tbf", "c1")
        assert ok is False
        assert "not found" in msg

    def test_submit_valid_token_file(self, tmp_dir):
        token = make_swarm_token(dim=64)
        path = os.path.join(tmp_dir, "t.tbf")
        TBFSerializer.save(path, token)
        ok, msg = submit(path, "c1")
        assert ok is True


class TestAggregateFromPaths:
    def test_aggregate_below_min_returns_none(self, tmp_dir):
        token = make_swarm_token(dim=32)
        path = os.path.join(tmp_dir, "t.tbf")
        TBFSerializer.save(path, token)
        vec = aggregate("R1", [path])
        assert vec is None  # need MIN_ACCEPTED_CONTRIBUTORS (5)

    def test_aggregate_enough_tokens_returns_vector(self, tmp_dir):
        paths = []
        for i in range(policy.MIN_ACCEPTED_CONTRIBUTORS):
            t = make_swarm_token(contributor_id=f"c{i}", dim=32)
            t.custom_metadata[AGGREGATION_WEIGHT] = 1.0
            p = os.path.join(tmp_dir, f"t{i}.tbf")
            TBFSerializer.save(p, t)
            paths.append(p)
        vec = aggregate("R1", paths)
        assert vec is not None
        assert vec.shape == (32,)


class TestScoreFromPaths:
    def test_score_round(self, tmp_dir):
        for i in range(3):
            t = make_swarm_token(contributor_id=f"c{i}", round_id="R1", dim=32)
            TBFSerializer.save(os.path.join(tmp_dir, f"t{i}.tbf"), t)
        paths = [os.path.join(tmp_dir, f"t{i}.tbf") for i in range(3)]
        scores = score("R1", paths)
        assert len(scores) == 3
        for v in scores.values():
            assert isinstance(v, float)
            assert 0 <= v <= 1


class TestSwarmAggregator:
    def test_aggregate_delegates_to_aggregate_tokens(self):
        tokens = [make_swarm_token(contributor_id=f"c{i}", dim=32) for i in range(3)]
        for t in tokens:
            t.custom_metadata[AGGREGATION_WEIGHT] = 1.0

        class DummyModel:
            pass

        agg = SwarmAggregator(DummyModel(), hub_dim=32)
        vec = agg.aggregate_hub(tokens, method=AggregationStrategy.MEAN)
        expected = aggregate_tokens(tokens, method=AggregationStrategy.MEAN)
        np.testing.assert_allclose(vec, expected, atol=1e-5)
        assert vec.shape == (32,)
