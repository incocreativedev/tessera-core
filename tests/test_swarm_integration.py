"""
Full-cycle swarm integration test per spec: submit → validate → score
→ aggregate → broadcast; verify lineage and version propagation.

Simulates multiple contributors (min 5 for policy), no actual model training.
"""

import os
import numpy as np
import pytest
from pathlib import Path

from tessera.token import TesseraToken, KnowledgeType
from tessera.swarm import (
    swarm_metadata,
    validate_for_swarm,
    score_token,
    aggregate_tokens,
    compute_credits,
    submit,
    aggregate,
    broadcast,
    score,
    SWARM_ROUND_ID,
    CONTRIBUTOR_ID,
    BROADCAST_VERSION,
    LINEAGE_PARENT_ROUNDS,
)
from tessera import policy
from tessera.policy import accept_token, check_round_acceptance, MIN_ACCEPTED_CONTRIBUTORS
from tessera.credits import CreditsLedger
from tessera.binary import TBFSerializer


def make_contributor_token(round_id: str, contributor_id: str, dim: int = 64) -> TesseraToken:
    """One contributor token with valid swarm metadata."""
    vec = np.random.randn(dim).astype(np.float32).tolist()
    meta = swarm_metadata(
        round_id=round_id,
        contributor_id=contributor_id,
        local_data_fingerprint=f"fp_{contributor_id}",
        quality_signals={"drift": 0.1, "recon_error": 0.05, "novelty": 0.6, "freshness": 1.0},
    )
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=vec,
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={},
        source_model_id=contributor_id,
        target_model_id=None,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        custom_metadata=meta,
    )


class TestFullCycleSwarm:
    """Full cycle: validate → score → aggregate → broadcast → lineage."""

    def test_full_cycle_submit_score_aggregate_broadcast(self, tmp_path):
        round_id = "round_001"
        # Need enough contributors so no one exceeds 15% cap (1/6 = 16.7%, 1/7 ≈ 14.3%)
        n_contributors = max(7, MIN_ACCEPTED_CONTRIBUTORS)
        contributor_ids = [f"site_{i}" for i in range(n_contributors)]

        # 1. Create contributor tokens and persist (simulate submit)
        token_paths = []
        tokens = []
        for cid in contributor_ids:
            t = make_contributor_token(round_id, cid, dim=64)
            tokens.append(t)
            path = tmp_path / f"token_{cid}.tbf"
            TBFSerializer.save(path, t)
            token_paths.append(str(path))

        # 2. Validate each (policy gate)
        accepted = []
        for t in tokens:
            ok, reason = accept_token(t)
            assert ok, reason
            ok2, _ = validate_for_swarm(t)
            assert ok2
            accepted.append(t)

        # 3. Score each and set aggregation weight
        round_context = {"round_id": round_id}
        scores = {}
        for t in accepted:
            u = score_token(t, round_context)
            cid = (t.custom_metadata or {}).get(CONTRIBUTOR_ID, t.source_model_id)
            scores[cid] = u
            t.custom_metadata["utility_score"] = u
            t.custom_metadata["aggregation_weight"] = max(0.0, u)

        # 4. Check round acceptance (min contributors, weight cap)
        ok_round, msg = check_round_acceptance(accepted)
        assert ok_round, msg

        # 5. Aggregate to hub vector
        hub_vec = aggregate_tokens(accepted, method="robust_weighted_mean")
        assert hub_vec.shape == (64,)
        assert np.isfinite(hub_vec).all()

        # 6. Broadcast token (central → contributors)
        broadcast_version = f"v-{round_id}"
        bt = broadcast(round_id, hub_vec, broadcast_version)
        assert bt.knowledge_type == KnowledgeType.SWARM
        assert bt.custom_metadata.get(SWARM_ROUND_ID) == round_id
        assert bt.custom_metadata.get(BROADCAST_VERSION) == broadcast_version
        assert round_id in bt.custom_metadata.get(LINEAGE_PARENT_ROUNDS, [])
        assert len(bt.uhs_vector) == 64

        # 7. Protocol aggregate() from paths returns same shape
        hub_from_paths = aggregate(round_id, token_paths)
        assert hub_from_paths is not None
        assert hub_from_paths.shape == (64,)

        # 8. score() from paths returns same contributor scores
        scores_from_paths = score(round_id, token_paths)
        assert set(scores_from_paths.keys()) == set(contributor_ids)
        for cid in contributor_ids:
            assert abs(scores_from_paths[cid] - scores[cid]) < 1e-6  # freshness uses time

    def test_credits_ledger_after_round(self):
        """Record credits for a round and verify rolling 30-day."""
        ledger = CreditsLedger()
        round_id = "round_002"
        contributor_ids = ["site_a", "site_b", "site_c", "site_d", "site_e"]

        for cid in contributor_ids:
            ledger.record_credit(
                contributor_id=cid,
                swarm_round_id=round_id,
                utility_score=0.7,
                quality_score=0.8,
                novelty_score=0.6,
                freshness_score=0.9,
                reliability_score=0.5,
            )

        total_a = ledger.get_contributor_credits("site_a")
        rolling_a = ledger.rolling_30_day_credits("site_a")
        assert total_a == 0.7
        assert rolling_a == 0.7

        # compute_credits (protocol) sums utility scores
        credits = compute_credits("site_a", [0.5, 0.3], {})
        assert credits == 0.8

    def test_submit_cli_path(self, tmp_path):
        """submit() validates and accepts a token file."""
        t = make_contributor_token("R1", "cli_test", dim=32)
        path = tmp_path / "sub.tbf"
        TBFSerializer.save(path, t)
        ok, msg = submit(str(path), "cli_test")
        assert ok is True
        assert "rejected" not in msg.lower() or "accepted" in msg.lower()
