"""End-to-end tests for tessera.mode_w — Mode W weight transfer."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tessera.mode_w import ModeWTransfer
from tessera.token import KnowledgeType, TesseraToken
from tessera.weight_ops import extract_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_loader():
    """Small reference dataloader (same format as train_loader in conftest)."""
    X = torch.randint(0, 128, (40, 8))
    y = (X.float().mean(dim=1) >= 64).long()
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


# ---------------------------------------------------------------------------
# Basic execute tests
# ---------------------------------------------------------------------------


class TestModeWTransferBasic:
    def test_execute_returns_tessera_token(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, "tx_model", "rx_model", hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert isinstance(token, TesseraToken)

    def test_token_knowledge_type_is_weight(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.knowledge_type == KnowledgeType.WEIGHT

    def test_modality_weights_w_dominant(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.modality_weights["W"] >= 0.85

    def test_uhs_vector_correct_dim(self, small_tx, small_rx, ref_loader):
        hub_dim = 64
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=hub_dim)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert len(token.uhs_vector) == hub_dim

    def test_source_and_target_ids(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(
            small_tx, small_rx, transmitter_id="alpha", receiver_id="beta", hub_dim=64
        )
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.source_model_id == "alpha"
        assert token.target_model_id == "beta"


# ---------------------------------------------------------------------------
# Privacy & drift
# ---------------------------------------------------------------------------


class TestModeWPrivacyAndDrift:
    def test_privacy_epsilon_in_token(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, privacy_epsilon=0.5, uhs_epochs=1)
        assert token.privacy_epsilon == 0.5

    def test_privacy_delta_in_token(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, privacy_delta=1e-6, uhs_epochs=1)
        assert token.privacy_delta == 1e-6

    def test_drift_score_finite_and_non_negative(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert np.isfinite(token.drift_score)
        assert token.drift_score >= 0.0


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------


class TestModeWCustomParams:
    def test_custom_svd_rank(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, svd_rank=2, uhs_epochs=1)
        assert isinstance(token, TesseraToken)
        assert token.custom_metadata["svd_rank"] == 2

    def test_custom_hub_dim(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=128)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert len(token.uhs_vector) == 128
        assert token.custom_metadata["hub_dim"] == 128

    def test_custom_svd_energy(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, svd_energy=0.80, uhs_epochs=1)
        assert token.custom_metadata["svd_energy"] == 0.80

    def test_correspondence_strategy_stored(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, correspondence_strategy="greedy", uhs_epochs=1)
        assert token.custom_metadata["correspondence_strategy"] == "greedy"


# ---------------------------------------------------------------------------
# Post-execute state
# ---------------------------------------------------------------------------


class TestModeWPostExecuteState:
    def test_correspondences_populated(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        transfer.execute(ref_loader, uhs_epochs=1)
        assert transfer.correspondences is not None
        assert isinstance(transfer.correspondences, list)

    def test_tx_snapshots_populated(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        transfer.execute(ref_loader, uhs_epochs=1)
        assert transfer.tx_snapshots is not None
        assert len(transfer.tx_snapshots) > 0

    def test_tx_uhs_populated(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        transfer.execute(ref_loader, uhs_epochs=1)
        assert transfer.tx_uhs is not None

    def test_receiver_weights_changed(self, small_tx, small_rx, ref_loader):
        """Receiver weights should be different from random init after transfer."""
        rx_weights_before = {
            name: p.data.clone() for name, p in small_rx.named_parameters() if "weight" in name
        }
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        transfer.execute(ref_loader, uhs_epochs=1)
        rx_weights_after = {
            name: p.data for name, p in small_rx.named_parameters() if "weight" in name
        }
        any_changed = any(
            not torch.allclose(rx_weights_before[n], rx_weights_after[n]) for n in rx_weights_before
        )
        assert any_changed, "At least one weight should have changed after Mode W transfer"


# ---------------------------------------------------------------------------
# Token metadata
# ---------------------------------------------------------------------------


class TestModeWTokenMetadata:
    def test_custom_metadata_contains_mode(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.custom_metadata.get("mode") == "W"

    def test_timestamp_present(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.timestamp is not None
        assert len(token.timestamp) > 0

    def test_generation_is_one(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.generation == 1

    def test_version_is_1_0(self, small_tx, small_rx, ref_loader):
        transfer = ModeWTransfer(small_tx, small_rx, hub_dim=64)
        token = transfer.execute(ref_loader, uhs_epochs=1)
        assert token.version == "1.0"
