"""Tests for tessera.transfer — Mode A transfer orchestrator."""

import torch
import pytest
from tessera.transfer import ModeATransfer
from tessera.token import TesseraToken
from tests.conftest import SmallTransformer


class TestModeATransfer:
    def test_execute_returns_token(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)

        # Quick train transmitter
        opt = torch.optim.Adam(tx.parameters(), lr=1e-3)
        tx.train()
        for batch_x, batch_y in train_loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(tx(batch_x), batch_y)
            loss.backward()
            opt.step()
        tx.eval()

        transfer = ModeATransfer(tx, rx, "tx_64", "rx_64")
        token = transfer.execute(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert isinstance(token, TesseraToken)
        assert len(token.uhs_vector) == 2048
        assert token.source_model_id == "tx_64"
        assert token.target_model_id == "rx_64"
        assert token.drift_score >= 0

    def test_cross_architecture_transfer(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=128, num_layers=3)

        # Quick train
        opt = torch.optim.Adam(tx.parameters(), lr=1e-3)
        tx.train()
        for batch_x, batch_y in train_loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(tx(batch_x), batch_y)
            loss.backward()
            opt.step()
        tx.eval()

        transfer = ModeATransfer(tx, rx, "tx_small", "rx_big")
        token = transfer.execute(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
        )

        assert isinstance(token, TesseraToken)
        assert token.drift_score >= 0

    def test_privacy_budget_in_token(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeATransfer(tx, rx, "a", "b")
        token = transfer.execute(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            uhs_epochs=1,
            finetune_epochs=1,
            privacy_epsilon=4.0,
            privacy_delta=1e-6,
        )

        assert token.privacy_epsilon == 4.0
        assert token.privacy_delta == 1e-6
