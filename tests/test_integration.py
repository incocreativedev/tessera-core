"""Integration tests — full pipeline from transfer through serialisation."""

import os
import torch
import numpy as np
from tessera import (
    ModeATransfer, TesseraToken, TokenSerializer,
    TBFSerializer, QuantType, DriftMeasure,
    ActivationFingerprint, compute_fingerprints,
    DifferentialPrivacy, ProjectionType, ProjectionHint,
    AnchorRegistry,
)
from tests.conftest import SmallTransformer


class TestPublicAPI:
    """Verify all 17 public symbols import correctly."""

    def test_all_exports(self):
        import tessera
        assert len(tessera.__all__) == 39

    def test_version(self):
        import tessera
        assert tessera.__version__ == "0.1.0"


class TestEndToEndPipeline:
    """Full transfer → serialise → load → verify pipeline."""

    def test_transfer_and_serialise_legacy(self, train_loader, val_loader, tmp_dir):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=128, num_layers=3)

        # Train transmitter briefly
        opt = torch.optim.Adam(tx.parameters(), lr=1e-3)
        tx.train()
        for bx, by in train_loader:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(tx(bx), by).backward()
            opt.step()
        tx.eval()

        # Transfer
        transfer = ModeATransfer(tx, rx, "tx", "rx")
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)

        # Serialise (legacy)
        path = os.path.join(tmp_dir, "token.safetensors")
        TokenSerializer.save_token(token, path)
        loaded = TokenSerializer.load_token(path)

        assert loaded.source_model_id == "tx"
        assert loaded.drift_score == token.drift_score

    def test_transfer_and_serialise_tbf(self, train_loader, val_loader, tmp_dir):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeATransfer(tx, rx, "a", "b")
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)

        # Serialise (TBF with HMAC)
        path = os.path.join(tmp_dir, "token.tbf")
        key = b"integration-test-key"
        TBFSerializer.save(path, token, quant=QuantType.FLOAT16, hmac_key=key)
        loaded = TBFSerializer.load(path, hmac_key=key)

        assert loaded.source_model_id == "a"
        max_err = max(abs(a - b) for a, b in zip(token.uhs_vector, loaded.uhs_vector))
        assert max_err < 0.01  # FLOAT16 precision

    def test_registry_round_trip(self, train_loader, val_loader, tmp_dir):
        """Register anchors from a transfer, then reload them."""
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=128, num_layers=3)
        tx.eval()

        transfer = ModeATransfer(tx, rx, "tx_64", "rx_128")
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)

        # Register the transmitter's UHS pair
        reg = AnchorRegistry(registry_dir=tmp_dir)
        reg.register(
            "tx_64",
            d_model=64,
            encoder=transfer.tx_uhs.encoder,
            decoder=transfer.tx_uhs.decoder,
        )

        assert "tx_64" in reg
        enc, dec = reg.load("tx_64")

        # Verify encoder produces same output
        x = torch.randn(4, 64)
        out1 = transfer.tx_uhs.encode(x)
        out2 = enc(x)
        torch.testing.assert_close(out1, out2)


class TestQuantisationPipeline:
    """Verify all quantisation levels work end-to-end."""

    def test_all_quant_types(self, tmp_dir):
        from tests.test_binary import make_token
        token = make_token(dim=512)

        for quant in QuantType:
            path = os.path.join(tmp_dir, f"quant_{quant.name}.tbf")
            TBFSerializer.save(path, token, quant=quant)
            loaded = TBFSerializer.load(path)
            assert loaded.knowledge_type == token.knowledge_type
            assert len(loaded.uhs_vector) == 512
