"""Tests for tessera.registry — Anchor model registry."""

import pytest
from tessera.registry import AnchorRegistry
from tessera.uhs import EncoderMLP, DecoderMLP


class TestAnchorRegistry:
    def test_register_and_list(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        enc = EncoderMLP(64, 256)
        dec = DecoderMLP(64, 256)
        reg.register("test_model", d_model=64, encoder=enc, decoder=dec)

        assert "test_model" in reg
        assert "test_model" in reg.list()

    def test_load_round_trip(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        enc = EncoderMLP(64, 256)
        dec = DecoderMLP(64, 256)
        reg.register("model_a", d_model=64, encoder=enc, decoder=dec)

        loaded_enc, loaded_dec = reg.load("model_a")
        # Check shapes match
        import torch
        x = torch.randn(4, 64)
        out_original = enc(x)
        out_loaded = loaded_enc(x)
        torch.testing.assert_close(out_original, out_loaded)

    def test_load_nonexistent_raises(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        with pytest.raises(KeyError, match="not found"):
            reg.load("nonexistent")

    def test_info(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        enc = EncoderMLP(128, 512)
        dec = DecoderMLP(128, 512)
        reg.register("big_model", d_model=128, encoder=enc, decoder=dec)

        info = reg.info("big_model")
        assert info["d_model"] == 128
        assert info["hub_dim"] == 512

    def test_multiple_anchors(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        for dim in [64, 128, 256]:
            enc = EncoderMLP(dim, 256)
            dec = DecoderMLP(dim, 256)
            reg.register(f"model_{dim}", d_model=dim, encoder=enc, decoder=dec)

        assert len(reg.list()) == 3

    def test_persistence(self, tmp_dir):
        """Registry should survive re-instantiation."""
        reg1 = AnchorRegistry(registry_dir=tmp_dir)
        enc = EncoderMLP(64, 256)
        dec = DecoderMLP(64, 256)
        reg1.register("persistent", d_model=64, encoder=enc, decoder=dec)

        reg2 = AnchorRegistry(registry_dir=tmp_dir)
        assert "persistent" in reg2
        assert reg2.info("persistent")["d_model"] == 64

    def test_repr(self, tmp_dir):
        reg = AnchorRegistry(registry_dir=tmp_dir)
        s = repr(reg)
        assert "AnchorRegistry" in s
        assert "anchors=0" in s
