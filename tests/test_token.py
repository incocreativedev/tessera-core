"""Tests for tessera.token — TesseraToken and TokenSerializer."""

import os
import numpy as np
from tessera.token import TesseraToken, KnowledgeType, TokenSerializer


def make_token(uhs_vector=None):
    """Helper to create a sample token."""
    if uhs_vector is None:
        uhs_vector = np.random.randn(2048).astype(np.float32).tolist()
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=uhs_vector,
        modality_weights={"A": 0.85, "W": 0.12, "B": 0.03},
        correlation_map={"token-abc": 0.94},
        lineage_dag={"nodes": [{"id": "n0", "type": "anchor"}], "root": "n0"},
        generation=1,
        projection_hints=[],
        privacy_epsilon=8.0,
        privacy_delta=1e-5,
        drift_score=0.034,
        source_model_id="model_a",
        target_model_id="model_b",
    )


class TestKnowledgeType:
    def test_enum_values(self):
        assert KnowledgeType.ACTIVATION.value == "ACTIVATION"
        assert KnowledgeType.WEIGHT.value == "WEIGHT"
        assert KnowledgeType.DATASET.value == "DATASET"
        assert KnowledgeType.BEHAVIOUR.value == "BEHAVIOUR"

    def test_from_value(self):
        assert KnowledgeType("ACTIVATION") == KnowledgeType.ACTIVATION


class TestTesseraToken:
    def test_create_token(self):
        token = make_token()
        assert token.knowledge_type == KnowledgeType.ACTIVATION
        assert len(token.uhs_vector) == 2048
        assert token.source_model_id == "model_a"
        assert token.drift_score == 0.034

    def test_to_dict_round_trip(self):
        token = make_token()
        d = token.to_dict()
        assert isinstance(d, dict)
        assert d["knowledge_type"] == "ACTIVATION"
        assert d["modality_weights"]["A"] == 0.85

        restored = TesseraToken.from_dict(d)
        assert restored.knowledge_type == token.knowledge_type
        assert restored.source_model_id == token.source_model_id
        assert restored.drift_score == token.drift_score

    def test_timestamp_auto_generated(self):
        token = make_token()
        assert token.timestamp  # Not empty
        assert "T" in token.timestamp  # ISO 8601

    def test_version_default(self):
        token = make_token()
        assert token.version == "1.0"


class TestTokenSerializer:
    def test_save_and_load(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test.safetensors")
        TokenSerializer.save_token(token, path)

        assert os.path.exists(path)
        assert os.path.exists(path.replace(".safetensors", ".json"))

        loaded = TokenSerializer.load_token(path)
        assert loaded.knowledge_type == token.knowledge_type
        assert loaded.source_model_id == token.source_model_id
        assert abs(loaded.drift_score - token.drift_score) < 1e-6
        assert len(loaded.uhs_vector) == 2048

    def test_vector_fidelity(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "fidelity.safetensors")
        TokenSerializer.save_token(token, path)
        loaded = TokenSerializer.load_token(path)

        max_error = max(abs(a - b) for a, b in zip(token.uhs_vector, loaded.uhs_vector))
        assert max_error < 1e-6, f"Vector fidelity loss: {max_error}"
