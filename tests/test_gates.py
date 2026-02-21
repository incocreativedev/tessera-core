"""Tests for tessera.gates — Projection types and hints."""

from tessera.gates import ProjectionType, ProjectionHint


class TestProjectionType:
    def test_enum_members(self):
        assert ProjectionType.ORTHOGONAL.value == "H"
        assert ProjectionType.CONDITIONAL.value == "CONDITIONAL"
        assert ProjectionType.SCALING.value == "P"
        assert ProjectionType.RESHAPE.value == "R"
        assert ProjectionType.SWAP.value == "SWAP"

    def test_from_value(self):
        assert ProjectionType("H") == ProjectionType.ORTHOGONAL
        assert ProjectionType("P") == ProjectionType.SCALING


class TestProjectionHint:
    def test_create(self):
        hint = ProjectionHint(
            projection_type=ProjectionType.SCALING,
            strength=0.8,
            target_layers=["layers.0", "layers.1"],
        )
        assert hint.projection_type == ProjectionType.SCALING
        assert hint.strength == 0.8
        assert len(hint.target_layers) == 2

    def test_to_dict(self):
        hint = ProjectionHint(
            projection_type=ProjectionType.ORTHOGONAL,
            strength=0.5,
            target_layers=["fc1"],
        )
        d = hint.to_dict()
        assert d["projection"] == "H"
        assert d["strength"] == 0.5
        assert d["target_layers"] == ["fc1"]

    def test_from_dict_round_trip(self):
        original = ProjectionHint(
            projection_type=ProjectionType.RESHAPE,
            strength=0.7,
            target_layers=["layer_a", "layer_b"],
        )
        d = original.to_dict()
        restored = ProjectionHint.from_dict(d)
        assert restored.projection_type == original.projection_type
        assert restored.strength == original.strength
        assert restored.target_layers == original.target_layers
