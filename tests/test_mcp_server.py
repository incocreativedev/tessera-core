"""Tests for tessera.mcp_server — MCP tool handlers."""

import json
import os
import tempfile

import numpy as np
import pytest

from tessera.binary import TBFSerializer
from tessera.token import TesseraToken, KnowledgeType

# Skip the entire module if the mcp package is not installed.
mcp_installed = True
try:
    import mcp  # noqa: F401
except ImportError:
    mcp_installed = False

pytestmark = pytest.mark.skipif(
    not mcp_installed,
    reason="MCP server tests require the 'mcp' package",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_token(dim=2048):
    """Create a minimal TesseraToken for testing."""
    vec = np.random.randn(dim).astype(np.float32).tolist()
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=vec,
        modality_weights={"A": 0.85, "W": 0.15},
        correlation_map={"tok_0": 0.42},
        lineage_dag={"nodes": [], "root": "n0"},
        source_model_id="test_source",
        target_model_id="test_target",
        drift_score=0.05,
    )


@pytest.fixture
def tbf_file(tmp_path):
    """Create a temporary TBF file for testing."""
    token = _make_token()
    filepath = tmp_path / "test_token.tbf"
    TBFSerializer.save(str(filepath), token)
    return str(filepath), token


@pytest.fixture
def tbf_file_hmac(tmp_path):
    """Create a temporary TBF file with HMAC for testing."""
    token = _make_token()
    filepath = tmp_path / "test_token_hmac.tbf"
    hmac_key = b"test-secret-key"
    TBFSerializer.save(str(filepath), token, hmac_key=hmac_key)
    return str(filepath), token, hmac_key


# ── Import the handlers (only when mcp is available) ─────────────────────────

if mcp_installed:
    from tessera.mcp_server import (
        _handle_inspect,
        _handle_validate,
        _handle_list_anchors,
        _handle_info,
        list_tools,
        call_tool,
    )


# ── Test: module imports ─────────────────────────────────────────────────────

class TestModuleImport:
    def test_import_succeeds(self):
        """The mcp_server module should import without error when mcp is installed."""
        import tessera.mcp_server  # noqa: F811
        assert hasattr(tessera.mcp_server, "app")
        assert hasattr(tessera.mcp_server, "main")

    def test_has_tool_handlers(self):
        """All expected handler functions should exist."""
        from tessera.mcp_server import _HANDLERS
        expected = {"tessera_inspect", "tessera_validate", "tessera_list_anchors", "tessera_info"}
        assert set(_HANDLERS.keys()) == expected


# ── Test: tessera_inspect ────────────────────────────────────────────────────

class TestInspect:
    def test_inspect_header(self, tbf_file):
        """Inspect without full=True should return header info."""
        filepath, _ = tbf_file
        result = json.loads(_handle_inspect({"filepath": filepath}))

        assert result["format"] == "TBF"
        assert result["version"] == "1.1"
        assert result["quantisation"] == "FLOAT32"
        assert result["vector_count"] == 2048
        assert result["header_crc_ok"] is True
        assert "total_file_bytes" in result

    def test_inspect_full(self, tbf_file):
        """Inspect with full=True should return all token metadata."""
        filepath, token = tbf_file
        result = json.loads(_handle_inspect({"filepath": filepath, "full": True}))

        assert result["source_model_id"] == "test_source"
        assert result["target_model_id"] == "test_target"
        assert result["vector_dim"] == 2048
        assert result["modality_weights"] == {"A": 0.85, "W": 0.15}
        assert result["generation"] == 1
        assert "timestamp" in result
        assert "privacy_epsilon" in result

    def test_inspect_missing_file(self):
        """Inspect on a non-existent file should raise an error."""
        with pytest.raises(Exception):
            _handle_inspect({"filepath": "/nonexistent/path/token.tbf"})


# ── Test: tessera_validate ───────────────────────────────────────────────────

class TestValidate:
    def test_validate_valid_file(self, tbf_file):
        """A valid TBF file should pass validation."""
        filepath, _ = tbf_file
        result = json.loads(_handle_validate({"filepath": filepath}))

        assert result["valid"] is True
        assert result["source"] == "test_source"
        assert result["target"] == "test_target"
        assert result["vector_dim"] == 2048
        assert result["quantisation"] == "FLOAT32"

    def test_validate_corrupted_file(self, tmp_path):
        """A corrupted file should fail validation."""
        # Create a valid file then corrupt it
        token = _make_token()
        filepath = tmp_path / "corrupted.tbf"
        TBFSerializer.save(str(filepath), token)

        # Corrupt a byte in the middle of the file
        data = bytearray(filepath.read_bytes())
        midpoint = len(data) // 2
        data[midpoint] ^= 0xFF
        filepath.write_bytes(bytes(data))

        result = json.loads(_handle_validate({"filepath": str(filepath)}))
        assert result["valid"] is False
        assert "error" in result

    def test_validate_not_tbf(self, tmp_path):
        """A non-TBF file should fail validation."""
        filepath = tmp_path / "not_a_token.tbf"
        filepath.write_text("this is not a TBF file")

        result = json.loads(_handle_validate({"filepath": str(filepath)}))
        assert result["valid"] is False
        assert "error" in result

    def test_validate_with_hmac(self, tbf_file_hmac):
        """Validation with the correct HMAC key should succeed."""
        filepath, _, hmac_key = tbf_file_hmac
        result = json.loads(
            _handle_validate({
                "filepath": filepath,
                "hmac_key": hmac_key.hex(),
            })
        )
        assert result["valid"] is True

    def test_validate_with_wrong_hmac(self, tbf_file_hmac):
        """Validation with the wrong HMAC key should fail."""
        filepath, _, _ = tbf_file_hmac
        result = json.loads(
            _handle_validate({
                "filepath": filepath,
                "hmac_key": "00" * 16,  # wrong key
            })
        )
        assert result["valid"] is False
        assert "HMAC" in result["error"]


# ── Test: tessera_list_anchors ───────────────────────────────────────────────

class TestListAnchors:
    def test_empty_registry(self, tmp_path):
        """An empty registry should return an empty list."""
        registry_dir = str(tmp_path / "empty_registry")
        result = json.loads(
            _handle_list_anchors({"registry_dir": registry_dir})
        )
        assert result == []

    def test_default_registry_dir(self):
        """Calling without registry_dir should use the default (~/.tessera/)."""
        # This should not raise, even if ~/.tessera/ has no anchors
        result = json.loads(_handle_list_anchors({}))
        assert isinstance(result, list)


# ── Test: tessera_info ───────────────────────────────────────────────────────

class TestInfo:
    def test_returns_expected_fields(self):
        """Info should return version, python, torch, cuda, and hub_dim."""
        result = json.loads(_handle_info({}))

        assert "tessera_version" in result
        assert result["tessera_version"] == "0.1.0"
        assert "python_version" in result
        assert "torch_version" in result
        assert "cuda_available" in result
        assert isinstance(result["cuda_available"], bool)
        assert result["default_hub_dim"] == 2048


# ── Test: call_tool dispatcher ───────────────────────────────────────────────

class TestCallTool:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Calling an unknown tool should return an error."""
        results = await call_tool("nonexistent_tool", {})
        assert len(results) == 1
        data = json.loads(results[0].text)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_dispatch_info(self):
        """The dispatcher should correctly route to tessera_info."""
        results = await call_tool("tessera_info", {})
        assert len(results) == 1
        data = json.loads(results[0].text)
        assert "tessera_version" in data

    @pytest.mark.asyncio
    async def test_dispatch_inspect(self, tbf_file):
        """The dispatcher should correctly route to tessera_inspect."""
        filepath, _ = tbf_file
        results = await call_tool("tessera_inspect", {"filepath": filepath})
        assert len(results) == 1
        data = json.loads(results[0].text)
        assert data["format"] == "TBF"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Tool errors should be caught and returned as JSON."""
        results = await call_tool(
            "tessera_inspect",
            {"filepath": "/nonexistent/file.tbf"},
        )
        assert len(results) == 1
        data = json.loads(results[0].text)
        assert "error" in data


# ── Test: list_tools ─────────────────────────────────────────────────────────

class TestListTools:
    @pytest.mark.asyncio
    async def test_returns_all_tools(self):
        """list_tools should return all four tool definitions."""
        tools = await list_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {
            "tessera_inspect",
            "tessera_validate",
            "tessera_list_anchors",
            "tessera_info",
        }

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """Each tool should have a valid JSON Schema inputSchema."""
        tools = await list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
