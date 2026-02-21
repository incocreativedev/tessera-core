"""
Tests for tessera.cli — command-line interface.

All tests invoke the CLI via subprocess to mirror real usage
(python -m tessera.cli <command>).
"""

import subprocess
import sys


def _run_cli(*args):
    """Run tessera CLI as a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "tessera.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestInfo:
    def test_info_returns_zero(self):
        result = _run_cli("info")
        assert result.returncode == 0

    def test_info_contains_version(self):
        import tessera

        result = _run_cli("info")
        assert tessera.__version__ in result.stdout


class TestInspect:
    def test_inspect_nonexistent_file_returns_nonzero(self):
        result = _run_cli("inspect", "/tmp/__nonexistent_tessera_file__.tbf")
        assert result.returncode != 0

    def test_inspect_nonexistent_file_prints_error(self):
        result = _run_cli("inspect", "/tmp/__nonexistent_tessera_file__.tbf")
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestValidate:
    def test_validate_nonexistent_file_returns_nonzero(self):
        result = _run_cli("validate", "/tmp/__nonexistent_tessera_file__.tbf")
        assert result.returncode != 0

    def test_validate_nonexistent_file_prints_error(self):
        result = _run_cli("validate", "/tmp/__nonexistent_tessera_file__.tbf")
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestListAnchors:
    def test_list_anchors_returns_zero(self, tmp_path):
        result = _run_cli("list-anchors", "--dir", str(tmp_path / "empty_registry"))
        assert result.returncode == 0

    def test_list_anchors_empty_registry_message(self, tmp_path):
        result = _run_cli("list-anchors", "--dir", str(tmp_path / "empty_registry"))
        assert "no anchors" in result.stdout.lower()


class TestTransfer:
    def test_transfer_returns_zero(self):
        result = _run_cli("transfer")
        assert result.returncode == 0

    def test_transfer_shows_guidance(self):
        result = _run_cli("transfer")
        assert "python api" in result.stdout.lower() or "demo_transfer" in result.stdout


class TestVersionFlag:
    def test_version_flag(self):
        import tessera

        result = _run_cli("--version")
        assert result.returncode == 0
        assert tessera.__version__ in result.stdout
