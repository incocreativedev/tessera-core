"""
tessera.cli — Command-line interface for the Tessera toolkit.

Provides inspection, validation, registry management, and system info
commands. All functionality uses stdlib argparse with zero additional
dependencies beyond what Tessera itself requires.

Usage:
    tessera inspect <file> [--full]
    tessera validate <file> [--hmac-key HEX]
    tessera list-anchors [--dir PATH]
    tessera benchmark [--quick]
    tessera info
    tessera transfer
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _cmd_inspect(args):
    """Inspect a Tessera token file (TBF or legacy)."""
    from .binary import TBFSerializer

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        return 1

    try:
        fmt = TBFSerializer.detect_format(filepath)
    except Exception as exc:
        print(f"Error detecting format: {exc}", file=sys.stderr)
        return 1

    if fmt == "unknown":
        print(f"Error: unrecognised file format: {filepath}", file=sys.stderr)
        return 1

    if fmt == "legacy":
        print(f"File format: legacy (SafeTensors + JSON)")
        print("Full inspection of legacy files is not yet supported in the CLI.")
        print("Use the Python API: TokenSerializer.load_token(path)")
        return 0

    # TBF format
    try:
        if args.full:
            token = TBFSerializer.load(filepath)
            data = {
                "file": str(filepath),
                "format": "TBF",
                "knowledge_type": token.knowledge_type.value,
                "source_model_id": token.source_model_id,
                "target_model_id": token.target_model_id,
                "drift_score": token.drift_score,
                "privacy_epsilon": token.privacy_epsilon,
                "privacy_delta": token.privacy_delta,
                "timestamp": token.timestamp,
                "version": token.version,
                "generation": token.generation,
                "uhs_vector_dim": len(token.uhs_vector),
                "modality_weights": token.modality_weights,
                "correlation_map": token.correlation_map,
                "lineage_dag": token.lineage_dag,
                "projection_hints": token.projection_hints,
                "custom_metadata": token.custom_metadata,
            }
        else:
            data = TBFSerializer.info(filepath)
            data["file"] = str(filepath)
    except Exception as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2, default=str))
    return 0


def _cmd_validate(args):
    """Validate a Tessera token file."""
    from .binary import TBFSerializer

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        return 1

    hmac_key = None
    if args.hmac_key:
        try:
            hmac_key = bytes.fromhex(args.hmac_key)
        except ValueError:
            print("Error: --hmac-key must be a valid hex string", file=sys.stderr)
            return 1

    try:
        token = TBFSerializer.load(filepath, hmac_key=hmac_key, verify_crc=True)
        dim = len(token.uhs_vector)
        src = token.source_model_id or "(unknown)"
        tgt = token.target_model_id or "(unknown)"
        print(f"\u2713 Valid TBF file: {filepath}")
        print(f"  Source:    {src}")
        print(f"  Target:    {tgt}")
        print(f"  Dimension: {dim}")
        return 0
    except Exception as exc:
        print(f"\u2717 Validation failed: {filepath}")
        print(f"  Error: {exc}", file=sys.stderr)
        return 1


def _cmd_list_anchors(args):
    """List registered anchor models."""
    from .registry import AnchorRegistry

    registry_dir = args.dir if args.dir else None
    try:
        registry = AnchorRegistry(registry_dir=registry_dir)
    except Exception as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    anchor_ids = registry.list()
    if not anchor_ids:
        path = registry.root
        print(f"No anchors registered in {path}")
        return 0

    # Print table header
    print(f"{'Anchor ID':<30} {'d_model':>8} {'hub_dim':>8} {'Path'}")
    print(f"{'-' * 30} {'-' * 8} {'-' * 8} {'-' * 40}")

    for aid in sorted(anchor_ids):
        try:
            meta = registry.info(aid)
            d_model = meta.get("d_model", "?")
            hub_dim = meta.get("hub_dim", "?")
            path = meta.get("path", "?")
            print(f"{aid:<30} {str(d_model):>8} {str(hub_dim):>8} {path}")
        except Exception:
            print(f"{aid:<30} {'?':>8} {'?':>8} (error reading metadata)")

    return 0


def _cmd_benchmark(args):
    """Run the cross-architecture benchmark suite."""
    # Locate benchmark script relative to project root
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    script_path = project_root / "benchmarks" / "cross_arch_benchmark.py"

    if not script_path.exists():
        print("Benchmark script not found.")
        print(f"  Expected: {script_path}")
        print()
        print("If you installed tessera via pip, benchmarks are not included.")
        print("Clone the repository to run benchmarks:")
        print("  git clone https://github.com/tessera-ai/tessera-core.git")
        print("  cd tessera-core")
        print("  python benchmarks/cross_arch_benchmark.py")
        return 1

    cmd = [sys.executable, str(script_path)]
    if args.quick:
        cmd.append("--quick")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def _cmd_info(args):
    """Print Tessera system information."""
    import tessera
    from .uhs import UHS_DIM

    # Python version
    py_version = sys.version.split()[0]

    # PyTorch version
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_str = f"Yes ({torch.cuda.get_device_name(0)})" if cuda_available else "No"
    except ImportError:
        torch_version = "not installed"
        cuda_str = "N/A"

    registry_path = Path.home() / ".tessera"

    print(f"Tessera v{tessera.__version__}")
    print(f"  Python:       {py_version}")
    print(f"  PyTorch:      {torch_version}")
    print(f"  CUDA:         {cuda_str}")
    print(f"  Hub dim:      {UHS_DIM}")
    print(f"  Registry:     {registry_path}")
    return 0


def _cmd_transfer(args):
    """Show transfer API usage guidance."""
    print("Knowledge transfer requires the Python API.")
    print()
    print("Example usage:")
    print()
    print("  from tessera import ModeATransfer, TBFSerializer")
    print()
    print("  transfer = ModeATransfer(")
    print('      transmitter, receiver, "model_a", "model_b"')
    print("  )")
    print("  token = transfer.execute(train_loader, val_loader)")
    print('  TBFSerializer.save("transfer.tbf", token)')
    print()
    print("See examples/demo_transfer.py for a complete walkthrough.")
    return 0


def main():
    """Tessera CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Tessera: activation-based AI-to-AI knowledge transfer toolkit",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # -- inspect ---------------------------------------------------------------
    p_inspect = subparsers.add_parser("inspect", help="Inspect a Tessera token file")
    p_inspect.add_argument("file", help="Path to .tbf or legacy token file")
    p_inspect.add_argument(
        "--full",
        action="store_true",
        help="Load full token and show all metadata",
    )

    # -- validate --------------------------------------------------------------
    p_validate = subparsers.add_parser("validate", help="Validate a Tessera token file")
    p_validate.add_argument("file", help="Path to .tbf token file")
    p_validate.add_argument(
        "--hmac-key",
        default=None,
        help="Hex-encoded HMAC key for integrity verification",
    )

    # -- list-anchors ----------------------------------------------------------
    p_anchors = subparsers.add_parser("list-anchors", help="List registered anchor models")
    p_anchors.add_argument(
        "--dir",
        default=None,
        help="Registry directory (default: ~/.tessera/)",
    )

    # -- benchmark -------------------------------------------------------------
    p_bench = subparsers.add_parser("benchmark", help="Run cross-architecture benchmark")
    p_bench.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster subset of benchmarks",
    )

    # -- info ------------------------------------------------------------------
    subparsers.add_parser("info", help="Print Tessera system information")

    # -- transfer --------------------------------------------------------------
    subparsers.add_parser("transfer", help="Show transfer API usage guidance")

    args = parser.parse_args()

    # Handle --version at top level
    if args.version:
        import tessera

        print(f"tessera {tessera.__version__}")
        return 0

    # Dispatch to subcommand
    dispatch = {
        "inspect": _cmd_inspect,
        "validate": _cmd_validate,
        "list-anchors": _cmd_list_anchors,
        "benchmark": _cmd_benchmark,
        "info": _cmd_info,
        "transfer": _cmd_transfer,
    }

    if args.command is None:
        parser.print_help()
        return 0

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    rc = handler(args)
    sys.exit(rc if rc else 0)


if __name__ == "__main__":
    main()
