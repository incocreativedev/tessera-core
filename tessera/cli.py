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
        print("File format: legacy (SafeTensors + JSON)")
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


# --- swarm --------------------------------------------------------------------


def _cmd_swarm_submit(args):
    """Submit a contributor token to central ingress."""
    from .swarm import submit

    ok, msg = submit(args.token, args.contributor_id)
    if ok:
        print(msg)
        return 0
    print(f"Error: {msg}", file=sys.stderr)
    return 1


def _cmd_swarm_aggregate(args):
    """Aggregate accepted tokens for a round (hub vector)."""
    from .swarm import aggregate

    token_paths = getattr(args, "tokens", None) or []
    if not token_paths and getattr(args, "token", None):
        token_paths = [args.token]
    if not token_paths:
        print("Error: provide token paths (e.g. --tokens a.tbf b.tbf)", file=sys.stderr)
        return 1
    vec = aggregate(args.round, token_paths)
    if vec is None:
        print("Error: not enough accepted contributors for round", file=sys.stderr)
        return 1
    print(json.dumps({"round_id": args.round, "vector_dim": len(vec), "sample": vec[:8]}))
    return 0


def _cmd_swarm_broadcast(args):
    """Emit broadcast token for round (large model -> contributors)."""
    from .swarm import aggregate, broadcast
    from .binary import TBFSerializer

    token_paths = getattr(args, "tokens", None) or []
    if not token_paths and getattr(args, "token", None):
        token_paths = [args.token]
    if not token_paths:
        print(
            "Error: provide token paths for aggregate (e.g. --tokens a.tbf b.tbf)", file=sys.stderr
        )
        return 1
    vec = aggregate(args.round, token_paths)
    if vec is None:
        print("Error: not enough accepted contributors for round", file=sys.stderr)
        return 1
    version = getattr(args, "broadcast_version", None) or f"round-{args.round}"
    token = broadcast(args.round, vec, version)
    out = getattr(args, "output", None) or f"broadcast-{args.round}.tbf"
    TBFSerializer.save(Path(out), token)
    print(f"Broadcast token written to {out}")
    return 0


def _cmd_swarm_score(args):
    """Score tokens for a round (utility)."""
    from .swarm import score

    token_paths = getattr(args, "tokens", None) or []
    if not token_paths and getattr(args, "token", None):
        token_paths = [args.token]
    if not token_paths:
        print("Error: provide token paths (e.g. --tokens a.tbf b.tbf)", file=sys.stderr)
        return 1
    scores = score(args.round, token_paths)
    print(json.dumps(scores, indent=2))
    return 0


def _cmd_swarm_credits(args):
    """Show credits / free-usage tier for contributor."""
    from pathlib import Path

    ledger_path = getattr(args, "ledger", None) or os.environ.get("TESSERA_CREDITS_LEDGER")
    cid = args.contributor_id

    if ledger_path:
        path = Path(ledger_path)
        if not path.exists():
            print(f"Ledger file not found: {path}", file=sys.stderr)
            return 1
        try:
            with open(path) as f:
                data = json.load(f)
            from .credits import CreditsLedger

            ledger = CreditsLedger.from_list(
                data if isinstance(data, list) else data.get("entries", [])
            )
        except Exception as e:
            print(f"Failed to load ledger: {e}", file=sys.stderr)
            return 1
        total = ledger.get_contributor_credits(cid)
        rolling = ledger.rolling_30_day_credits(cid)
        print(f"Contributor: {cid}")
        print(f"Total credits (all time): {total:.4f}")
        print(f"Rolling 30-day credits:   {rolling:.4f}")
        return 0

    print(f"Contributor: {cid}")
    print("No ledger specified. Use --ledger <path> or TESSERA_CREDITS_LEDGER to show credits.")
    print("Credits (v1): rolling 30-day sum of accepted utility scores.")
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

    # -- swarm ------------------------------------------------------------------
    p_swarm = subparsers.add_parser(
        "swarm", help="Swarm round-trip: submit, aggregate, broadcast, score, credits"
    )
    swarm_sub = p_swarm.add_subparsers(dest="swarm_command")

    p_submit = swarm_sub.add_parser("submit", help="Submit a contributor token to central ingress")
    p_submit.add_argument("--token", required=True, help="Path to .tbf token file")
    p_submit.add_argument(
        "--contributor-id", required=True, dest="contributor_id", help="Contributor ID"
    )

    p_agg = swarm_sub.add_parser("aggregate", help="Aggregate accepted tokens for a round")
    p_agg.add_argument("--round", required=True, dest="round", help="Round ID")
    p_agg.add_argument("--tokens", nargs="+", default=[], help="Paths to accepted .tbf token files")

    p_broad = swarm_sub.add_parser("broadcast", help="Emit broadcast token for round")
    p_broad.add_argument("--round", required=True, dest="round", help="Round ID")
    p_broad.add_argument(
        "--tokens", nargs="+", default=[], help="Paths to accepted .tbf token files"
    )
    p_broad.add_argument(
        "--broadcast-version",
        dest="broadcast_version",
        default=None,
        help="Broadcast version string",
    )
    p_broad.add_argument("--output", default=None, help="Output .tbf path")

    p_score = swarm_sub.add_parser("score", help="Score tokens for a round")
    p_score.add_argument("--round", required=True, dest="round", help="Round ID")
    p_score.add_argument("--tokens", nargs="+", default=[], help="Paths to .tbf token files")

    p_cred = swarm_sub.add_parser("credits", help="Show credits for contributor")
    p_cred.add_argument(
        "--contributor-id", required=True, dest="contributor_id", help="Contributor ID"
    )
    p_cred.add_argument(
        "--ledger",
        default=None,
        dest="ledger",
        help="Path to credits ledger JSON (or set TESSERA_CREDITS_LEDGER)",
    )

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
        "swarm": None,  # handled below
    }

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "swarm":
        swarm_dispatch = {
            "submit": _cmd_swarm_submit,
            "aggregate": _cmd_swarm_aggregate,
            "broadcast": _cmd_swarm_broadcast,
            "score": _cmd_swarm_score,
            "credits": _cmd_swarm_credits,
        }
        sub = getattr(args, "swarm_command", None)
        if sub is None:
            p_swarm.print_help()
            return 0
        handler = swarm_dispatch.get(sub)
        if handler is None:
            p_swarm.print_help()
            return 1
        rc = handler(args)
        sys.exit(rc if rc else 0)

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    rc = handler(args)
    sys.exit(rc if rc else 0)


if __name__ == "__main__":
    main()
