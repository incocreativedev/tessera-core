"""
tessera.mcp_server — MCP (Model Context Protocol) server for Tessera.

Exposes Tessera functionality as MCP tools for use by LLM agents and
other MCP-compatible clients.

Tools:
    tessera_inspect    — Inspect a TBF token file (header or full metadata)
    tessera_validate   — Validate a TBF token file (CRC + optional HMAC)
    tessera_list_anchors — List registered anchor models
    tessera_info       — Get Tessera installation info

Usage:
    python -m tessera.mcp_server
    # or
    tessera-mcp  (if installed with pip install 'tessera-core[mcp]')
"""

import asyncio
import json
import sys

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "MCP server requires the 'mcp' package.\n"
        "Install with: pip install 'tessera-core[mcp]'"
    )
    sys.exit(1)

from tessera.binary import TBFSerializer
from tessera.registry import AnchorRegistry
from tessera.uhs import UHS_DIM
import tessera

# ── Server instance ───────────────────────────────────────────────────────────

app = Server("tessera")

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="tessera_inspect",
        description=(
            "Inspect a Tessera Binary Format (TBF) token file. "
            "Returns header information by default, or full metadata "
            "when 'full' is set to true."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the .tbf token file to inspect.",
                },
                "full": {
                    "type": "boolean",
                    "description": (
                        "If true, load the full token and return all metadata fields. "
                        "If false (default), return only the header summary."
                    ),
                    "default": False,
                },
            },
            "required": ["filepath"],
        },
    ),
    Tool(
        name="tessera_validate",
        description=(
            "Validate a Tessera Binary Format (TBF) token file. "
            "Checks CRC integrity and optionally verifies HMAC authentication."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the .tbf token file to validate.",
                },
                "hmac_key": {
                    "type": "string",
                    "description": (
                        "Optional hex-encoded HMAC key for authentication verification. "
                        "Example: 'deadbeef01020304'"
                    ),
                },
            },
            "required": ["filepath"],
        },
    ),
    Tool(
        name="tessera_list_anchors",
        description=(
            "List all registered anchor models in a Tessera registry. "
            "Returns anchor IDs, model dimensions, and hub dimensions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "registry_dir": {
                    "type": "string",
                    "description": (
                        "Path to the registry directory. "
                        "Defaults to ~/.tessera/ if not specified."
                    ),
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="tessera_info",
        description=(
            "Get Tessera installation information including version, "
            "Python version, PyTorch version, CUDA availability, and "
            "default hub dimension."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return all available Tessera MCP tools."""
    return TOOLS


# ── Tool implementations ─────────────────────────────────────────────────────

def _handle_inspect(arguments: dict) -> str:
    """Inspect a TBF token file."""
    filepath = arguments["filepath"]
    full = arguments.get("full", False)

    if not full:
        info = TBFSerializer.info(filepath)
        return json.dumps(info, indent=2)
    else:
        token = TBFSerializer.load(filepath, verify_crc=True)
        result = {
            "knowledge_type": token.knowledge_type.value,
            "source_model_id": token.source_model_id,
            "target_model_id": token.target_model_id,
            "vector_dim": len(token.uhs_vector),
            "modality_weights": token.modality_weights,
            "correlation_map": token.correlation_map,
            "lineage_dag": token.lineage_dag,
            "generation": token.generation,
            "projection_hints": token.projection_hints,
            "privacy_epsilon": token.privacy_epsilon,
            "privacy_delta": token.privacy_delta,
            "drift_score": token.drift_score,
            "timestamp": token.timestamp,
            "version": token.version,
            "custom_metadata": token.custom_metadata,
        }
        return json.dumps(result, indent=2, default=str)


def _handle_validate(arguments: dict) -> str:
    """Validate a TBF token file."""
    filepath = arguments["filepath"]
    hmac_key_hex = arguments.get("hmac_key")

    hmac_key = None
    if hmac_key_hex:
        hmac_key = bytes.fromhex(hmac_key_hex)

    try:
        token = TBFSerializer.load(
            filepath, hmac_key=hmac_key, verify_crc=True
        )
        result = {
            "valid": True,
            "source": token.source_model_id,
            "target": token.target_model_id,
            "vector_dim": len(token.uhs_vector),
            "quantisation": TBFSerializer.info(filepath).get("quantisation", "unknown"),
        }
    except Exception as exc:
        result = {
            "valid": False,
            "error": str(exc),
        }

    return json.dumps(result, indent=2)


def _handle_list_anchors(arguments: dict) -> str:
    """List registered anchor models."""
    registry_dir = arguments.get("registry_dir")
    registry = AnchorRegistry(registry_dir)

    anchors = []
    for anchor_id in registry.list():
        info = registry.info(anchor_id)
        anchors.append({
            "anchor_id": anchor_id,
            **info,
        })

    return json.dumps(anchors, indent=2)


def _handle_info(arguments: dict) -> str:
    """Get Tessera installation info."""
    import platform

    torch_version = "unknown"
    cuda_available = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "not installed"

    result = {
        "tessera_version": tessera.__version__,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "default_hub_dim": UHS_DIM,
    }
    return json.dumps(result, indent=2)


# ── Dispatcher ────────────────────────────────────────────────────────────────

_HANDLERS = {
    "tessera_inspect": _handle_inspect,
    "tessera_validate": _handle_validate,
    "tessera_list_anchors": _handle_list_anchors,
    "tessera_info": _handle_info,
}


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch an MCP tool call to the appropriate handler."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )
        ]

    try:
        result = handler(arguments)
    except Exception as exc:
        result = json.dumps({"error": str(exc)})

    return [TextContent(type="text", text=result)]


# ── Entry point ───────────────────────────────────────────────────────────────

async def run():
    """Start the MCP server with stdio transport."""
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


def main():
    """CLI entry point for the Tessera MCP server."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
