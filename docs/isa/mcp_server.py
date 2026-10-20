#!/usr/bin/env python3
"""MCP server exposing the RDNA3 ISA corpus (FTS5 + vector hybrid search).

Run with the venv python, e.g.
  docs/isa/.venv/bin/python docs/isa/mcp_server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from isasearch import ISA  # noqa: E402

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as Server  # noqa: E402
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server  # noqa: E402

mcp = Server("rdna3-isa")
isa = ISA()


@mcp.tool()
def isa_search(query: str, k: int = 5) -> str:
    """Hybrid search over the AMD RDNA3 ISA reference (609-page spec).

    Handles both exact mnemonics (V_ADD_F32, buffer_store, ds_swizzle...) and
    conceptual questions (how do I broadcast a value across a wave).
    Returns ranked passages with corpus file paths and PDF page refs."""
    res = isa.search(query, k=max(1, min(k, 20)))
    if not res:
        return "no results"
    out = []
    for d in res:
        head = f"### {d['mnemonic'] or d['path']}"
        meta = f"path={d['path']} format={d['format']} section={d['section']} match={d['src']}"
        text = d["text"] if d.get("text") else (d.get("snippet") or "")
        out.append(f"{head}\n{meta}\n{text[:2000]}")
    return "\n\n---\n\n".join(out)


@mcp.tool()
def isa_get(mnemonic: str) -> str:
    """Full ISA reference entry for one instruction, all encoding variants.

    Accepts doc names (V_PERMLANE16_B32) or LLVM spellings with _e32/_e64
    suffixes (v_add_f32_e32). Returns complete markdown incl. pseudocode."""
    docs = isa.full_text(mnemonic)
    if not docs:
        return f"no entry for {mnemonic!r}; try isa_search"
    return "\n\n---\n\n".join(f"### {d['path']}\n\n{d['text']}" for d in docs)


@mcp.tool()
def isa_find(substring: str) -> str:
    """List instruction mnemonics containing a substring (case-insensitive).

    Example: 'permlane' -> all V_PERMLANE* mnemonics with their format."""
    rows = isa.conn.execute(
        "SELECT DISTINCT mnemonic, format FROM docs WHERE upper(mnemonic) LIKE ? "
        "ORDER BY mnemonic LIMIT 80", ("%" + substring.upper() + "%",)).fetchall()
    if not rows:
        return "no matches"
    return "\n".join(f"{r['mnemonic']} ({r['format']})" for r in rows)


@mcp.tool()
def isa_section(query: str, k: int = 5) -> str:
    """Search the narrative chapters (encoding formats ch15, WMMA ch7.9, operand
    rules, clauses, LDS, memory model) — excludes per-instruction entries."""
    res = isa.search(query, k=max(1, min(k, 10)), kind="narrative")
    if not res:
        return "no results"
    return "\n\n---\n\n".join(
        f"### {d['path']}\n{(d.get('snippet') or '')[:2200]}" for d in res)


if __name__ == "__main__":
    mcp.run()
