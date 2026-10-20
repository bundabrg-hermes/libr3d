# docs/isa — AMD RDNA3 ISA reference corpus (searchable)

Machine-friendly split of the official **"RDNA3" Shader
Instruction Set Architecture Reference Guide** (GPUOpen, 15-Aug-2023
revision, 609 pp.) into 1,464 per-instruction markdown files and 234
narrative section files, plus a hybrid lexical+vector index.

## Layout

```
inst/<mnemonic>.md        one file per instruction × encoding (frontmatter:
                          mnemonic, format, opcode, PDF pages)
inst/x__<fmt>.md          same instruction described for a second encoding
                          (e.g. v_add_f32.md = VOP2, v_add_f32__vop3.md)
sections/<slug>.md        chapters 1–15 + family intros (encoding tables,
                          WMMA 7.9, clauses, LDS rules, microcode formats)
INDEX.md                  full corpus map
isa.db                    FTS5 + sqlite-vec index (regenerable, gitignored)
source/rdna3-isa.pdf      pinned source (sha256 recorded in isa.db meta and below)
scripts/                  parser, index builder, search library, CLI
mcp_server.py             MCP stdio server (isa_search/isa_get/isa_find/isa_section)
isa                       CLI wrapper
```

## CLI

```bash
docs/isa/isa get V_PERMLANE16_B32       # full entry (LLVM _e32/_e64 names accepted)
docs/isa/isa search "wave broadcast lane"
docs/isa/isa find PERMLANE
docs/isa/isa list --format VOP3P
```

## Search design

- Router: queries containing `NAME_WITH_UNDERSCORE` tokens go lexical-first
  (exact mnemonic → FTS5 phrase over underscore-split mnemonics);
  conceptual queries go vector (BAAI/bge-small-en-v1.5); mixed queries are
  fused with RRF.
- Mnemonics are indexed both raw and underscore-spaced, so `v_add_f32_e32`
  still resolves to doc name `V_ADD_F32`.

## MCP (Hermes)

Registered as server `rdna3isa`:

```bash
hermes mcp add rdna3isa \
  --command /home/hermes/projects/libr3d/libr3d/docs/isa/.venv/bin/python \
  --args /home/hermes/projects/libr3d/libr3d/docs/isa/mcp_server.py
```

Tools appear as `mcp_rdna3isa_*` after a Hermes restart.

## Rebuild

```bash
cd docs/isa
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python pymupdf fastembed sqlite-vec mcp
.venv/bin/python scripts/parse_isa.py     # PDF -> markdown corpus
.venv/bin/python scripts/build_index.py   # corpus -> isa.db
```

Source: https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf
SHA-256: d87bdd09e85839e86c1374b5485f76c57cbb81a3bfbd4eb8a5ab197509151a72
(TechPowerUp's copy is an identical mirror of this file.)
