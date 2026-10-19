# libr3d coverage plan

Sources: radiance v1.8.1 — `docs/PLUGIN.md` §10, `docs/OPS.md`, `arch/common/*` (the op set a
bf16 dense model issues), `core/` (the sampler ops the engine issues), `libref/` (the oracle).

## Ground rules

- Every op in the conventional vocabulary has a libref host fallback and a kbench oracle. Every
  kernel we ship is checked against libref with `rad-kbench` on recorded geometries before it is
  considered done.
- Ops `docs/OPS.md` marks **device-only** have no oracle; they are deferred and will need
  hand-written validation.
- Partial coverage is legal: unresolved ops are errors only if issued. An incomplete libr3d
  simply doesn't serve models whose op set it hasn't reached.
- RDNA3 has no native fp8 (E4M3) hardware. fp8/w4a8/fused-quant rows are out of scope until the
  bf16 path works; when they come, expect conversion-at-boundary designs, not emulated MMA.

## Tier 0 — load smoke (this skeleton)

Zero rows. Plugin loads on gfx11, ABI 22 accepted, selection unchanged.
Check: `rad-info --plugins` on the card.

## Tier 1 — bf16 minimum viable model (goal: decode a 2B-class bf16 model on RDNA3)

Model ops issued by the bf16 dense path (`arch/common/rad_block_attn*.h`, `rad_block_mlp.h`,
`rad_arch.h`):

| op            | role                                   | notes |
|---------------|----------------------------------------|-------|
| embed_lookup  | token -> row                            | trivial gather |
| gemm_nt       | qkv / gate+up / down / lm_head          | the big one: wave32 WMMA bf16 on gfx11 |
| rmsnorm       | pre-attn / pre-mlp / final              | |
| rope          | rotary positions                        | neox + default variants |
| attn_paged    | decode + prefill attention              | second big one; per head_dim/gqa geometry |
| kv_store      | append to paged KV cache                | |
| act (silu_mul)| SwiGLU activation-multiply               | confirm exact op name/graph via --debug-graph |
| cast          | dtype edges                             | |
| all_reduce    | TP only                                 | Tier 2 (TP1 needs no collective) |

Sampler ops issued by core (`sample_*`), ~11 in the default chain:
`sample_temp, sample_topk, sample_topp, sample_minp, sample_typical, sample_penalties,
sample_dry, sample_xtc, sample_mask, sample_merge_topk, sample_pick, sample_argmax,
sample_logprobs` — exact served subset to be confirmed with `--debug-graph` on the card.

## Tier 2 — multi-rank and shapes

`all_reduce` / `all_gather` (+ `_host` twins — consumer P2P needs them even on RDNA4, so the
design is proven), wider head_dim/gqa geometries, prefill tiles.

## Tier 3 — quantised vocabulary

fp8 GEMM families via boundary conversion (OCP E4M3 bytes per the core's convention), i8a8,
w4a16-style weight formats that avoid fp8 MMA. Device-only fused ops: last, with bespoke tests.

## Verification loop (per kernel)

1. `docker` build of libr3d against pinned radiance (no card).
2. Host-tier tests (`ctest -LE 'gpu|oot'` upstream patterns; kbench fixture tier).
3. On rdna3-dev with the RDNA3 card: `rad-kbench` vs libref over recorded geometries, red-zone OOB;
   then `rad-info --plugins`; then serve MiniCPM5-2B bf16.
