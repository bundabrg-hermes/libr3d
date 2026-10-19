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

**Ported and oracle-green on the card (2026-10-19):** embed_lookup, rmsnorm, kv_store,
act_mul (all four acts), cast (bf16 copy / widen / narrow), rope (neox+gptj; multi-component
modes and yarn/llama3 refused by constraint), gemm_nt (bf16, correctness-first tile),
attn_paged (decode + windowed, correctness-first).
Remaining Tier 1: the sampler chain, then the MiniCPM5-2B bf16 serve test. WMMA speed
variants of gemm_nt are a Tier-1.5 optimization, not a gate.

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

## Verified state on hardware (gfx1100, RX 7900 XTX class, rdna3-dev ROCR index 0)

2026-10-19, radiance-dev:v1.8.1 (radiance v1.8.1, ROCm 10.1):

- **Load gate:** `rad-info --plugins` on `--home build3/radiance_home` reports
  `libr3d 0.1.0 kernels 3 kernel(s), 3 schema(s), built for gfx1100 gfx1101 gfx1102 gfx1103
  gfx11-generic` against `AQL backend, 1 device(s) [0:gfx1100]`, exit 0. The fat binary's
  gfx11-generic code objects load on gfx1100.
- **Oracle gates, all vs libref on the card over the 741-case fixture recorded from
  qwen3.8-27b-fp8:**
  | kernel             | result | red-zone |
  |--------------------|--------|----------|
  | rmsnorm_bf16       | 90 checked, 90 passed, worst rel_l2 2.395e-04 | 40 cases, 0 OOB |
  | embed_lookup_bf16  | 20 checked, 20 passed, worst rel_l2 0.000e+00 | 10 cases, 0 OOB |
  | kv_store_bf16      | 20 checked, 20 passed, worst rel_l2 0.000e+00 | 0 cases (draw tier), 0 OOB |
  | act_mul_bf16       | 44 checked, 44 passed (silu/gelu families) | 0 OOB |
  | cast_bf16_* (3 rows) | replayed with act_mul: 76 passed in that run | 0 OOB |
  | rope_bf16          | 17 device-rope cases, all passed in the 221/221 sweep | 0 OOB |
  | gemm_nt_bf16       | 269 checked, 269 passed, worst rel_l2 0.000e+00 | 28 cases, 0 OOB |
  | attn_paged_bf16    | 138 checked, 138 passed, worst rel_l2 0.000e+00 | 69 cases, 0 OOB |

  gemm_nt ran alone at 269/269 **bit-exact** against libref across every recorded geometry
  (qkv / gate+up / down / lm_head shapes to 2^27). The zero is structural, not luck: the
  kernel walks k ascending per output — the oracle's exact chain — and bf16 x bf16 is exact
  in f32, so the tile-boundary reassociations the kernel *could* have made it never makes.
  Same rule as the copy-shaped ops: a future nonzero rel_l2 here is a regression signal, not
  rounding noise.

  Full sweep `--op rmsnorm --op embed_lookup --op kv_store --op rope --op act_mul --op cast
  --max-bytes 12288`: **221 checked, 221 passed, worst rel_l2 2.395e-04, 0 wrote outside a
  red zone.** `rad-info --plugins` now reports 8 kernels / 6 schemas.
  Exact zeros are what the copy-shaped ops should give: a bf16 gather/scatter that matches the
  oracle bit-for-bit cannot round differently. Any future nonzero there is a regression signal.

  attn_paged_bf16 ran at **138/138, all bit-exact**, after fixing a thread-local score max:
  only thread 0 updated it, so threads 1..head_dim-1 saw "no visible keys", took the zero-row
  path, and stored zero columns. The repro harness `tools/repro_attn.hip` named it in one run;
  the same class of bug as the rope store-address race noted under Pitfalls below.
  Like gemm_nt the zero is structural: max-subtract then sum in the oracle's order,
  so no reassociation to disagree about.

  Final full-fixture sweep (all ops, default budget): **881 checked, 881 passed, 24 skipped,
  164 red-zone cases, 0 violations.** The 24 skips are the 7-14 GiB embed/logits_gemm cases
  that did not fit -- see the RAM pitfall below. `rad-info --plugins`: 9 kernels / 7 schemas.

- **Sampler chain + band ops (2026-10-19, same fixture/card).** `sample_temp/mask/argmax/topk/
  topp/minp/pick` and `add/gather_rows/quant_act_fp8/gated_quant_fp8` joined after the 881
  sweep. The quantisers are unavoidable even for a bf16 trunk: the model quantises MLP
  activations per 128-group and those ops are weightless, so the planner cannot push them to
  the host and refuses the graph at startup. fp8 here is *storage*, not arithmetic -- only the
  e4m3 encoder is touched, which RDNA3 does not need hardware for. Per-op runs: 20 + 16 + 88
  + 20 + 28 + 14 x 6, **all passed, all worst rel_l2 0.000e+00, 0 red-zone violations.**
  Full-fixture regression after the fixes: **1009 checked, 1009 passed, 24 skipped (same
  big-shape RAM skips), 0 violations.** `rad-info`: 21 kernels / 19 schemas on `[0:gfx1100]`.
- **Pitch pitfall (the one that ate a session):** `RadTensor::stride[0]` is in ELEMENTS. Any
  kernel that walks a `uint8_t*` row base as `base + row * pitch` must be handed a BYTE pitch;
  `row2_pitch` stays element-unit and each launch converts with `rad_dtype_bytes`. f32/f8
  (1 elt == 1 B) and M==1 (row 0, offset 0) are immune to the mix, which is why the failure
  signature -- "bf16 with rows>1" -- is the tell. A write overrun trips the red zone loudly;
  a pure READ overrun (gated_h taking `gub - gu` as an element base) only shows as a fatal
  queue fault mid-sweep. Pass row bases as pointers so the mix is unrepresentable.
- **Not yet done:** no full-model serve has run on this card. An interrupted session's closing
  report claimed MiniCPM5-2B was serving on the 7900 with embed fixes at commit `0949f0f` --
  that hash exists in no repo, the weights were never downloaded, and no serve artifact exists
  on either machine: the wrap-up was hallucinated at interruption. Treat that claim as the
  lesson it is (reconcile closing reports against git and the filesystem before believing
  them), and the gate below as genuinely open. The first run of MiniCPM5-2B bf16
  (weights are on neither box yet) is the remaining Tier-1 exit test; op coverage says the
  graph should resolve end to end.

- **RAM pitfall (rdna3-dev):** co-tenant serve's host-resident model holds ~25 GB of rdna3-dev's 31 GB, so the
  kbench auto budget drops to ~1.7 GiB and big embed/logits cases skip. Forcing them with
  `--max-bytes 12288` while co-tenant serve runs OOM-kills the container (exit 137, no dmesg line --
  `ps aux --sort=-rss` is how to spot it). Either run big sweeps while co-tenant serve is down, or
  re-run the ops that previously passed at 12288 (embed) with the auto budget and accept skips.

Reproduce (on rdna3-dev, after `cmake -B build3 -S . && cmake --build build3`):

```
docker run --rm --shm-size=4g --device /dev/kfd --device /dev/dri --group-add video \
  -v $PWD:/work -e ROCR_VISIBLE_DEVICES=0 radiance-dev:v1.8.1 bash -lc '
  mkdir -p /tmp/home/kernels /tmp/home/architectures &&
  cp /opt/radiance/share/radiance/kernels/libref.so \
     /work/build3/radiance_home/kernels/libr3d.so /tmp/home/kernels/ &&
  rad-kbench --home /tmp/home --fixture /opt/radiance/share/radiance/kernels.rkb --op rmsnorm'
```

Pitfalls this settled:
- `rad-info --plugins` takes the home as `--home PATH`; a `RAD_HOME` environment variable is
  silently ignored and the output lists the install's plugins only — an empty grep is not a
  load failure.
- A loaded kernel plugin must declare the schema of every op it implements; libref is a
  kbench oracle, not a search-path plugin, so on RDNA3 libr3d declares its ops' schemas
  itself, copied verbatim from libr4d so co-load `schema_same()` passes.
- The search path requires `architectures/` to exist beside `kernels/` (empty is fine).
- On rdna3-dev the 7900 is ROCR index **0** (`[0:gfx1100 1:gfx1201 2:gfx1201]`), not DRM card1.
- kbench needs ~3 GiB free in the container; case budget auto-shrinks and names skips.
- `--max-bytes` is plain MiB (`12g` parses as 12 MiB). The 27B fixture's embed_lookup cases need
  ~7.1 GiB each; `--max-bytes 12288` runs all 20 on the 24 GiB card.
- A kbench PASS can be vacuous: a case whose positions are 0 rotates by the identity, so an
  addressing bug in rope hid there while every sampled-position case failed. Prefer a repro
  with unit impulses (`tools/repro_rope.hip`, `UNIT=1`) -- the output element then names the
  angle the kernel actually used.

## The rope race (2026-10-19), worth remembering

The first rope port dropped the head axis from the store address (`xr = row + h*head_dim`):
every head's threads read-modify-wrote head 0's pairs, compounding rotations into
`cos(k*ang)` with k set by the scheduler -- large, deterministic-looking, slightly
run-stable rel errors that grew with M. Neither the row-stride fold (ref_gemm.h `row_stride`'s
trailing-dimension walk, kept: rank-3 `[M, heads, head_dim]` qkv is expressible) nor the
position-stride read (ref's `offlin`, kept) was implicated; the fixture descriptors were plain
rank-2 stride-1. The decisive tools were the standalone repro and a direct compile of the
kernel outside the plugin .so, which together split "plumbing vs code" in one step.
