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

## Serve on the 7900 + decode GEMV (2026-10-19, later)

Supersedes the "no full-model serve has run" bullet in the section below. MiniCPM5-2B bf16 serves on
the 7900 (`libr3d-serve` container, radiance-dev:v1.8.1, :9005, `--radiance-home
/rhome:/opt/radiance/share/radiance` with `build/radiance_home/kernels/libr3d.so`).
Measured decode: **0.35 -> 11.1 tok/s** (96-token request), warmup 22.5 s -> 0.71 s;
text coherent end to end. Two findings did that, one kernel and one contract:

1. **The `M` parameter is arena capacity, not the row count.** The engine passes
   `M=8192` (the `--max-num-batched-tokens` step-batch capacity) on every call while Y
   carries the live rows: `[1,N]` at decode, `[12,N]` at prefill. `ref_gemm.cpp` never
   reads the parameter -- it derives `M = numel(Y) / N`. A row that trusts the parameter
   computes a full prefill batch on every decode token *and writes garbage into arena
   rows past Y's own* -- invisible to every guard, because the arena is exactly that
   big. libr3d's gemm entries now mirror ref's derivation. This -- not bandwidth -- was
   the whole 0.35 tok/s: profile-ops showed decode calls taking the same 34 ms as
   prefill calls at identical geometry.
2. **Decode GEMV row** (M==1 band) in `r3d_gemm_bf16.hip`: one wave32 per output row,
   uint4 (8 x bf16) vector walks, `shfl_xor` butterfly reduce, flat `[row]` store
   (`y_pitch` is the between-M-rows pitch and M==1 has none -- scaling by it was a
   1.4 MiB OOB write fault kbench caught on the first case). The tile kernel keeps
   every other M. Guards: K%8, weight pitch %8, 16 B-aligned bases, else tile.

Two more lessons, both from kbench on this row:
- **Wave width is a project-wide ABI pin.** `-mno-wavefrontsize64` is set per TU in
  CMakeLists; a wave64-shaped shuffle reduce (lane=tid&63, offsets 32..1) fails with
  rel_l2 ~= 1.0 on exactly the dispatched cases and nothing else. Derive lane math
  from the pinned width, not the hardware default.
- **kbench must own the card.** Running it while the serve container held the 7900
  produced 28 false `refused ... device error` FAILs (queue faults from contention);
  a solo re-run of the identical build was 269/269. Check `docker ps` before believing
  a fault-laden sweep.

GEMV reassociates through the butterfly, so M=1 cases now land at worst rel_l2
4.48e-08, not the tile's structural 0 -- that line is the GEMV's expected signature;
the M>1 bit-exact zeros remain the regression tripwire. logits_gemm's seven 7.1 GiB
fixture cases stay budget-skipped while co-tenant serve holds 25 GB of 31 GB host RAM; the
GEMV serves them live in the model (decode steps at `y=[1,130560]`, coherent output).

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
  | kv_store_bf16      | NOT DEVICE-VERIFIED (see correction below: fixture is fp8-KV only) | - |
  | act_mul_bf16       | NOT DEVICE-VERIFIED (see correction below: fixture has no act_mul cases) | - |
  | cast_bf16_bf16     | reached in per-op runs (10 device cases); bf16→f32 / f32→bf16 pairs have no fixture cases | 0 OOB |
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

## The coverage correction (2026-10-19, second round)

`logits_gemm_bf16` joined (the lm_head edge: gemm_nt's tile under the `n_vocab`/`n_embd`
spelling; the tile's Y gate already admitted the f32 logits plane). Adding it and reading the
full-sweep report exposed what the earlier per-op "passes" had hidden. The fixture records
`op -> rows run`, not just pass counts, and per-op runs had been read as "our row ran" without
checking that line. It had not, for four rows:

| row | fixture reality | status |
|---|---|---|
| `kv_store_bf16` | every case is `kv_dtype=fp8_e4m3` (qwen3.8-27b-fp8 serves fp8 KV) | compiled + loads; never device-verified |
| `act_mul_bf16` | no `act_mul` cases at all (this deployment fuses act+quant into the bands) | compiled + loads; never device-verified |
| `cast_bf16_f32` / `cast_f32_bf16` | only `from=bf16 to=bf16` recorded | bf16→bf16 pair IS verified; the f32 pairs are not |
| `logits_gemm_bf16` | `lm_head` recorded as affine E4M3: 2 planes (codes:f8e4m3 + scale:f32) | compiled + loads; never device-verified |

Earlier claims of "kv_store 20/20 bit-exact, act_mul 44/44 passed" were libref replaying its
own recorded answers: our rows had been declined, and the "0.000e+00" was libref-vs-itself.
The decline mechanism, from `core/plugin/select.cpp`:

- A DEVICE row with no `layout` hook **"reads the weight as stored"**: `weight_refusal()`
  declines it whenever the case's weight encoding spans >1 plane. The engine would otherwise
  hand an fp8 codes plane to a kernel reading bf16 — plausible-magnitude garbage. libref's
  host shim dequantises, so the cases still check, vacuously for us.
- Constraint declines are honest geometry: fp8 KV and fp8 heads are not this row's contract.
- Probe that settled it (kept in `tools/kf_probe.cpp`): loads the fixture via
  `rad::Kfixture::load` and dumps each operand's encoding. Link it against
  `tools/kfixture.cpp` with `-Wl,--unresolved-symbols=ignore-all` (the draw helpers it
  drags in are never called by `load`).

Consequence, stated plainly: the qwen3.8-27b-fp8 fixture CANNOT gate a bf16-head or bf16-KV
row. Those four rows graduate on the serve milestone — record a fixture from the bf16
container (`rad-kbench -m minicpm5-2b... --record FILE`) and they resolve against plain bf16
planes for the first time.

Hygiene adopted from this:
- Read the report's "not reached: libr3d N" line in EVERY sweep; a pass count says nothing
  about which library ran. Coverage ratchet: `--coverage DIR --coverage-write` locks the
  reached set (22 libr3d rows live in `~/projects/libr3d/logs/coverage/` on rdna3-dev); a later
  `--coverage DIR` run fails if any locked row silently drops out.
- Big sweeps while co-tenant serve runs (25 GB host RSS of 31 GB): a temporary swapfile works —
  `sudo fallocate -l 20G /swap-kbench && sudo chmod 600 /swap-kbench && sudo mkswap
  /swap-kbench && sudo swapon /swap-kbench`, run with `--max-bytes 12288`, then `swapoff`
  and delete. Verified: all 7 logits cases passed as libref self-checks that way at ~50 s
  wall without OOM.

## Prefill-logit KLD gate — `tools/kldgate/` (2026-10-20)

Why: stew675/llama-cpp-rdna-boosts#113. A default-on bf16 prefill kernel shipped with mean
KLD 0.03 past every text/coherence gate: greedy and sampled text read the decode distribution
*from the post-prefill state* and are blind to a uniform prefill shift. Their doctrine, worth
adopting whole: the model-level KLD is the alarm, the per-op parity is the truth (the same
per-op perturbation spans >100x in model KLD depending on the surrounding pipeline), and the
gate arms are base-vs-prior-arm, not base-vs-rebuilt-oracle.

Design, adapted to radiance (no perplexity tool, no prompt_logprobs over HTTP): two serve
arms on the same card and weights, one at a time, A/B'd through the llama.cpp-style
`/completion` with `temperature 0, n_probs 16 (engine cap; 32 is a 400), post_sampling_probs
false`. The base arm (scalar prefill tile via `R3D_NO_WMMA=1`) free-runs the 11-prompt corpus;
the test arm is then asked to *re-score the base's trajectory under teacher forcing*
(`score` mode): position i is served the exact text prompt + base tokens 0..i-1, so every
compared position shares identical context. The forced-context rebuild is byte-exact (token
`bytes` arrays through an incremental UTF-8 decoder) with a `/tokenize` count guard — a
retokenization desync truncates that trajectory and counts the drop rather than comparing
mismatched text. The prefix cache makes each forced request incremental, so the pass costs
~190 ms/position.

Gate: mean KLD <= 2e-3 AND p99 <= 5e-3 AND top-1 agreement >= 0.98. Max is informational:
one razor-tie position amplifies through the union-KL renormalize (observed 0.029 at a single
position; the same-arm noise probe on that corpus maxes at 0.0016), and the defect class is a
*mean* shift, not an outlier. Noise floor of the harness itself (same arm scored against
itself): mean ~2e-4, max ~6e-3 — serve-to-serve variance of fresh vs incremental prefill
under the prefix cache. The gate sits an order of magnitude above the floor; #113-class sits
one above the gate.

First run through the gate (WMMA m64 tile vs scalar base, 1257 positions): mean 3.1e-4,
p99 2.25e-3, top-1 0.993 -> PASS. About 9 forced-context argmax flips across the corpus, all
near-ties (-1.9318 vs -1.9292 class): rounding-order purity delta, not drift. Artifacts:
`/tmp/kldgate-20261020-083741/report.md` on rdna3-dev.

Lessons, kept because they were paid for:

- Comparing free-running arms by position index is invalid the moment the trajectories part:
  the first (broken) run reported mean KLD 2.84 — distributions conditioned on *different
  texts*. Position-0 forensics settles it in one probe: identical top-16 overlap with
  different argmax means the instrument is comparing apples to oranges, not that the kernel
  drifted.
- The same-serve self-test (`collect` then `score` against one running arm) is the instrument's
  self-check: it must return ~0 KL and few dropped positions. It caught a bytes/str plumbing
  bug on the spot and defines the noise floor the thresholds must clear.
- radiance HTTP: `n_probs <= 16`; `prompt_logprobs` is accepted and ignored on both
  `/v1/completions` and `/completion` — full-vocab prefill KLD needs it upstream (worth
  requesting next time we talk to the radiance maintainers, alongside the preemption report).
- Retokenization desync at BPE join boundaries is real (the mid-expression `cpp-kernel`
  prompt desyncs at position 13): guard, drop, count — never score a position whose token
  identity is not verified.
- A GPU container's `docker rm -f` can lag its kill by seconds (device teardown); `docker
  run` in the next breath conflicts. Wait for `docker inspect` to report the name gone.

## Serve admission cap: --max-num-seqs 8 (2026-10-20)

The serve recipe ran with radiance's default `--max-num-seqs 256` against a KV pool that
backs 27.9 full-length sessions, and radiance does not preempt gracefully (V3/V4 overload
wedged the engine + GPU). Capping admission at 8 was meant as pure safety. It moved real
memory and real speed:

- KV worst case: 11% backing (256 x 16384) -> **100% backing** (8 x 16384); oversubscription
  is now structurally impossible. The 18.31 GiB pool itself is VRAM leftover and unchanged.
- Per-step fixed structures scale with seq slots: pinned host staging ring 51.83 -> 2.27
  MiB; device slab 17.24 MiB -> 772.5 KiB (block tables 17.0 -> 0.5 MiB).
- **Decode: 11.1 -> 23.6-27.4 tok/s** sustained (500/369-token greedy probes; the 40-token
  probe read 58.9, short runs flatter). Decode cost had been dominated by per-step fixed
  work scaling with the step-batch arena, not by the GEMMs — a second confirmation of the
  M-parameter lesson: arena capacity leaks into timings wherever the engine trusts it.

`SERVE_ONLY=1 bash tools/kldgate/run.sh` now single-sources the birth recipe. Note when
benchmarking old numbers: anything measured before this flag carries a ~2x decode handicap.

## Prefill bottleneck is attention, not GEMM (2026-10-20, two-probe fit)

Measured on the live serve, fresh (unwarmed) prompts, `max_tokens=1`, seqs8 build:

| prompt tokens | TTFT |
|---|---|
| 981  | 2.85 s |
| 1961 | 10.68 s |

The 3.75x for a 2x prompt is the quadratic signature. Fit `T = a*M + b*M^2`:
a = 0.365 ms/token (linear), b = 2.59 us/token^2 (quadratic). At M=981 that splits
2.49 s quadratic / 0.36 s linear -- 87% of prefill is the attention term, and the
linear term (all GEMMs, ~3.9 TFLOP) already runs at ~11 TFLOP/s effective. The
quadratic term runs at ~133 GFLOP/s: our `attn_paged` device kernel is correctness-
first scalar FMA (grid = row x head blocks of head_dim threads, each block rescans
the whole KV history, no tiling, no tensor units).

Rejected hypotheses, in order: (1) host AVX attention was ruled out by `docker stats`
during a 10.7 s prefill: 0.4-0.8% CPU. (2) The engine's `--profile-ops` dumps nothing
to docker logs (table goes elsewhere) and its synchronisation distorts what it does
measure; the two-point scaling fit settles shape questions in 30 s with zero rebuilds.

Consequences for the queue: WMMA flash-attention moves to first (it was third); the
GEMM staging pipeline drops to second -- it optimizes the term that is 13% at 1k ctx.
It also retroactively explains the WMMA-vs-scalar GEMM tie: a tile swap inside the
13% term cannot move a total dominated by the 87% term.

Baseline for the FA work: 2.85 s @ 981 tok (fresh context; a prefix-cached repeat is
not a valid prefill baseline).

## WMMA flash-attention shipped: 2.1-3.1x prefill (2026-10-20)

The two-pass WMMA tile (`r3d_attn_paged_wmma.hip`) is live on the serve. Measured
fresh-context TTFT, same seqs8 recipe:

| prompt tokens | v1 scalar | WMMA tile | speedup |
|---|---|---|---|
| 981  | 2.85 s  | 1.33 s | 2.14x |
| 1961 | 10.68 s | 3.46 s | 3.09x |

Decode also moved 24-27 -> ~37 tok/s (the tile serves M==1 rows too: same bytes,
tensor cores make the wasted FMAs free). kbench attn 138/138 worst rel_l2 3.135e-03
(v1 arm unchanged at 0.000e+00); full sweep 1102/1102, 7 budget-skips, no collateral.

### The bug that hid the tile for two sessions: numeric bf16 staging

`bf16(uint16_t)` on `rocwmma::bfloat16_t` CONVERTS the integer instead of reinterpreting
its bits: the fp8 decode of code 0xf1 (-144, bits 0xc310) staged as bf16(49936) = 49920.
Every Q/K/V/P fragment staging site poisoned its own tile; scores hit 1e11, outputs 8e8.
Sibling of the known trap (casting a bf16 fragment TO short truncates numerically).
Rule: fragment staging writes raw BIT patterns (`bf16_store`), never a constructed bf16.
Checked the gemm m64 TU: it stages uint16 directly, clean.

### Why the failures read as "passes" for so long: the vacuous fallback

Two layers of silence stacked up:

1. A debug build added device `printf` to this kernel; printf + a 41 KB static-LDS
   launch fails with hipErrorIllegalState(401) on ROCm 10.1 (probe: launch alone is
   fine at 41 KB; adding printf alone makes it 401). The launcher turned the launch
   error into RAD_E_DEVICE, the entry fell back to v1, every case served bit-exact --
   a "pass" that never ran the tile. Device printf is not usable in big-LDS kernels;
   stage intermediates to a global dump buffer instead (repro `REPRO_DBG=1` arm).
2. In kbench the fp8 rows failed while the bf16 rows "passed" -- but head_dim=256
   fixtures are refused by the tile gate (v1 serves them, rel 0.000), so a bf16 pass
   in the report does not prove the WMMA tile ran. Check for nonzero rel_l2 in the
   WMMA arm vs 0.000 in the R3D_NO_WMMA arm before believing a WMMA pass.

Triage tools added: `tools/probe_l401*.hip` (launch-state bisect by build axis:
arch list / dlopen / LDS size) and `tools/probe_fa_main.cpp` (launch the real tile
through its own launcher, standalone).

## 2026-10-20: gemm m64 k-slice pipeline (DBUF) — GEMM staging retarget

After the attention fix the prefit quadratic moved to the GEMM linear term's stall:
the m64 tile serialised stage -> barrier -> WMMA -> barrier per k-slice, so every
slice ate the full operand global-load latency. Ported the structure of
llama-cpp-rdna-boosts block 06 (MMB, gfx1100-validated): next slice prefetched
global->registers during this slice's WMMA, A staging double-buffered (5 KB total
LDS, occupancy unchanged), B deliberately NOT double-buffered -- block 06 measured
B's second buffer as a loss (extra LDS traffic exceeds the barrier it removes).

Measured on the card, same binary, env switch (R3D_GEMM_NO_DBUF=1 = old schedule),
unique uuid prefixes, fresh serve each arm:

| probe          | single-buffer | DBUF pipeline | speedup |
|----------------|---------------|---------------|---------|
| 981 tok median | 0.356 s       | 0.289 s       | 1.23x   |
| 1961 tok median| 0.691 s       | 0.550 s       | 1.26x   |

Distributions do not overlap (dbuf max 0.300 < single min 0.327 @981). kbench
gemm_nt 269/269 both arms, worst rel_l2 identical to 4 digits (5.164e-04): the
schedule moves LDS write timing only, arithmetic is bit-identical, which is why no
separate KLD run was needed. `R3D_DEBUG_GEMM=1` prints the schedule name at launch
(the prove-the-kernel-ran line); logits_gemm fixtures stayed budget-skipped (the
host-radiance co-tenant holds the RAM) but share gemm_nt's launch path verbatim.

Side observation: today's re-measured baselines (0.36 s @981) run ~4x under the
historical post-FA note (1.33 s) -- co-tenant and box state differ from that
session; another reason A/B must be same-day paired runs, never historical deltas.

## 2026-10-20: the WMMA attn tile is latency-bound, and long-ctx decode collapses

Serve-side measurement (MiniCPM5-2B fixture, 42 layers, full-attention, KV 1024
B/token, --bench kbench timing table from the shipped fixture pack):

| ctx (tokens) | decode rate | decode step | prefill TTFT | prefill quadratic share |
|--------------|-------------|-------------|--------------|-------------------------|
| ~30          | 48.2 tok/s  | 21 ms       | -            | -                       |
| 2048         | 11.2 tok/s  | 89 ms       | 0.578 s      | ~0.06 s (10%)           |
| 4075         | -           | -           | 1.284 s      | ~0.20 s (16%)           |
| 8200         | 3.1 tok/s   | 323 ms      | 3.671 s      | ~1.38 s (38%)           |

Host CPU 0.1% during the 8k decode (docker stats): every ctx-proportional ms is
DEVICE time -- no AVX fallback anywhere in this. Decode step is linear in ctx
(0.038 ms per ctx token); the prefill quadratic is the KV re-read per 64-row
query tile (~M/2 x ctx KV bytes), and the observed 1.38 s at 8k implies ~15 GB/s
for those reads against a card doing 500-700 GB/s cold.

kbench `attn_paged --bench --timing-only` attributes it exactly. The tile row
(attn_paged_bf16, confirmed reached) sustains, at steady deep-ctx cases:
hd=128 rows 2.6-15 GB/s; hd=256 rows 0.5 GB/s; and even the small-ctx prefill
cases never exceed ~500 GF/s -- ~100x under WMMA peak. Root cause is the tile
structure, not a fallback: one block per (head, query tile) -- a whole q_len=1
layer runs on only n_head blocks of 96 CUs; 32-key steps with ~4 __syncthreads
each; single-buffered K/V staging (load->barrier->mma->barrier fully serialised);
at hd=256 the Q+KV+S LDS carve forces one block/CU, matching its 30x deficit vs
hd=128. The old "2.1-3.1x from WMMA" claim stands only against the scalar
reference; the tile itself is the long-ctx bottleneck.

Fix directions (next milestone, ranked): (1) DECODE split-K -- the op schema
already carries a `splits` parameter; stream KV chunks in parallel blocks and
combine (flash-decode), turning 24 blocks into ~1000 and the 8k step from
323 ms toward its ~1 ms bandwidth floor; (2) PREFILL tile redesign -- kBN 128+,
double-buffered staging (gemm DBUF lesson applies), fewer barriers, softmax on
fragments where possible, occupancy budget set per head_dim.

MXFP4 feasibility note while we were in the vocabulary: radiance 1.8.1 loads OCP
MXFP4 natively (RAD_FP4E2M1; e2m1 codes + E8M0 scale per 32 K) and serves it at
op `gemm_nt_q` encoding `mxfp4a8` (fp8 e4m3 activations). `qwen3.8-27b-mxfp4.rad`
(19 GiB) is already on rdna3-dev, and the SHIPPED fixture pack kernels.rkb contains
recorded mxfp4a8 + fp8a8 gemm_nt_q cases -- the oracle loop for a Tier-3 tile
runs with zero model risk. Boundary-conversion design (e2m1*e8m0 -> bf16
fragments per k-slice, exponent add only) fits the existing m64 DBUF structure;
e8m0 scales are powers of two, foldable into the per-group accumulator scale.

## 2026-10-20: decode split-K shipped — deep-ctx decode 3.0 -> 14.5 tok/s

The fix direction above is now the decode band's default. Rows
`attn_paged_decode_bf16kv_h{64,128}` (priority 1100) take window 0, q_len <= 8:
the two-pass tile body, shared verbatim with the prefill row via
`r3d_attn_wmma_body.inc` under a `SPLITK` template parameter, runs once per KV
slice across `q_heads * n_seq * splits` blocks, stores un-normalized (O, m, l)
partials into the engine scratch arena, and a merge kernel (one block per
q head x query row) rescales them into the out row. Shape law on the host, one
closed form: `splits = min(192/(q_heads*n_seq), tiles_kv/2, tiles_kv, 64)`, `splits`
tunable 0 = the law. Kill switch `R3D_NO_SPLITK=1` runs the SPLITK=0 tail through
the same entry; `R3D_FORCE_SPLITS=N` pins the split count for benches.

Same-binary serve A/B (pair-differenced decode rate, `tools/decode_probe.py`,
co-tenant state shared by both arms):

| ctx    | tile arm  | split-KV arm | speedup |
|--------|-----------|--------------|---------|
| 2048   | 10.6 t/s  | 33.5 t/s     | 3.2x    |
| 8192   | 3.0 t/s   | 14.5 t/s     | 4.8x    |

The tile arm reproduces this file's own 3.1 tok/s measurement to the digit in
the same window — the baseline was not a drifted serve. The split arm still
runs 12 slices at 8k, not the ~1000 blocks the estimate imagined: the law caps
at tiles_kv/2 (a slice must hold >= 2 tiles) and 64, and q_heads*n_seq is the
grid floor — but 192 blocks over 96 CUs is already the difference between one
wave of latency chains and many. The bandwidth floor (~1 ms/step) remains open
ground for a bigger-kBN slice kernel; this is the 4.8x that the structure
supports without touching the tile body.

Verification, in the order that caught things: `--record-rows`-derived fixture
(`tools/attn_rows.rkb`, 16 KiB, regenerate with
`rad-kbench --op attn_paged --record-rows ...` from the shipped pack) replays
the h128/h64 decode rows at 8-split and passes 8/8 (the SHIPPED pack is
head_dim-256-only — every attn case in it, so a pass column from it alone
proves nothing about these rows: the vacuous-pass trap, again, and kbench
says so itself: `2 of 194 rows ran`). `tools/repro_attn_splitk.hip` then
separates mechanism halves: it runs the row entry on a poisoned arena and
RE-MERGES the partials host-side against an independent oracle — device-out
and host-merge rel_l2 must both be small; which one fails says which half lies.
It is what pinned two plane-base bugs the card sweep showed but did not locate
(m-plane base omitted the splits factor; l-plane omitted it too; writer and
reader had AGREED on the wrong layout, so only a third implementation on the
contract caught it). `R3D_FORCE_SPLITS` 1/2/8/64 all green:
probe 8/8, kbench rows 8/8 (worst 0.000), full sweep with both fixtures and
`--all-rows`: 1083/1083, worst rel_l2 unchanged at 3.135e-03, 24 budget skips
(co-tenant serve holds 29 GiB of the 31). Serve now runs from `build6/radiance_home`
(build5 retired) with split-KV as the default arm; `/completion` sanity reads
coherent.

Serve lesson for the day, worth keeping: the probe's first pair after a serve
start can read -518 tok/s (t8 > t108 while the engine warms its KV pool);
the tool reports every pair, and medians hide the sign of the nonsense rather
than deleting it. Discard cold-start pairs by eye, not by averaging.

## 2026-10-20: prefill v2 tile schedule — refuted, kept dormant as R3D_ATTN_V2

Direction (2) from the ranking above got its first attempt: `r3d_attn_wmma_body2.inc`
re-schedules the same tile — 256 threads (8 waves) instead of 128, global->register
prefetch of the next K/V slice overlapping the MMA step (the gemm DBUF lesson),
barriers cut from 4+6 per two passes to 2+3, bad-byte accounting moved strictly
after barriers. The refactor also moved the v1 body into the shared
`r3d_attn_wmma_body.inc` so both schedules and both bands (prefill, split-KV
decode) instantiate from one source.

Arithmetic identity held exactly as a schedule change must: the oracle sweep
(`tools/attn_oracle_sweep.hip`, 7 geometries x GQA/window/unmapped-edge variants)
prints identical rel_l2 AND identical worst-case bit distances per case on both
arms, the split-KV probe passes splits 2/8/64 on both arms, and `tools/attn_bench.hip`
hashes v1 and v2 outputs bit-identically on all five serve-shaped geometries.

The perf hypothesis did not. Same card, same binary, min over 15 reps:

| hd/qlen/ctx    | v1 us | v2 us | v2/v1 |
|----------------|-------|-------|-------|
| 128/64/1024    | 763   | 959   | 0.80x |
| 128/128/2048   | 1433  | 1853  | 0.77x |
| 128/981/8192   | 12010 | 21065 | 0.57x |
| 128/512/4096   | 2930  | 6859  | 0.43x |
| 64/512/4096    | 2002  | 2099  | ~1.0x |

Read: barrier COUNT is not the serialization term. The v2 block (256 threads,
40.75 KB LDS at hd=128) packs 1 block/CU, so every barrier stalls the whole CU
with nothing else resident; v1's narrower blocks put two independent barrier
chains per CU, and their overlap was worth more than the barriers v2 removed.
Register-prefetching the slice also trades LDS pressure for VGPR pressure at
256 VGPR — no headroom left to hide latency either way. The occupancy axis is
the same one the split-KV win rode; making the BLOCK bigger is the wrong end of
it. v1 stays the default; `R3D_ATTN_V2=1` runs v2 in the same .so for postmortem
(the promotion pattern inverted: opt-in instead of opt-out).

What direction (2) still needs, unchanged by this result: more BLOCKS at deep
prefill, not fatter ones — kBN 128+ slices and per-head_dim occupancy budgets,
ideally splitting query tiles across blocks the way decode split its KV.

Two harness bugs worth remembering, both caught the sweep before the kernel
ever ran: absent optional operands are value-initialized `RadTensor`s whose
`data` is NULL inside the full positional array (a garbage `rank` from an
uninitialized descriptor makes numel helpers walk past the struct — the bench's
host-side segfault — and a garbage descale pointer poisons every score, faking
a geometry-dependent kernel failure that was really stack reuse); and the sweep
gate must be rel_l2-based like kbench's, with bit-distance as diagnostic only —
the oracle sums in double, so cancellation-heavy near-zero elements legitimately
sit thousands of bit-lines apart.

## 2026-10-20: prefill v3 thin block — refuted; staging is instruction-rate-bound (vectorized: 1.2-1.64x)

Two experiments rode one commit window. (a) `r3d_attn_wmma_body3.inc` (opt-in
`R3D_ATTN_V3=1`) tested the recorded direction directly: 32-row query tiles,
64 threads, 24,880 B LDS -> 2 blocks/CU at hd=128 where v1's 41 KB carve fits
one. Per-row arithmetic verbatim from v1; the FNV gate confirmed bit-identical
outputs on all five geometries. Timing: 0.35-0.80x -- deep cases 2.57x SLOWER
(11948 -> 30735 us at 128/981/8192). Reading the columns: v3 moved 2x the KV
bytes (half the rows share each K walk) at ~78% of v1's effective byte rate;
the occupancy bought nothing. The v2 post-mortem's "occupancy is the axis" was
overstated: BOTH fatter (v2) and thinner (v3) blocks leave the binding term
untouched, and v3's rate loss per doubled byte shows what it is -- the
2-byte-at-a-time gather. 4096 scalar global loads + 4096 scalar LDS stores per
K tile at hd=128; the tile runs its deep cases at ~90 GB/s because the copy
loop issues 2 bytes per instruction, not because it stalls.

(b) `stage_kv_tile` walks EIGHT bf16 per `uint4` load/store when the copy is
provably bit-identical: bf16 KV, contiguous rows (`kv_sd == 1`), 16B-aligned
bases, 8-aligned live row bases; the original scalar loop serves everything
else, and fp8 stays scalar (per-byte decode). Q staging same treatment. LDS
pitches D+4 -> D+8 so 16B LDS stores stay legal at every row (41776 B total at
hd=128, still under the static ceiling). A copy cannot round differently --
the bench's hashes ARE the gate: all five geometries hash byte-identical to
the scalar arm's runs, which is also what exempts this from a KLD run
(schedule-class change with identity evidence).

| hd/qlen/ctx   | v1 scalar us | vec us | speedup |
|---------------|--------------|--------|---------|
| 128/64/1024   | 761          | 510    | 1.49x   |
| 128/128/2048  | 1463         | 893    | 1.64x   |
| 128/981/8192  | 11948        | 7368   | 1.62x   |
| 128/512/4096  | 2935         | 2043   | 1.44x   |
| 64/512/4096   | 2020         | 1671   | 1.21x   |

Effective KV rates rise to 143-146 GB/s deep-ctx (still ~3.5x under the card).
The decode rows share the body and share the win's mechanism: split-KV probe
passes on splits 1/2/8/64 (device arm and host re-merge, 0 bad cells, 0 NaN
slots), oracle sweep PASS 7/7 including the window and unmapped-edge cases,
full kbench sweep (both fixtures, `--all-rows`, coverage ratchet) 1083/1083
worst 3.135e-03 unchanged, exit 0 -- with the serve container up and idle
(both sweep arms this session ran without the card to themselves and stayed
fault-free; contention false-fails remain a reason to distrust FAILs, not
passes).

Kept: v3 stays behind `R3D_ATTN_V3=1` (same opt-in postmortem pattern as v2).

### Serve-side A/B (promotion run, same session, paired by mount swap)

build6 (scalar staging) vs build7 (vec), identical container/model/flags,
uuid-salted prompts, arms interleaved within one hour window:

| probe            | build6       | build7       | verdict |
|------------------|--------------|--------------|---------|
| TTFT @981 (x5)   | 0.255 med    | 0.248 med    | wash (attention share small at short ctx; distributions touch) |
| TTFT @1961 (x5)  | 0.485 med    | 0.446 med    | win, non-overlapping (0.484 min vs 0.454 max) |
| decode @2k (x1)  | 31.7 tok/s   | 38.8 tok/s   | 1.22x single-shot |
| decode @8k (x1)  | 14.0 tok/s   | 22.1 tok/s   | 1.57x -- lands exactly in the tile-bench predicted band (1.44-1.64x) |

Decode window = cached-prompt pattern (n_predict=1 warm with cache_prompt=true,
then timed generation; `tokens_predicted` is the count field). The 2k first
sample early-stopped after 1 token: staging is bit-identical so it is not a
kernel signal -- filler text near an EOS-prone shape; decode probes must pin
gen length and treat early stop as a spoiled sample, not a rate.
R3D_NO_WMMA=1 arm at 8k prefill exceeded a 300 s client budget (minutes-slow
scalar attention) -- deep-context necessity of the WMMA row, quantitatively.
Promoted: serve now mounts build7 as default (0ce1007); no env flag needed
for rollback except rebuild-and-remount of build6 (kept on disk).
Open next: the gemm DBUF lesson (prefetch the next slice during this tile's
mma) is now the untried axis for attention -- at 143 GB/s there is ~3.5x of
card left on the staging path alone.

## 2026-10-20: attention DBUF staging shipped — the gemm lesson transfers (1.19-1.27x tile, +21% serve decode@8k)

Closed the open item above the same day. `body.inc` gained a `DBUF` template
arm: `ld_kv_regs` walks the NEXT k-slice's gather global->registers while the
current tile stages LDS and runs mma, so the memory stall shrinks to
max(load, stage) instead of load+stage. Two structural consequences:

* Row addressing (`RB` row-bases, `CB` col-bad bytes) double-buffers on LDS --
  the prefetch must not scribble the addressing the in-flight mmas read. CB
  updates land strictly after the barrier that publishes the staged tile.
* Barriers drop to a3 in pass1 and a4 in pass2 (v1: a4/a6) -- a side effect
  of the restructure, not the point. v2 taught us barrier count is not the
  bottleneck; here the win is the overlap.

Arithmetic is v1's chain verbatim (same fragment order, same LDS slots): all
five bench geometries hash byte-equal across arms, both directions.

Paired tile bench (build8, same binary, old arm sandwiched, drift <1.3%):

| hd/qlen/ctx | DBUF us | old us | DBUF/old |
|---|---|---|---|
| 128/64/1024 | 368 | 461 | 1.25x |
| 128/128/2048 | 688 | 888 | 1.29x |
| 128/981/8192 | 6265 | 7512 | 1.20x |
| 128/512/4096 | 1698 | 2055 | 1.21x |
| 64/512/4096 | 1358 | 1673 | 1.23x |

Deep-ctx KV rate 141 -> 171 GB/s (~2.9x of card still left; the next stall is
register-prefetch depth: one tile ahead, could be two).

Serve A/B (build8 mount, arms differ only by `R3D_ATTN_NO_DBUF=1`, two rounds
alternating, uuid-poisoned prompts):

| metric | DBUF | old | verdict |
|---|---|---|---|
| decode 8k tok/s | 26.4/27.2/27.6/28.6 | 22.1/22.5/23.9 | non-overlapping, +21% |
| TTFT 1961 s | 0.442-0.467 | 0.466-0.507 | ~4% faster |
| TTFT 981 s | 0.254 | 0.263 | ~3%, near noise |

Gates, both arms: kbench shipped pack 1075/1075 (worst rel_l2 3.135e-03
unchanged), attn_rows 8/8, oracle sweep 7/7, splitk probe spl 2/8 identical.
Kill switch `R3D_ATTN_NO_DBUF=1`; fp8 KV and odd shapes never enter the arm
(vec-path gate). Live serve: build8, DBUF default.

Lesson for the skill: when a copy path is vectorized but still slow, the
next lever is pipelining the copy across the compute step -- re-tiling
geometry (v2/v3) was waste; the same DBUF pattern transferred from gemm to
attention staging nearly verbatim.
