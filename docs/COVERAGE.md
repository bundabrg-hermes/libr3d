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

## The 27B MXFP4 target: what the container actually asks for (2026-10-20)

`/srv/models/rad/qwen3.8-27b-mxfp4.rad` (18.05 GiB, RAD2, arch `qwen35`, source
quant fp8_e4m3, converted from amd/Qwen3.8-27B-Quark-AWQ-MXFP4). Assessed with a
recorded fixture, not by reading code: `rad-kbench -m <skeleton> --record` needs
no card at all, because the skeleton keeps the tables and holes the data, and
recording draws weights from seeds.

```
rad-kbench -m /models/qwen3.8-27b-mxfp4.rad --skeleton /scratch/skel27.rad   # 7 MB on disk
rad-kbench -m /scratch/skel27.rad --record /scratch/mxfp4.rkb --max-ctx 4096  # ~6 min, CPU
```

**The model is a hybrid, and most of it is not attention.** 64 layers, full
attention every 4th (16 layers: head_dim 256, GQA 6, gated q/gate interleave,
partial rotary 64/256, MROPE interleaved); the other **48 are Gated DeltaNet**.
Drafter scope is five `dflash.blk.*` layers at layer indices 64-68 (head_dim
128, plus `attn_conv_base`/`attn_conv_proj` and `ffn_conv_base`/`ffn_conv_proj`),
with `dflash.draft_head.weight` at 2 bits (`u2*f16[1x128]-u8[1x128]`, 331 MiB).
Trunk `output.weight` is one fp8 tensor with 128x128 block scales (1.18 GiB),
not bf16 -- `logits_gemm_bf16` cannot serve it. Weight bytes: 71.5% MXFP4
(319 tensors, 12.89 GiB), 20.2% bf16.

**Completeness, from the fixture (op x library matrix in the kbench report):**

| declared graph | ops | libr3d holes |
|---|---|---|
| drafter + vision tower | 41 | 15 |
| `--num-speculative-tokens 0 --mm-max-patches 0` | 31 | 9 |

Holes with the drafter off: `gdn_chunk_scan`, `gdn_conv_update`,
`gdn_gated_rmsnorm`, `gdn_kkt_solve`, `gdn_recurrent_update`, `gemm_nt_q`,
`mul`, `rope_table`, `sigmoid`. The drafter/vision scope adds ten ops, six of
them holes (`attn_dense`, `dflash_select`, `gemm_nt_bias`, `grid_embed`,
`layernorm`, `scatter_rows`).

**The MTP heads are avoidable at runtime, GDN is not.** Speculation off drops the
drafter's whole scope, its conv tensors and its 2-bit head from the declared
graph -- so the `mtp_num_hidden_layers 1` headers cost us nothing as long as we
serve `--num-speculative-tokens 0` (which our fixture serve already does). The
gate is the five GDN rows: 48 of 64 layers, and libr3d has no GDN family at all.
Two more GDN ops (`gdn_conv_prep`, `gdn_conv_recurrent_update`) are in the
program but absent from the matrix, because record declined them -- see below.
The graph column reflects ops with recorded cases, so it also under-reports
`embed_lookup`/`logits_gemm`, whose cases the memory budget skipped (each needs
>7 GiB).

**A dtype disagreement worth reporting upstream.** The container stores
`ssm_dt.bias` and `ssm_a_log` as **bf16** (48 elements) while
`arch/common/rad_block_gdn_fp8.h` declares `w_dt_bias`/`w_a_log` as **RAD_F32**
`{n_head_v}`. kbench then declines 40 `gdn_conv_prep` cases ("operand 'dt_bias'
is a weight and the declaration bound none to it") and substitutes
`ssm_norm.weight` (128) for `dt_bias` (wants 48) on `gdn_conv_recurrent_update`.
The check that fires is in `tools/rad_kbench.cpp`, so it is a harness-side
substitution failure, and an engine load settled it: against the skeleton the
engine accepts the container outright (`model: qwen35 arch=qwen35
quant=fp8_e4m3 64 layers`, `declare: 1297 ops, 1600 weights, 64 buffers, 4 KV
groups`) with no complaint about those weights. The recipe casts only
`ssm_ab.weight`; no f32 cast for the other two is needed to load.

That load also names the op list the engine declares for this trunk, which is
the authoritative hole inventory: **seven** GDN ops, not the five the fixture
matrix shows (`gdn_conv_prep` and `gdn_conv_recurrent_update` are in the program
but recorded no case), plus `gemm_nt_q`, `mul`, `sigmoid`, `rope_table`. Run
capped (`--memory 6g`, host backend, no device) it reached placement and stopped
on arithmetic, not on the model: static weights want 12.99 GiB of the 12.99 GiB
claimable, so KV gets nothing. Budget to remember for the real card: 12.99 GiB
static + 2.41 GiB activation arena + ~129 MiB step staging -- i.e. the 24 GB
gfx1100 holds it alone, not beside the fixture serve.

**Feasibility, stated plainly.** On gfx1100 the trunk's shape arithmetic is not
the problem -- 15.6 GiB of weights fit the 24 GB card and GDDR6 at ~480 GB/s
bounds decode near 30 tok/s. The problem is that two whole op families are
missing (GDN chunked scan + triangular solve, and MXFP4 GEMM), and GDN is the
largest single body of work attempted in this library. **The same container is
already fully served on the R9700 cards**: the stock `stilldeadcode/radiance`
image ships `libr4d 0.5.0, 351 kernels, built for gfx1201` (absent from our
gfx11 build, which is why it looked missing), covering GDN, h256/gqa6 attention,
`gemm_mxfp4a8_{m64,decode,tiled}` and the drafter. Trying the 27B MXFP4 on rdna3-dev is
therefore a scheduling question -- one R9700 window on the stock image -- not a
kernel question. libr3d's MXFP4 row is still worth doing, but GDN first, or the
48 GDN layers run on AVX in host RAM regardless of how good the GEMM gets.

## 2026-10-20 (late): the GQA fold at serve, and an A/B that tested nothing for two sessions

**Fold serve A/B, the first real numbers** (same .so build10, arms differ only by
`R3D_NO_GQA`; pairs alternated twice to cancel drift; MiniCPM fixture, rdna3-dev):

| arm | decode @8192 | decode @2048 | TTFT @1961 (min) |
|---|---|---|---|
| plain (per-q-head, spl=12) | 27.0 / 27.2 tok/s | 41–42 | 0.510 s |
| **fold (per-kv-head, spl=64)** | **48.9 / 50.3** | **47.6–49.0** | 0.470 s |

Non-overlapping distributions at 8k across both rounds; the mechanism is exactly
the promised one (KV streamed once per q-head group). TTFT unchanged (prefill is
GEMM-bound). Fold is the shipped default; serve now mounts build10.

**The fold "serve crash" was a harness illusion.** The A/B's fold phases never
started a server at all: `docker rm -f` silently failed (its error was swallowed
by `>/dev/null 2>&1`), radiance ignores SIGTERM and its GPU teardown stalls
behind the co-tenant (`kfd_process_wq_release` hog warnings in dmesg right at
the arm switches), so the kill lands but the reap doesn't -- an EXITED corpse
keeps holding the container name and the next `docker run` dies on `Conflict`,
whose error the orchestrator's `tail -1` also dropped. The `Connection refused`
+ stale-`live step` output that read like a fold warmup hang was the *previous
arm's* idle server being torn down. Fixes: rm retry loop with a loud failure,
`--stop-signal SIGKILL`, `RUN FAIL` echo on docker run, orchestrator `tail -2`.
Rule: an A/B arm that never printed its launch proof tested nothing -- the
fold arm had been silently absent from every serve comparison before this.

**Probe/gate repairs.** `repro_attn_splitk` (b) re-merged the poisoned arena at
SPL=1, where the entry routes to the SPLITK=0 tile arm and no partials exist --
a vacuous check that "failed" 1.000 while (a) was passing (the gate's grep
pattern dropped the (a) line, which is how a green suite shipped a scary
phantom). Now (b) skips at SPL=1 with a comment, every probe line prints (a).
`REPRO_CUS`/`REPRO_GQA` exercise the engine's real call shape (cu_seqlens on
every decode call); the new fold+cus+gqa8+qlen8 case passes **pre-fix too** --
`kBM=64` makes the loose bound exact through qlen 8 x gqa 8 -- but the cus
branch of `per_seq` now multiplies by gqa as its own comment always claimed,
which is the exact bound only when gqa > 8 (27B-class, and dflash2 at qlen>1):
insurance for the 27B track, not a live bug on this one.

## 2026-10-20: gemm_nt_q mxfp4a8 shipped -- the quantised GEMM opens on RDNA3

`gemm_nt_q_mxfp4` (src/r3d_gemm_mxfp4.hip) answers `gemm_nt_q` at
`dtype=mxfp4a8` -- 58 recorded fixture cases from `qwen3.8-27b-fp8.rad`, zero
model risk. RDNA3 has no fp8/fp4 hardware, so this is a boundary conversion on
the m64 WMMA tile, and the two decodes involved are exact in bf16 (e4m3
mantissa <= 3 bits, e2m1 <= 2). The scales never touch bf16 rounding either:

* B fragments stage as `e2m1_bits + ((x - eref[n]) << 7)`: bf16 is a biased
  exponent, so the add IS the multiply by 2^(x-eref), exact while normal.
  `eref[n]` is the row's max non-NaN E8M0 (the `b_ref` role, derived in-kernel
  at block entry; the passed `b_ref` is accepted and ignored -- both host
  libraries ignore it too).
* WMMA accumulates raw products for exactly one 128-of-K activation group
  (8 slices of 16; 32 divides 128); the flush multiplies the partial by
  `sa[m, grp]` in f32 through LDS (`Acc`), and the store multiplies once by
  `2^(eref[n]-127)`. `(sub*2^d)*sa == (sub*sa)*2^d`: one rounding of the same
  mantissa, the same class the mma's opaque k-tree already lives in.
* NaN scale bytes stage their whole 32-column group as bf16 NaN. The oracle's
  `(sub*sa)*NaN` poisons the entire output even for an all-zero-code group
  (0*NaN is NaN); OPS.md's "treat as empty" is a permission for zero
  activation scales, not for NaN weight scales. Padding (m>=M, n>=N, k>=K)
  stages plain zeros -- distinguishable from NaN only because the prefetch
  carries a liveness bit.

Why `Acc` lives in LDS: this ROCm's rocwmma has no coordinate fragment (no
`rocm_t`), so a register-side per-row multiply means hardcoding v_wmma's lane
layout. The 42 KB carve is 1 block/CU; the GEMM is weight-bandwidth-bound
(M=1 decode reads 64 MB of nibbles per call at N=5120 K=25600), and 80..2560
blocks over 96 CUs leave nothing a second resident block could exploit.
`R3D_GEMM_NO_DBUF=1` is the schedule arm, `R3D_NO_MXFP4=1` the serve kill
switch (the row refuses, the engine answers from libref, same binary).

Gates, in order: loader (`rad-info --plugins` with libref co-loaded -- the
verbatim libr4d pGemmNtQ/oGemmNtQ dedupes clean); attributed sweep (libavx
excluded so counts attribute): **58/58 ok, worst rel_l2 3.21e-04 vs tol 3e-2**,
M=1 through 512, ragged M (65/129/258) included, launch proof line present,
58 cases guarded, 0 wrote outside; full pack with libavx+libref: **1850/1850,
0 failed**, 24 named skips all 7+ GiB embed_lookup/logits_gemm cases over the
1.36 GiB co-tenant budget (the same named-skips class as every recent run).
Serve build11: MiniCPM never issues gemm_nt_q -- the row is inert there; it
ships to be the 27B's door, which the GDN family still has to open.

Fixture note (tools/kf_probe_q.cpp, build inside radiance-dev):
`g++ -std=c++20 kf_probe_q.cpp kfixture.cpp opshapes.cpp core/format/encoding.cpp
core/util/rad_util.cpp -I/opt/radiance-src{,/tools,/core,/abi}
-I/opt/radiance/include/radiance`. The recorded geometry spells the contract:
a_scale [M, K/128] f32 (sigmoid-filled, so strictly positive), b_scale [N, K/32]
e8m0, b_ref present as the full exponent plane (a [N] per-row reduction is what
the oGemmNtQ comment means, not the recorded shape), a_sum ABSENT in every case.

## 2026-10-20: mul / sigmoid / rope_table / gdn_gated_rmsnorm -- elementwise holes closed, the GDN family opens

With `gemm_nt_q` shipped, the 27B trunk's remaining holes were five recorded GDN
ops plus three elementwise ones. A correction first: the previous session's note
that `quant_act_fp8` (the producer of this GEMM's fp8 activations) "still has no
device row" was wrong -- the band has shipped `quant_act_fp8_act` since MiniCPM,
and the assessment table above never listed it as a hole. The missing producer
never existed; today's sweep confirms the quantizer rows replay clean.

Four rows, all fixture-gated against the real 27B pack (995-case nodraft):

| row | op | worst rel_l2 (tol) | note |
|---|---|---|---|
| `mul_act` | mul | 0.00e+00 (6e-03) | bit-identical at n=6144 M=1..512 |
| `sigmoid_act` | sigmoid | 0.00e+00 (6e-03) | the double-rounded exp pays off exactly |
| `rope_table_f32` | rope_table | 1.9e-08 (1e-05) | first row where the sincosf-vs-double delta is *visible*: the store is f32, so the ~1 ULP difference surfaces at 1e-8 instead of hiding under a bf16 narrowing. Still 500x under the case tolerance. |
| `gdn_gated_rmsnorm_bf16` | gdn_gated_rmsnorm | 9.9e-06 (6e-03) | tree-vs-sequential row sum, the same rounding class `rmsnorm_bf16` has always traded; 600x margin on real GDN data |

Regressions with all libraries: kernels.rkb **1867/1867** (was 1850 — the +17 are
the stock pack's 7 rope_table + 10 gdn_gated_rmsnorm cases, which already carry
27B shapes there and now run device-side as well as host), mxfp4 full pack
(drafter + vision) **4513/4513**; worst rel_l2s unchanged at 3.135e-03 /
2.859e-03, the 24 named skips the same 7+ GiB embed/logits budget class. After
this batch the full-pack `not reached: libr3d` list is 7 rows, all known classes
(budget-skips, cast pairs absent from that pack, M==1 decode rows) — no 27B-side
surprises.

Design notes that outlive this batch:

* **`offlin` is a liar for linear walkers.** `ref_sigmoid` walks numel(Y) through
  `offlin`, whose identity branch is contiguity -- a rank-2 with a pitch wider
  than n reads *different elements* than a linear walk would. The `sigmoid` row
  therefore refuses pitch > n even though `row2_pitch` (and the `add` row, which
  walks per-row and agrees with offlin there) accepts it. Same class of trap as
  the descale-pointer poisoning in the attention harnesses: matching ref means
  matching its *index function*, not just its arithmetic.
* **The GDN gate_rowoff is worth keeping as a first-class shape.** The 27B's
  gated norm runs against a plain [M, 128] z today, but the engine's fused
  [rows, heads*channels] slice walk costs one extra division in the kernel and
  buys coverage of the fused epilogue the recurrent fold will produce.
* **r3d_gemm_mxfp4.hip was compiled without the house `-ffp-contract=off`**
  (CMakeLists source-property line missing). Its chains happen to contain no
  contractible mul+add pattern, so nothing was wrong -- but the flag is a
  per-TU promise, and one edit away from breaking silently. Line added.

Next in the family, in fixture-sake order: `gdn_conv_update` (rolling [.., 3]
conv state, one plane), `gdn_kkt_solve`, `gdn_chunk_scan` (both head_k 128, M to
512), then `gdn_recurrent_update` last: it carries three encoded weight planes,
which a device row cannot consume without a layout hook. The two unrecorded ops
(`gdn_conv_prep`, `gdn_conv_recurrent_update`) wait on the dt_bias dtype fix
upstream or on `--record-rows` once their siblings' kernels exist.

## gdn_conv_update_bf16 -- the second GDN row, bit-exact across the whole 27B decode band (2026-10-20)

The decode-band convolution over the rolling window (ref_ops.h spec 10): one
block per (sequence, channel), ref's `REF_PAR_FOR(N * conv_dim)` decomposition
verbatim. The fixture pack records q_len 1..512 at head_k=head_v=128,
conv_width=4, conv_dim 10240 (16 q/k + 48 v heads), all bf16, bias absent,
state rank-3 [S, conv_dim, width-2+M] -- a spec-decode band, not just plain
decode: `num_accepted` picks the read offset so a rejection is a read-offset
change, never a recompute.

The arithmetic came out exactly as the gemm doctrine predicts: a width-tap f32
chain in ref's ascending-j order over exact bf16 loads (bf16xbf16 products are
exact in f32), silu on the double-rounded sigmoid `sigmoid_act` already proved
bit-identical. Every check on every sweep: **rel_l2 0.000e+00**, including the
rewritten INOUT `conv_state` -- four output operands per case, none diverging.

Gates (all on the real packs, serve untouched -- the row is inert for MiniCPM):

| sweep | result |
|---|---|
| nodraft 27B pack, `--op gdn_conv_update` | 80/80, worst 0.000e+00, per-case table attributes `gdn_conv_update_bf16 \| libr3d \| device` |
| stock kernels.rkb, all libraries | 1877/1877 (24 named budget-skips as ever), worst rel_l2 unchanged at 3.135e-03, coverage ratchet re-locked at 30 libr3d rows |
| full mxfp4 pack (drafter + vision) | 4553/4553 -- up from 4513 checked, the +40 being the conv_update cases that now run device-side |

Design calls worth keeping:

* **ref's op-wide window guards became per-sequence skips.** `off + width - 1 >
  state_len` reads device data on the launch side; ref can fail the whole call,
  a plugin block can only abstain. A failing guard leaves that sequence's state
  exactly as it was -- visible in the INOUT diff if a caller ever trips it.
* **`offlin` is the flat identity on DENSE index tensors** -- for a rank-2
  [S, M] `state_idx` the ref reads element `n` flat, NOT `n*M`. The row demands
  dense index tensors (then agrees with ref by construction) and refuses
  strided ones rather than reimplementing the decomposition.
* **T comes from `x->shape`, not `cu`.** ref reads `cu[N]` host-side because it
  is host code; every valid call has cu's last entry equal to x's row count, and
  the tensor shape is the only one a launcher can see without a sync.
* The probe `tools/kf_probe_ops.cpp` (dump any op's recorded geometries without
  a card, one sample per param digest) is the right first move for any new row:
  it cost one container run and settled geometry, dtype, optionality, and
  rank before a line of kernel code.

GDN holes left for the 27B trunk: `gdn_kkt_solve`, `gdn_chunk_scan`,
`gdn_recurrent_update` (three encoded planes, needs the layout hook), plus the
two unrecorded ops.

## gdn_kkt_solve_bf16 -- the third GDN row, forward substitution bit-exact (2026-10-20)

The chunk triangular inverse `A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1`,
one block per (chunk, sequence, v head). Geometry pinned by `kf_probe_ops` before any
code: head_k 128, chunk 64 recorded ONLY via A's shape `[M, 48, 64]` (the param digests
carry `M` and `head_k` alone -- ref's fallback of reading chunk from A's trailing shape
is the live path, not the dead one), k bf16 `[T,16,128]`, beta/g f32 `[T,48]`, cu i32,
M 1..512.

The row is deliberately the SCALAR sibling of libr4d's `k128_c64` WMMA row: ref's inverse
is a per-column forward-substitution chain (`A[i][j] = delta - sum_{l=j}^{i-1} M[i][l]
A[l][j]`, f32, l ascending), and a thread-owns-column kernel walks that chain in ref's
exact order -- bit-exact by construction, which the WMMA blocked merge cannot claim.
Chunks are independent by construction, so the chunk index became the grid's free axis
(it is what fills the CUs where ref's N*H parallelism would idle the card at prefill);
thread ownership makes the substitution barrier-free inside its loops. LDS ~25 KB
(strict-lower triangle + per-thread columns padded to 65) -> two blocks/CU.

Gates (all on the real packs, serve untouched throughout):

| sweep | result |
|---|---|
| nodraft 27B pack, `--op gdn_kkt_solve` | 80/80, worst 0.000e+00; all 10 digests attributed `gdn_kkt_solve_bf16 \| libr3d \| device` |
| stock kernels.rkb, all libraries | **1887/1887** (was 1877 -- the stock pack carries 10 kkt cases, one per digest M=1..512, previously host-only, now device-side AND bit-exact), worst rel_l2 unchanged at 3.135e-03, libr3d coverage 31 rows |
| full mxfp4 pack (drafter + vision) | **4593/4593** (was 4553, the +40 the kkt cases), worst 2.859e-03 unchanged, the same 7 known not-reached rows, 1215 guarded-allocation cases 0 OOB |

Lessons, permanent:

* **T is `k->shape[rank-3]`, not `[rank-2]`.** The conv row's "T from the tensor shape"
  reading reads shape[rank-2] -- correct for a rank-2 `x`, but for a rank-3 `[T, Hg, K]`
  operand that slot is the HEAD axis. Wrong T mis-derived H/Hg, inflated grid z past the
  real head count, and wrote 256 B past A on the first L==chunk cases -- kbench's guarded
  allocations named the operand and the size on the very first card run. "shape[rank-2]
  is the token axis" is a rank-2 fact, not a house rule. (591de90 caught, a8d69aa fixed.)
* The stock kernels.rkb is not shape-free of the 27B: it now carries gdn cases at
  head_k/head_v 128 (the conv and kkt digests), so a new GDN row shows up in the STOCK
  regression count too. Expect the +N; it is the row proving itself twice.

GDN holes left for the 27B trunk: `gdn_chunk_scan`, `gdn_recurrent_update` (three encoded
planes, needs the layout hook), plus the two unrecorded ops.

## gdn_chunk_scan_bf16 -- the fourth GDN row, the serial scan with the state in registers (2026-10-20)

The chunked delta-rule scan `W = beta(V - e^g K S^T)`, `V' = A W`, `O = scale e^g Q S^T + sA V'`,
`S = e^{g_last} S + V'^T K'`. Geometry pinned by `kf_probe_ops` before code: head_k=head_v=128,
chunk 64 recorded (again) only via A's shape `[M,48,64]`; q/k bf16 `[T,16,128]`, v/o bf16
`[T,48,128]`, g/beta f32 `[T,48]`, h0/ht f32 `[1,48,128,128]`, cu i32, state_idx i32; 10 digests
M=1..512 in BOTH packs. This is the first r3d GDN row whose `scale` parameter is live: every
recorded case carries 0.0883883 = 1/sqrt(head_k), read as ref does (f64 param, f32 use).

Design calls worth keeping:

* **The chunk recurrence is serial by nature** -- W reads the state the previous chunk left --
  so unlike kkt_solve there is NO free chunk grid axis. The block is ref's own `REF_PAR_FOR`
  unit: one per (sequence, v head).
* **The transposed f32 state lives in registers**: one thread owns one value column e of
  St[d][e], a 128-float `#pragma unroll`ed array (launch_bounds(128,1) opens the full
  256-VGPR budget). Whole-column ownership is also what keeps bit-exactness cheap: every
  ascending-d chain stays a scalar walk of ref's order; splitting columns across lanes would
  hand chains between lanes just to stay serial.
* `V' = A W` runs **descending i in place** over the W buffer: row i consumes only j <= i, so
  each write lands after its slot's last read. One LDS buffer, not two.
* The sA row and staged g/beta are LDS broadcasts with **barriers outside every predicate**
  (threads owning no column still march through them), and the sA loop carries TWO barriers
  per row: row-ready-before-reads and reads-done-before-the-next-row-overwrites.
* **A negative `state_idx` slot skips the whole (n, h) row -- the o writes included**, exactly
  ref's `continue`; the row pitch may be wider than one slot (`stride[0]`, column 0 is the
  slot); h0/ht strides are READ off the tensor at rank >= 4 (ref's page-padding comment).
* Launcher derived T from `q->shape[rank-3]` from day one -- the kkt lesson applied before
  the card could re-teach it, and this row's FIRST card run was green, launcher included.

Gates (all on the real packs, serve and co-tenant serve untouched throughout):

| sweep | result |
|---|---|
| nodraft 27B pack, `--op gdn_chunk_scan` | **80/80**, worst 0.000e+00; all 10 digests attributed `gdn_chunk_scan_bf16 \| libr3d \| device`; 0 OOB |
| stock kernels.rkb, all libraries | **1897/1897** (was 1887 -- the stock pack carries its 10 chunk_scan digests too, now device-side AND bit-exact), worst rel_l2 unchanged at 3.135e-03, libr3d 41 rows |
| full mxfp4 pack (drafter + vision) | **4633/4633** (was 4593, +40 from the chunk_scan rows), worst 2.859e-03 unchanged, 1255 guarded-allocation cases 0 OOB |

The graph ratchet for the nodraft 27B skeleton moves 29 -> **30 of 31**: the last device hole
is `gdn_recurrent_update` (the decode path, with the three encoded weight planes -- the
layout-hook row).

Lessons, permanent:

* **`tools/gate.sh` now IS the gate** (`op <name>`, `stock`, `full` on rdna3-dev): the exact kbench
  command lines, the two staged homes (attribution = oracle+libr3d only; regression adds
  libavx), persistent ref caches under /scratch. No more reconstructing gate lines from
  report headers.

GDN holes left for the 27B trunk: `gdn_recurrent_update` (the last graph hole), plus the two
unrecorded ops (`gdn_conv_recurrent_update` and the conv-prep sibling) and the two fixtures
declined by the ssm-dt dtype quirk.

## 2026-10-20: gdn_recurrent_update_bf16 -- the last device-graph hole closes, 31/31

The decode recurrence against the paged state cache: per-token delta rule
`S = e^g S + beta (v - S k) k^T`, the qk l2 norms, the q readout, and the gated RMS norm
folded into the epilogue (ref_gdn.cpp; 48 of 64 layers run it every decode step). Row
`gdn_recurrent_update_bf16`, kernel in `src/r3d_gdn.hip` after the chunk_scan section,
schema `pRecurrent/oRecurrent` verbatim libref (kSchemas-declared so the row stands alone),
constraints `head_k==128, 1<=head_v<=128, state_form in {per_candidate}`.

**The layout hook was a phantom.** The handover carried `gdn_recurrent_update` as
"needs the weight-layout hook for three encoded planes" -- the probe settled it in one
run: `A_log`, `dt_bias`, `norm_w` are `scheme=plain`, ONE plane each, stored bf16 in this
container. `weight_refusal` declines a hookless device row only above one plane, so the
row reads them AS STORED through `ldt2` -- dtype-driven exactly like the oracle's `ldt`,
which is what keeps bf16-stored weights bit-equal on both sides. The three-week-old
"blocked by the layout hook" note was inherited, probed once, and dissolved.

**One block per (sequence, v head) -- thread owns one value column.** The same
transposed-state register layout as chunk_scan (thread e owns `St[d][e]`, 128 f32
registers), and the delta rule turned out to be nearly barrier-free BY THE MATH: the
update's two d loops read only `kn[d]` into thread e's own column, so decay, the S k
product, the rank-1 add and the q readout are all register-local per thread. Only two
reductions cross threads, and BOTH walk ref's serial order on purpose:

* the qk l2 sums -- thread 0 over the staged q/k rows (not two per-thread chains);
* the rms sum `sq2 += orow[e]^2` -- a 128-long ASCENDING-e chain, so `block_sum` (the
  tree) would reassociate it. The squares stage to LDS and thread 0 walks the chain.
  128 cheap LDS reads per token is the price of the contract; correctness trades no chain.

Barriers: six per token, matched on every path (the `!own` tail marches through two
beside the rms pair). Early returns (`slen<=0`, negative input slot) are block-uniform.

**The two bugs the first card run found, both inherited-shape traps:**

* `r3d_softplus` shipped as `log1p(-x)` / `log1p(x)` -- the exp fell out of the port, and
  `log1p(-1) = -inf` sent the gate exponent to +inf and the state to 1e14. Ref's
  act_softplus takes `log1p(exp(+/-x))` in both branches; the fix restores it verbatim.
* `norm_w` was offset TWICE -- `nw + e*w_s1b` AND index `e`. The row landed on element
  2e of the bf16 plane, so every channel got the wrong stored scale while the state
  stayed bit-exact (the weight feeds only the epilogue) -- a failure signature that
  localised the bug to the epilogue in one read of the report. One offset only.

`state_form=per_candidate` is constraint-gated, and the rule that "a constraint on a key
the query lacks does not hold" makes it exact: the arch block always declares state_form
("anchor when speculating, else per_candidate"), so serve decode and the nodraft pack hit
this row, while anchor batches (the drafter, and every gdn_recurrent case in the FULL
pack) stay with libr4d/libavx untouched -- confirmed by the full-pack report: zero
anchor cases attributed to libr3d. `o_q/o_scale` (the folded out-projection quantiser,
absent from every recorded 27B case) is REFUSED, never silently ignored.

Gates (all via `tools/gate.sh`, serve and co-tenant serve untouched throughout):

| sweep | result |
|---|---|
| nodraft 27B pack, `--op gdn_recurrent_update` | **64/64**, worst rel_l2 **0.000e+00** (bit-exact, all 8 q_len digests 1..128 attributed `gdn_recurrent_update_bf16 \| libr3d \| device`), 0 OOB |
| stock kernels.rkb, all libraries | **1897/1897** unchanged, worst rel_l2 unchanged at 3.135e-03 (the stock pack's recurrent cases are anchor-form -- correctly not taken) |
| full mxfp4 pack (drafter + vision) | **4633/4633** unchanged, worst 2.859e-03 unchanged, 0 OOB (its recurrent cases are all anchor-form, all libavx/libref as before) |

**The ratchet: 30 -> 31 of 31.** After staging the new `.so` into the serve home, the
engine restarted clean (the startup refusal it exercised since day one is gone from this
op): `kernel selection libavx (host) 564, libr3d 564 op bands` -- every device band
covered, decode ~60 tok/s on a short request, health 200.

Lessons, permanent:

* **A serial reference reduction is the contract, not an implementation detail.** When
  ref accumulates `sq2 += orow[e]^2` ascending, the tree sum is a different number.
  Stage to LDS, walk it with one thread; the cost is bounded and the bit is exact.
* **Early returns in barrier-carrying kernels must be block-uniform** -- guard on
  blockIdx-derived values only (`slen`, `slot_in` qualify; `orow` never does).
* **The probe answers the hook question for free.** Encoded != multi-plane: plain-scheme
  single-plane weights are weight_refusal-invisible and dtype-driven reads on both sides.

GDN holes for the 27B trunk: NONE -- the graph is whole. What remains in the family is
the two unrecorded ops (`gdn_conv_recurrent_update` and the conv-prep sibling, blocked by
the upstream ssm-dt dtype quirk, harness-side) and the anchor-form row the drafter needs
(already libr4d's; the constraint keeps both libraries honest).

## logits_gemm_e4m3 — the head GEMV on joined planes; 27B placement chain mapped (2026-10-20)

The 27B container stores `output.weight` as fp8_e4m3 codes + a bf16 scale grid
(`rad.planes`, block 128x128). No libr3d row could claim it: hookless device rows
are declined for multi-plane weights, and the host kernels get refused placement
for mapped-eligible tiers ("stored form is not the container's"). Startup died at
`logits_gemm -> [host:libavx]` on the 27B config.

* New TU `src/r3d_gemm_logits.hip` (`logits_gemm_e4m3`): reads the planes header
  verbatim from the ABI (magic PLNS), decodes codes+scale in-register. WMMA
  rejected on math grounds — decoded e4m3 x bf16 needs >11 mantissa bits, an
  f16-rounded product sits ~2e-3 off, and the head is the tolerance-critical op.
  Bandwidth-bound GEMV instead: warp per (row-block, m-group), xs staged per m,
  uint4 codes walk + 8-byte scale walk (br=128) or scalar walk (br=1), f32
  accumulation in the reference's ascending k order. MT in {1,2,4,8} from M.
* Two bugs, both caught ONLY by the standalone unit test
  (`tools/test_logits_e4m3.cpp`, card + CPU oracle, 8 cases: 8/8 after fix):
  1. The block-staged "uniform" scale was thread 0's *warp's* row — every other
     warp silently used the wrong head row's scale (br=1 exposed it; br=128 with
     rblk = r/128 made the whole block genuinely share a scale row, masking it).
  2. Row ordering: declaring this row *before* `logits_gemm_bf16` steals the
     plain bf16 band, because a hook answer of "unsupported" means *store as is*,
     not "decline". It would then refuse those at launch and serve would fall to
     host logits every token. bf16 row first; the encoded band declines the
     hookless bf16 row at declare time anyway.
* Attribution: the 27B serve startup line no longer lists
  `logits_gemm -> [host:libavx]` — the row claims the band. The stock/full pack
  sweeps budget-skip all logits_gemm cases (7.1-14.2 GiB against the co-tenant's
  RAM), so the unit test is the correctness evidence, plus serve adoption.

## 27B serve placement chain — ONE blocker, and it is the conv pair (2026-10-20, corrected same day)

An earlier draft of this section blamed an "fp8 GEMM wall". `--debug-selection`
refutes that: the trunk GEMM bands are claimed (995 op bands on libr3d; only
`attn_paged/rope/kv_store/*_table` even print a device+host pair line). The real
chain, from `sel27.log` (kept in `/scratch/r3d/`):

1. `gdn_conv_prep` + `gdn_conv_recurrent_update` resolve host-only (the two
   unrecorded ops). Their weights (`ssm_inz.weight/scale/ref`, `ssm_out.*` —
   three-plane encodings, plus `ssm_ab`) are then "weights only a host kernel
   can read".
2. Planner rule (engine_bringup, WARN x720): a block whose group contains such a
   weight keeps the WHOLE group in the container's mapping (mapped tier).
3. The group includes `attn_norm.weight`/`ffn_norm.weight`, which are declared
   RAD_F32 and stored bf16 -> `widen=true` at load -> `weight_file_direct` false
   -> the mapped-tier guard in engine_bringup refuses: "stored form is not the
   container's (widened at load)". Startup stops there.

Why minicpm never hit it: pure-llama, norms declared bf16 (identity, no widen),
so group-pinning to the mapping is legal there. Engine-side fixes exist (split a
widened weight out of a pinned group — it's 512 B/layer; or declare the qwen35
norm operand bf16), but the libr3d-side unlock is in scope and was always the
backlog: **device rows for `gdn_conv_prep` + `gdn_conv_recurrent_update` with a
layout hook for the three-plane ssm weights**. No host-only member -> no
group-pinning -> norms get staged -> startup proceeds. The kbench recording of
those ops is separately blocked by the ssm-dt bf16/f32 dtype quirk — gate via
`--record-rows` or a recorded-rows fix, not on the serve path.

`--debug-selection` also pinned the VRAM split for the eventual serve: weights
6.67 GiB static, KV 14.44 GiB, mapped 9.67 GiB, host pool 0 B. The card has room.

Incident: the first serve27 attempt OOM-killed `libr3d-serve` (MiniCPM demo,
exit 137) — 27B host-side mapping on a box holding co-tenant serve + one serve has
~2.6 GB headroom. Do not run serve27 while minicpm serve is up; minicpm serve
was restored after staging (health 200, 564/564 device bands).

## Whole-GDN serve A/B on minicpm: NULL experiment, closed (2026-10-20)

`R3D_NO_GDN=1` vs plain, paired same-session decode@8k on `libr3d-serve`:
40.2-41.3 vs 43.9-48.8 tok/s. The gap is noise — MiniCPM5-2B is pure `llama`
arch, **zero GDN ops** (rad-info: 42 layers plain attention, 2 kv heads). Both
arms executed byte-identical paths. The kill switch itself is verified working
where GDN exists (`rad-info --plugins`: 43 kernels / 37 schemas plain, 38 / 33
with the env). A meaningful whole-GDN serve A/B needs the 27B serve, which is
blocked behind the fp8_e4m3 GEMM wall above. The earlier claim that "minicpm
serve exercises GDN" was false — the evidence was `grep -c gdn` on a binary
container (589 hits = quantization noise). `grep -a` on binaries proves nothing;
ask `rad-info`.

## gdn_conv_prep_bf16 + gdn_conv_recurrent_update_bf16: SHIPPED, byte-exact (2026-10-21)

The pair that pinned the 27B placement chain (`0d9862f`, cold-fix `f2b1e13`).
Device rows remove the host-only group members, which is what force-mapped the
groups holding widened norms. `conv_prep` is the prefill preamble in three
launches (conv+state-commit, serial-order per-head l2 norms, chunk-reset gate
cumsum) so libref's phase order holds: the norms read the ROUNDED conv output,
the cumsum resets per 64-chunk. The fused row runs conv + recurrent in one
launch per (sequence, v head), staging through LDS rounded by x's dtype to
reproduce the pair's staging round trip; the conv window commits BEFORE the
state_idx skip, as the ref pair defines it.

Evidence (`mxfp4_nodraft.rkb` and stock/full carry ZERO recorded cases for
these ops — ssm-dt quirk), `tools/test_conv_pair.cpp`: dlopen both plugins,
launch each row through its own `rad_kernel_at` entry (the engine's own table),
seed -> device -> reseed -> ref -> memcmp, all four geometries byte-exact:
prep A (cold+warm slots, chunk-crossing cumsum, gates fused in x, slot order
out of batch order), prep B (V==K==128, separate gate projections, two chunk
resets), fused A (GQA grp 3, V<K idle-tail threads, n_acc rollback, negative
output slot), fused B (V==K==128, z absent, window longer than commit span).

Gates after the pair: stock 1917/1917 (worst 3.135e-03 unchanged), full
4633/4633 (see below), 0 OOB.

Two lessons paid for:
- **ref gates the history READ on has_init, not just the commit.** My conv
  kernel read the slot whenever it existed: a cold prefill on a recycled cache
  slot convolved over garbage. prep A caught it (exactly the cold tokens 70..72
  diverged); prep B stayed byte-exact throughout, which is why both belong.
- **a sequence's INPUT state slot must be disjoint from every slot another
  sequence WRITES this step.** The test's fused A first shared slot 0 between
  seq0's commit and seq1's read; BOTH arms raced (diff sets shifted run to
  run). That is a legal-violation detector worth keeping in mind for real
  batch construction, not just tests.

## gemm_nt_q + gemm_nt_q_gated at fp8a8: FP8 containers place fully on device (2026-10-21)

`qwen3.5-9b-fp8.rad` (rtn e4m3 128x128, bf16 scales) resolved both rows host on
gfx1100, so the planner put every projection's arithmetic on the host and the
mapped-tier group pin refused startup outright ("'blk.0.attn_norm.weight' was
placed on the mapped tier"). With libr3d mounted, `--debug-selection` at spec 0
+ mm 0 named EXACTLY these two ops as host-only and zero missing device bands —
the scope was the pair, no more.

`r3d_gemm_fp8a8.hip`: the m64 WMMA skeleton with an exact e4m3→bf16 LUT on both
planes (products of decoded e4m3 are exact in bf16, so only the accumulation
order differs from ref and the kbench tolerance covers it). The scale law is
ref_gemm.h quant_dot verbatim: per 128-of-K group, sub = ascending f32 of exact
products, acc += (sub * sa) * sb — two flush multiplies in that order, sa from
the f32 a_scale plane, sb from the bf16 b_scale grid (128-row block grid OR
per-row grid, make_scales' rblk rule). The weight scale CANNOT fold into the
bf16 fragment the way mxfp4's e8m0 exponent did: rtn scales are arbitrary
floats, not powers of two.

`gemm_nt_q_gated` composes rather than fuses: the tile writes its bf16 [M,N]
gate_up plane into the launch scratch (hook: 64*N*2, band M <= 64 — recorded
drafter bands are exactly M ∈ {1..64}), the ported gated_quant_fp8 epilogue
closes it into the code pair. Byte-identical to the unfused pair by
construction (the epilogue is that kernel), which is the whole argument for
composing first and fusing later.

Evidence: fixture `fp8_9b.rkb` (1461 cases, recorded from the container's
skeleton — the skeleton keeps fp8 planes whole, so weights are the real ones);
attribution home (libr3d+libref only): gemm_nt_q 536/536 worst rel_l2 3.7e-04
vs 5e-3 tol; gemm_nt_q_gated 56/56 worst 8.6e-06 vs 3e-2; 0 OOB; stock
1995/1995 and full 4713/4713 unchanged. Serve acceptance: `--placement
all_vram --num-speculative-tokens 0 --mm-max-patches 0`, "9.29 GiB placed from
9.29 GiB of container read", no host-only line, health 200, "capital of
Australia" -> Canberra. Decode 9.8 tok/s = ~98 GB/s effective: the m64 arm
covers ~27% of the card's bandwidth — the decode-shaped tile is the open perf
lever (llama.cpp Q4_K_XL same card: 75.8 t/s no-draft).

Two schema/pitch traps paid for:
- **operand ROLES differ per op**: gemm_nt_q's a_scale is optional,
  gemm_nt_q_gated's is REQUIRED. Copying the plain table into the gated schema
  made the loader refuse libref entirely at load ("two different schemas ...
  refusing libref") — every libref row vanished with it, not just the one op.
- **band kernels index through BYTE offsets**: the epilogue's gu/ob pitches are
  bytes (the kernel offsets a byte pointer, then indexes elements), while the
  f32 scale pitch is element units. Passing element pitches strides every row
  by half a row — silent, plausible-looking garbage.
