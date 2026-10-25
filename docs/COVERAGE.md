1|# libr3d coverage plan
2|
3|Sources: radiance v1.11.0 — `docs/PLUGIN.md` §10, `docs/OPS.md`, `arch/common/*` (the op set a
4|bf16 dense model issues), `core/` (the sampler ops the engine issues), `libref/` (the oracle).
5|
6|## Ground rules
7|
8|- Every op in the conventional vocabulary has a libref host fallback and a kbench oracle. Every
9|  kernel we ship is checked against libref with `rad-kbench` on recorded geometries before it is
10|  considered done.
11|- Ops `docs/OPS.md` marks **device-only** have no oracle; they are deferred and will need
12|  hand-written validation.
13|- Partial coverage is legal: unresolved ops are errors only if issued. An incomplete libr3d
14|  simply doesn't serve models whose op set it hasn't reached.
15|- RDNA3 has no native fp8 (E4M3) hardware. fp8/w4a8/fused-quant rows are out of scope until the
16|  bf16 path works; when they come, expect conversion-at-boundary designs, not emulated MMA.
17|
18|## Tier 0 — load smoke (this skeleton)
19|
20|Zero rows. Plugin loads on gfx11, ABI 22 accepted, selection unchanged.
21|Check: `rad-info --plugins` on the card.
22|
23|## Tier 1 — bf16 minimum viable model (goal: decode a 2B-class bf16 model on RDNA3)
24|
25|Model ops issued by the bf16 dense path (`arch/common/rad_block_attn*.h`, `rad_block_mlp.h`,
26|`rad_arch.h`):
27|
28|| op            | role                                   | notes |
29||---------------|----------------------------------------|-------|
30|| embed_lookup  | token -> row                            | trivial gather |
31|| gemm_nt       | qkv / gate+up / down / lm_head          | the big one: wave32 WMMA bf16 on gfx11 |
32|| rmsnorm       | pre-attn / pre-mlp / final              | |
33|| rope          | rotary positions                        | neox + default variants |
34|| attn_paged    | decode + prefill attention              | second big one; per head_dim/gqa geometry |
35|| kv_store      | append to paged KV cache                | |
36|| act (silu_mul)| SwiGLU activation-multiply               | confirm exact op name/graph via --debug-graph |
37|| cast          | dtype edges                             | |
38|| all_reduce    | TP only                                 | Tier 2 (TP1 needs no collective) |
39|
40|Sampler ops issued by core (`sample_*`), ~11 in the default chain:
41|`sample_temp, sample_topk, sample_topp, sample_minp, sample_typical, sample_penalties,
42|sample_dry, sample_xtc, sample_mask, sample_merge_topk, sample_pick, sample_argmax,
43|sample_logprobs` — exact served subset to be confirmed with `--debug-graph` on the card.
44|
45|**Ported and oracle-green on the card (2026-10-19):** embed_lookup, rmsnorm, kv_store,
46|act_mul (all four acts), cast (bf16 copy / widen / narrow), rope (neox+gptj; multi-component
47|modes and yarn/llama3 refused by constraint), gemm_nt (bf16, correctness-first tile),
48|attn_paged (decode + windowed, correctness-first).
49|Remaining Tier 1: the sampler chain, then the MiniCPM5-2B bf16 serve test. WMMA speed
50|variants of gemm_nt are a Tier-1.5 optimization, not a gate.
51|
52|## Tier 2 — multi-rank and shapes
53|
54|`all_reduce` / `all_gather` (+ `_host` twins — consumer P2P needs them even on RDNA4, so the
55|design is proven), wider head_dim/gqa geometries, prefill tiles.
56|
57|## Tier 3 — quantised vocabulary
58|
59|fp8 GEMM families via boundary conversion (OCP E4M3 bytes per the core's convention), i8a8,
60|w4a16-style weight formats that avoid fp8 MMA. Device-only fused ops: last, with bespoke tests.
61|
62|## Verification loop (per kernel)
63|
64|1. `docker` build of libr3d against pinned radiance (no card).
65|2. Host-tier tests (`ctest -LE 'gpu|oot'` upstream patterns; kbench fixture tier).
66|3. On rdna3-dev with the RDNA3 card: `rad-kbench` vs libref over recorded geometries, red-zone OOB;
67|   then `rad-info --plugins`; then serve MiniCPM5-2B bf16.
68|
69|## Serve on the 7900 + decode GEMV (2026-10-19, later)
70|
71|Supersedes the "no full-model serve has run" bullet in the section below. MiniCPM5-2B bf16 serves on
72|the 7900 (`libr3d-serve` container, radiance-dev:v1.8.1, :9005, `--radiance-home
73|/rhome:/opt/radiance/share/radiance` with `build/radiance_home/kernels/libr3d.so`).
74|Measured decode: **0.35 -> 11.1 tok/s** (96-token request), warmup 22.5 s -> 0.71 s;
75|text coherent end to end. Two findings did that, one kernel and one contract:
76|
77|1. **The `M` parameter is arena capacity, not the row count.** The engine passes
78|   `M=8192` (the `--max-num-batched-tokens` step-batch capacity) on every call while Y
79|   carries the live rows: `[1,N]` at decode, `[12,N]` at prefill. `ref_gemm.cpp` never
80|   reads the parameter -- it derives `M = numel(Y) / N`. A row that trusts the parameter
81|   computes a full prefill batch on every decode token *and writes garbage into arena
82|   rows past Y's own* -- invisible to every guard, because the arena is exactly that
83|   big. libr3d's gemm entries now mirror ref's derivation. This -- not bandwidth -- was
84|   the whole 0.35 tok/s: profile-ops showed decode calls taking the same 34 ms as
85|   prefill calls at identical geometry.
86|2. **Decode GEMV row** (M==1 band) in `r3d_gemm_bf16.hip`: one wave32 per output row,
87|   uint4 (8 x bf16) vector walks, `shfl_xor` butterfly reduce, flat `[row]` store
88|   (`y_pitch` is the between-M-rows pitch and M==1 has none -- scaling by it was a
89|   1.4 MiB OOB write fault kbench caught on the first case). The tile kernel keeps
90|   every other M. Guards: K%8, weight pitch %8, 16 B-aligned bases, else tile.
91|
92|Two more lessons, both from kbench on this row:
93|- **Wave width is a project-wide ABI pin.** `-mno-wavefrontsize64` is set per TU in
94|  CMakeLists; a wave64-shaped shuffle reduce (lane=tid&63, offsets 32..1) fails with
95|  rel_l2 ~= 1.0 on exactly the dispatched cases and nothing else. Derive lane math
96|  from the pinned width, not the hardware default.
97|- **kbench must own the card.** Running it while the serve container held the 7900
98|  produced 28 false `refused ... device error` FAILs (queue faults from contention);
99|  a solo re-run of the identical build was 269/269. Check `docker ps` before believing
100|  a fault-laden sweep.
101|
102|GEMV reassociates through the butterfly, so M=1 cases now land at worst rel_l2
103|4.48e-08, not the tile's structural 0 -- that line is the GEMV's expected signature;
104|the M>1 bit-exact zeros remain the regression tripwire. logits_gemm's seven 7.1 GiB
105|fixture cases stay budget-skipped while co-tenant serve holds 25 GB of 31 GB host RAM; the
106|GEMV serves them live in the model (decode steps at `y=[1,130560]`, coherent output).
107|
108|## Verified state on hardware (gfx1100, RX 7900 XTX class, rdna3-dev ROCR index 0)
109|
110|2026-10-19, radiance-dev:v1.8.1 (radiance v1.8.1, ROCm 10.1):
111|
112|- **Load gate:** `rad-info --plugins` on `--home build3/radiance_home` reports
113|  `libr3d 0.1.0 kernels 3 kernel(s), 3 schema(s), built for gfx1100 gfx1101 gfx1102 gfx1103
114|  gfx11-generic` against `AQL backend, 1 device(s) [0:gfx1100]`, exit 0. The fat binary's
115|  gfx11-generic code objects load on gfx1100.
116|- **Oracle gates, all vs libref on the card over the 741-case fixture recorded from
117|  qwen3.8-27b-fp8:**
118|  | kernel             | result | red-zone |
119|  |--------------------|--------|----------|
120|  | rmsnorm_bf16       | 90 checked, 90 passed, worst rel_l2 2.395e-04 | 40 cases, 0 OOB |
121|  | embed_lookup_bf16  | 20 checked, 20 passed, worst rel_l2 0.000e+00 | 10 cases, 0 OOB |
122|  | kv_store_bf16      | NOT DEVICE-VERIFIED (see correction below: fixture is fp8-KV only) | - |
123|  | act_mul_bf16       | NOT DEVICE-VERIFIED (see correction below: fixture has no act_mul cases) | - |
124|  | cast_bf16_bf16     | reached in per-op runs (10 device cases); bf16→f32 / f32→bf16 pairs have no fixture cases | 0 OOB |
125|  | rope_bf16          | 17 device-rope cases, all passed in the 221/221 sweep | 0 OOB |
126|  | gemm_nt_bf16       | 269 checked, 269 passed, worst rel_l2 0.000e+00 | 28 cases, 0 OOB |
127|  | attn_paged_bf16    | 138 checked, 138 passed, worst rel_l2 0.000e+00 | 69 cases, 0 OOB |
128|
129|  gemm_nt ran alone at 269/269 **bit-exact** against libref across every recorded geometry
130|  (qkv / gate+up / down / lm_head shapes to 2^27). The zero is structural, not luck: the
131|  kernel walks k ascending per output — the oracle's exact chain — and bf16 x bf16 is exact
132|  in f32, so the tile-boundary reassociations the kernel *could* have made it never makes.
133|  Same rule as the copy-shaped ops: a future nonzero rel_l2 here is a regression signal, not
134|  rounding noise.
135|
136|  Full sweep `--op rmsnorm --op embed_lookup --op kv_store --op rope --op act_mul --op cast
137|  --max-bytes 12288`: **221 checked, 221 passed, worst rel_l2 2.395e-04, 0 wrote outside a
138|  red zone.** `rad-info --plugins` now reports 8 kernels / 6 schemas.
139|  Exact zeros are what the copy-shaped ops should give: a bf16 gather/scatter that matches the
140|  oracle bit-for-bit cannot round differently. Any future nonzero there is a regression signal.
141|
142|  attn_paged_bf16 ran at **138/138, all bit-exact**, after fixing a thread-local score max:
143|  only thread 0 updated it, so threads 1..head_dim-1 saw "no visible keys", took the zero-row
144|  path, and stored zero columns. The repro harness `tools/repro_attn.hip` named it in one run;
145|  the same class of bug as the rope store-address race noted under Pitfalls below.
146|  Like gemm_nt the zero is structural: max-subtract then sum in the oracle's order,
147|  so no reassociation to disagree about.
148|
149|  Final full-fixture sweep (all ops, default budget): **881 checked, 881 passed, 24 skipped,
150|  164 red-zone cases, 0 violations.** The 24 skips are the 7-14 GiB embed/logits_gemm cases
151|  that did not fit -- see the RAM pitfall below. `rad-info --plugins`: 9 kernels / 7 schemas.
152|
153|- **Sampler chain + band ops (2026-10-19, same fixture/card).** `sample_temp/mask/argmax/topk/
154|  topp/minp/pick` and `add/gather_rows/quant_act_fp8/gated_quant_fp8` joined after the 881
155|  sweep. The quantisers are unavoidable even for a bf16 trunk: the model quantises MLP
156|  activations per 128-group and those ops are weightless, so the planner cannot push them to
157|  the host and refuses the graph at startup. fp8 here is *storage*, not arithmetic -- only the
158|  e4m3 encoder is touched, which RDNA3 does not need hardware for. Per-op runs: 20 + 16 + 88
159|  + 20 + 28 + 14 x 6, **all passed, all worst rel_l2 0.000e+00, 0 red-zone violations.**
160|  Full-fixture regression after the fixes: **1009 checked, 1009 passed, 24 skipped (same
161|  big-shape RAM skips), 0 violations.** `rad-info`: 21 kernels / 19 schemas on `[0:gfx1100]`.
162|- **Pitch pitfall (the one that ate a session):** `RadTensor::stride[0]` is in ELEMENTS. Any
163|  kernel that walks a `uint8_t*` row base as `base + row * pitch` must be handed a BYTE pitch;
164|  `row2_pitch` stays element-unit and each launch converts with `rad_dtype_bytes`. f32/f8
165|  (1 elt == 1 B) and M==1 (row 0, offset 0) are immune to the mix, which is why the failure
166|  signature -- "bf16 with rows>1" -- is the tell. A write overrun trips the red zone loudly;
167|  a pure READ overrun (gated_h taking `gub - gu` as an element base) only shows as a fatal
168|  queue fault mid-sweep. Pass row bases as pointers so the mix is unrepresentable.
169|- **Not yet done:** no full-model serve has run on this card. An interrupted session's closing
170|  report claimed MiniCPM5-2B was serving on the 7900 with embed fixes at commit `0949f0f` --
171|  that hash exists in no repo, the weights were never downloaded, and no serve artifact exists
172|  on either machine: the wrap-up was hallucinated at interruption. Treat that claim as the
173|  lesson it is (reconcile closing reports against git and the filesystem before believing
174|  them), and the gate below as genuinely open. The first run of MiniCPM5-2B bf16
175|  (weights are on neither box yet) is the remaining Tier-1 exit test; op coverage says the
176|  graph should resolve end to end.
177|
178|- **RAM pitfall (rdna3-dev):** co-tenant serve's host-resident model holds ~25 GB of rdna3-dev's 31 GB, so the
179|  kbench auto budget drops to ~1.7 GiB and big embed/logits cases skip. Forcing them with
180|  `--max-bytes 12288` while co-tenant serve runs OOM-kills the container (exit 137, no dmesg line --
181|  `ps aux --sort=-rss` is how to spot it). Either run big sweeps while co-tenant serve is down, or
182|  re-run the ops that previously passed at 12288 (embed) with the auto budget and accept skips.
183|
184|Reproduce (on rdna3-dev, after `cmake -B build3 -S . && cmake --build build3`):
185|
186|```
187|docker run --rm --shm-size=4g --device /dev/kfd --device /dev/dri --group-add video \
188|  -v $PWD:/work -e ROCR_VISIBLE_DEVICES=0 radiance-dev:v1.8.1 bash -lc '
189|  mkdir -p /tmp/home/kernels /tmp/home/architectures &&
190|  cp /opt/radiance/share/radiance/kernels/libref.so \
191|     /work/build3/radiance_home/kernels/libr3d.so /tmp/home/kernels/ &&
192|  rad-kbench --home /tmp/home --fixture /opt/radiance/share/radiance/kernels.rkb --op rmsnorm'
193|```
194|
195|Pitfalls this settled:
196|- `rad-info --plugins` takes the home as `--home PATH`; a `RAD_HOME` environment variable is
197|  silently ignored and the output lists the install's plugins only — an empty grep is not a
198|  load failure.
199|- A loaded kernel plugin must declare the schema of every op it implements; libref is a
200|  kbench oracle, not a search-path plugin, so on RDNA3 libr3d declares its ops' schemas
201|  itself, copied verbatim from libr4d so co-load `schema_same()` passes.
202|- The search path requires `architectures/` to exist beside `kernels/` (empty is fine).
203|- On rdna3-dev the 7900 is ROCR index **0** (`[0:gfx1100 1:gfx1201 2:gfx1201]`), not DRM card1.
204|- kbench needs ~3 GiB free in the container; case budget auto-shrinks and names skips.
205|- `--max-bytes` is plain MiB (`12g` parses as 12 MiB). The 27B fixture's embed_lookup cases need
206|  ~7.1 GiB each; `--max-bytes 12288` runs all 20 on the 24 GiB card.
207|- A kbench PASS can be vacuous: a case whose positions are 0 rotates by the identity, so an
208|  addressing bug in rope hid there while every sampled-position case failed. Prefer a repro
209|  with unit impulses (`tools/repro_rope.hip`, `UNIT=1`) -- the output element then names the
210|  angle the kernel actually used.
211|
212|## The rope race (2026-10-19), worth remembering
213|
214|The first rope port dropped the head axis from the store address (`xr = row + h*head_dim`):
215|every head's threads read-modify-wrote head 0's pairs, compounding rotations into
216|`cos(k*ang)` with k set by the scheduler -- large, deterministic-looking, slightly
217|run-stable rel errors that grew with M. Neither the row-stride fold (ref_gemm.h `row_stride`'s
218|trailing-dimension walk, kept: rank-3 `[M, heads, head_dim]` qkv is expressible) nor the
219|position-stride read (ref's `offlin`, kept) was implicated; the fixture descriptors were plain
220|rank-2 stride-1. The decisive tools were the standalone repro and a direct compile of the
221|kernel outside the plugin .so, which together split "plumbing vs code" in one step.
222|
223|## The coverage correction (2026-10-19, second round)
224|
225|`logits_gemm_bf16` joined (the lm_head edge: gemm_nt's tile under the `n_vocab`/`n_embd`
226|spelling; the tile's Y gate already admitted the f32 logits plane). Adding it and reading the
227|full-sweep report exposed what the earlier per-op "passes" had hidden. The fixture records
228|`op -> rows run`, not just pass counts, and per-op runs had been read as "our row ran" without
229|checking that line. It had not, for four rows:
230|
231|| row | fixture reality | status |
232||---|---|---|
233|| `kv_store_bf16` | every case is `kv_dtype=fp8_e4m3` (qwen3.8-27b-fp8 serves fp8 KV) | compiled + loads; never device-verified |
234|| `act_mul_bf16` | no `act_mul` cases at all (this deployment fuses act+quant into the bands) | compiled + loads; never device-verified |
235|| `cast_bf16_f32` / `cast_f32_bf16` | only `from=bf16 to=bf16` recorded | bf16→bf16 pair IS verified; the f32 pairs are not |
236|| `logits_gemm_bf16` | `lm_head` recorded as affine E4M3: 2 planes (codes:f8e4m3 + scale:f32) | compiled + loads; never device-verified |
237|
238|Earlier claims of "kv_store 20/20 bit-exact, act_mul 44/44 passed" were libref replaying its
239|own recorded answers: our rows had been declined, and the "0.000e+00" was libref-vs-itself.
240|The decline mechanism, from `core/plugin/select.cpp`:
241|
242|- A DEVICE row with no `layout` hook **"reads the weight as stored"**: `weight_refusal()`
243|  declines it whenever the case's weight encoding spans >1 plane. The engine would otherwise
244|  hand an fp8 codes plane to a kernel reading bf16 — plausible-magnitude garbage. libref's
245|  host shim dequantises, so the cases still check, vacuously for us.
246|- Constraint declines are honest geometry: fp8 KV and fp8 heads are not this row's contract.
247|- Probe that settled it (kept in `tools/kf_probe.cpp`): loads the fixture via
248|  `rad::Kfixture::load` and dumps each operand's encoding. Link it against
249|  `tools/kfixture.cpp` with `-Wl,--unresolved-symbols=ignore-all` (the draw helpers it
250|  drags in are never called by `load`).
251|
252|Consequence, stated plainly: the qwen3.8-27b-fp8 fixture CANNOT gate a bf16-head or bf16-KV
253|row. Those four rows graduate on the serve milestone — record a fixture from the bf16
254|container (`rad-kbench -m minicpm5-2b... --record FILE`) and they resolve against plain bf16
255|planes for the first time.
256|
257|Hygiene adopted from this:
258|- Read the report's "not reached: libr3d N" line in EVERY sweep; a pass count says nothing
259|  about which library ran. Coverage ratchet: `--coverage DIR --coverage-write` locks the
260|  reached set (22 libr3d rows live in `~/projects/libr3d/logs/coverage/` on rdna3-dev); a later
261|  `--coverage DIR` run fails if any locked row silently drops out.
262|- Big sweeps while co-tenant serve runs (25 GB host RSS of 31 GB): a temporary swapfile works —
263|  `sudo fallocate -l 20G /swap-kbench && sudo chmod 600 /swap-kbench && sudo mkswap
264|  /swap-kbench && sudo swapon /swap-kbench`, run with `--max-bytes 12288`, then `swapoff`
265|  and delete. Verified: all 7 logits cases passed as libref self-checks that way at ~50 s
266|  wall without OOM.
267|
268|## Prefill-logit KLD gate — `tools/kldgate/` (2026-10-20)
269|
270|Why: stew675/llama-cpp-rdna-boosts#113. A default-on bf16 prefill kernel shipped with mean
271|KLD 0.03 past every text/coherence gate: greedy and sampled text read the decode distribution
272|*from the post-prefill state* and are blind to a uniform prefill shift. Their doctrine, worth
273|adopting whole: the model-level KLD is the alarm, the per-op parity is the truth (the same
274|per-op perturbation spans >100x in model KLD depending on the surrounding pipeline), and the
275|gate arms are base-vs-prior-arm, not base-vs-rebuilt-oracle.
276|
277|Design, adapted to radiance (no perplexity tool, no prompt_logprobs over HTTP): two serve
278|arms on the same card and weights, one at a time, A/B'd through the llama.cpp-style
279|`/completion` with `temperature 0, n_probs 16 (engine cap; 32 is a 400), post_sampling_probs
280|false`. The base arm (scalar prefill tile via `R3D_NO_WMMA=1`) free-runs the 11-prompt corpus;
281|the test arm is then asked to *re-score the base's trajectory under teacher forcing*
282|(`score` mode): position i is served the exact text prompt + base tokens 0..i-1, so every
283|compared position shares identical context. The forced-context rebuild is byte-exact (token
284|`bytes` arrays through an incremental UTF-8 decoder) with a `/tokenize` count guard — a
285|retokenization desync truncates that trajectory and counts the drop rather than comparing
286|mismatched text. The prefix cache makes each forced request incremental, so the pass costs
287|~190 ms/position.
288|
289|Gate: mean KLD <= 2e-3 AND p99 <= 5e-3 AND top-1 agreement >= 0.98. Max is informational:
290|one razor-tie position amplifies through the union-KL renormalize (observed 0.029 at a single
291|position; the same-arm noise probe on that corpus maxes at 0.0016), and the defect class is a
292|*mean* shift, not an outlier. Noise floor of the harness itself (same arm scored against
293|itself): mean ~2e-4, max ~6e-3 — serve-to-serve variance of fresh vs incremental prefill
294|under the prefix cache. The gate sits an order of magnitude above the floor; #113-class sits
295|one above the gate.
296|
297|First run through the gate (WMMA m64 tile vs scalar base, 1257 positions): mean 3.1e-4,
298|p99 2.25e-3, top-1 0.993 -> PASS. About 9 forced-context argmax flips across the corpus, all
299|near-ties (-1.9318 vs -1.9292 class): rounding-order purity delta, not drift. Artifacts:
300|`/tmp/kldgate-20261020-083741/report.md` on rdna3-dev.
301|
302|Lessons, kept because they were paid for:
303|
304|- Comparing free-running arms by position index is invalid the moment the trajectories part:
305|  the first (broken) run reported mean KLD 2.84 — distributions conditioned on *different
306|  texts*. Position-0 forensics settles it in one probe: identical top-16 overlap with
307|  different argmax means the instrument is comparing apples to oranges, not that the kernel
308|  drifted.
309|- The same-serve self-test (`collect` then `score` against one running arm) is the instrument's
310|  self-check: it must return ~0 KL and few dropped positions. It caught a bytes/str plumbing
311|  bug on the spot and defines the noise floor the thresholds must clear.
312|- radiance HTTP: `n_probs <= 16`; `prompt_logprobs` is accepted and ignored on both
313|  `/v1/completions` and `/completion` — full-vocab prefill KLD needs it upstream (worth
314|  requesting next time we talk to the radiance maintainers, alongside the preemption report).
315|- Retokenization desync at BPE join boundaries is real (the mid-expression `cpp-kernel`
316|  prompt desyncs at position 13): guard, drop, count — never score a position whose token
317|  identity is not verified.
318|- A GPU container's `docker rm -f` can lag its kill by seconds (device teardown); `docker
319|  run` in the next breath conflicts. Wait for `docker inspect` to report the name gone.
320|
321|## Serve admission cap: --max-num-seqs 8 (2026-10-20)
322|
323|The serve recipe ran with radiance's default `--max-num-seqs 256` against a KV pool that
324|backs 27.9 full-length sessions, and radiance does not preempt gracefully (V3/V4 overload
325|wedged the engine + GPU). Capping admission at 8 was meant as pure safety. It moved real
326|memory and real speed:
327|
328|- KV worst case: 11% backing (256 x 16384) -> **100% backing** (8 x 16384); oversubscription
329|  is now structurally impossible. The 18.31 GiB pool itself is VRAM leftover and unchanged.
330|- Per-step fixed structures scale with seq slots: pinned host staging ring 51.83 -> 2.27
331|  MiB; device slab 17.24 MiB -> 772.5 KiB (block tables 17.0 -> 0.5 MiB).
332|- **Decode: 11.1 -> 23.6-27.4 tok/s** sustained (500/369-token greedy probes; the 40-token
333|  probe read 58.9, short runs flatter). Decode cost had been dominated by per-step fixed
334|  work scaling with the step-batch arena, not by the GEMMs — a second confirmation of the
335|  M-parameter lesson: arena capacity leaks into timings wherever the engine trusts it.
336|
337|`SERVE_ONLY=1 bash tools/kldgate/run.sh` now single-sources the birth recipe. Note when
338|benchmarking old numbers: anything measured before this flag carries a ~2x decode handicap.
339|
340|## Prefill bottleneck is attention, not GEMM (2026-10-20, two-probe fit)
341|
342|Measured on the live serve, fresh (unwarmed) prompts, `max_tokens=1`, seqs8 build:
343|
344|| prompt tokens | TTFT |
345||---|---|
346|| 981  | 2.85 s |
347|| 1961 | 10.68 s |
348|
349|The 3.75x for a 2x prompt is the quadratic signature. Fit `T = a*M + b*M^2`:
350|a = 0.365 ms/token (linear), b = 2.59 us/token^2 (quadratic). At M=981 that splits
351|2.49 s quadratic / 0.36 s linear -- 87% of prefill is the attention term, and the
352|linear term (all GEMMs, ~3.9 TFLOP) already runs at ~11 TFLOP/s effective. The
353|quadratic term runs at ~133 GFLOP/s: our `attn_paged` device kernel is correctness-
354|first scalar FMA (grid = row x head blocks of head_dim threads, each block rescans
355|the whole KV history, no tiling, no tensor units).
356|
357|Rejected hypotheses, in order: (1) host AVX attention was ruled out by `docker stats`
358|during a 10.7 s prefill: 0.4-0.8% CPU. (2) The engine's `--profile-ops` dumps nothing
359|to docker logs (table goes elsewhere) and its synchronisation distorts what it does
360|measure; the two-point scaling fit settles shape questions in 30 s with zero rebuilds.
361|
362|Consequences for the queue: WMMA flash-attention moves to first (it was third); the
363|GEMM staging pipeline drops to second -- it optimizes the term that is 13% at 1k ctx.
364|It also retroactively explains the WMMA-vs-scalar GEMM tie: a tile swap inside the
365|13% term cannot move a total dominated by the 87% term.
366|
367|Baseline for the FA work: 2.85 s @ 981 tok (fresh context; a prefix-cached repeat is
368|not a valid prefill baseline).
369|
370|## WMMA flash-attention shipped: 2.1-3.1x prefill (2026-10-20)
371|
372|The two-pass WMMA tile (`r3d_attn_paged_wmma.hip`) is live on the serve. Measured
373|fresh-context TTFT, same seqs8 recipe:
374|
375|| prompt tokens | v1 scalar | WMMA tile | speedup |
376||---|---|---|---|
377|| 981  | 2.85 s  | 1.33 s | 2.14x |
378|| 1961 | 10.68 s | 3.46 s | 3.09x |
379|
380|Decode also moved 24-27 -> ~37 tok/s (the tile serves M==1 rows too: same bytes,
381|tensor cores make the wasted FMAs free). kbench attn 138/138 worst rel_l2 3.135e-03
382|(v1 arm unchanged at 0.000e+00); full sweep 1102/1102, 7 budget-skips, no collateral.
383|
384|### The bug that hid the tile for two sessions: numeric bf16 staging
385|
386|`bf16(uint16_t)` on `rocwmma::bfloat16_t` CONVERTS the integer instead of reinterpreting
387|its bits: the fp8 decode of code 0xf1 (-144, bits 0xc310) staged as bf16(49936) = 49920.
388|Every Q/K/V/P fragment staging site poisoned its own tile; scores hit 1e11, outputs 8e8.
389|Sibling of the known trap (casting a bf16 fragment TO short truncates numerically).
390|Rule: fragment staging writes raw BIT patterns (`bf16_store`), never a constructed bf16.
391|Checked the gemm m64 TU: it stages uint16 directly, clean.
392|
393|### Why the failures read as "passes" for so long: the vacuous fallback
394|
395|Two layers of silence stacked up:
396|
397|1. A debug build added device `printf` to this kernel; printf + a 41 KB static-LDS
398|   launch fails with hipErrorIllegalState(401) on ROCm 10.1 (probe: launch alone is
399|   fine at 41 KB; adding printf alone makes it 401). The launcher turned the launch
400|   error into RAD_E_DEVICE, the entry fell back to v1, every case served bit-exact --
401|   a "pass" that never ran the tile. Device printf is not usable in big-LDS kernels;
402|   stage intermediates to a global dump buffer instead (repro `REPRO_DBG=1` arm).
403|2. In kbench the fp8 rows failed while the bf16 rows "passed" -- but head_dim=256
404|   fixtures are refused by the tile gate (v1 serves them, rel 0.000), so a bf16 pass
405|   in the report does not prove the WMMA tile ran. Check for nonzero rel_l2 in the
406|   WMMA arm vs 0.000 in the R3D_NO_WMMA arm before believing a WMMA pass.
407|
408|Triage tools added: `tools/probe_l401*.hip` (launch-state bisect by build axis:
409|arch list / dlopen / LDS size) and `tools/probe_fa_main.cpp` (launch the real tile
410|through its own launcher, standalone).
411|
412|## 2026-10-20: gemm m64 k-slice pipeline (DBUF) — GEMM staging retarget
413|
414|After the attention fix the prefit quadratic moved to the GEMM linear term's stall:
415|the m64 tile serialised stage -> barrier -> WMMA -> barrier per k-slice, so every
416|slice ate the full operand global-load latency. Ported the structure of
417|llama-cpp-rdna-boosts block 06 (MMB, gfx1100-validated): next slice prefetched
418|global->registers during this slice's WMMA, A staging double-buffered (5 KB total
419|LDS, occupancy unchanged), B deliberately NOT double-buffered -- block 06 measured
420|B's second buffer as a loss (extra LDS traffic exceeds the barrier it removes).
421|
422|Measured on the card, same binary, env switch (R3D_GEMM_NO_DBUF=1 = old schedule),
423|unique uuid prefixes, fresh serve each arm:
424|
425|| probe          | single-buffer | DBUF pipeline | speedup |
426||----------------|---------------|---------------|---------|
427|| 981 tok median | 0.356 s       | 0.289 s       | 1.23x   |
428|| 1961 tok median| 0.691 s       | 0.550 s       | 1.26x   |
429|
430|Distributions do not overlap (dbuf max 0.300 < single min 0.327 @981). kbench
431|gemm_nt 269/269 both arms, worst rel_l2 identical to 4 digits (5.164e-04): the
432|schedule moves LDS write timing only, arithmetic is bit-identical, which is why no
433|separate KLD run was needed. `R3D_DEBUG_GEMM=1` prints the schedule name at launch
434|(the prove-the-kernel-ran line); logits_gemm fixtures stayed budget-skipped (the
435|host-radiance co-tenant holds the RAM) but share gemm_nt's launch path verbatim.
436|
437|Side observation: today's re-measured baselines (0.36 s @981) run ~4x under the
438|historical post-FA note (1.33 s) -- co-tenant and box state differ from that
439|session; another reason A/B must be same-day paired runs, never historical deltas.
440|
441|## 2026-10-20: the WMMA attn tile is latency-bound, and long-ctx decode collapses
442|
443|Serve-side measurement (MiniCPM5-2B fixture, 42 layers, full-attention, KV 1024
444|B/token, --bench kbench timing table from the shipped fixture pack):
445|
446|| ctx (tokens) | decode rate | decode step | prefill TTFT | prefill quadratic share |
447||--------------|-------------|-------------|--------------|-------------------------|
448|| ~30          | 48.2 tok/s  | 21 ms       | -            | -                       |
449|| 2048         | 11.2 tok/s  | 89 ms       | 0.578 s      | ~0.06 s (10%)           |
450|| 4075         | -           | -           | 1.284 s      | ~0.20 s (16%)           |
451|| 8200         | 3.1 tok/s   | 323 ms      | 3.671 s      | ~1.38 s (38%)           |
452|
453|Host CPU 0.1% during the 8k decode (docker stats): every ctx-proportional ms is
454|DEVICE time -- no AVX fallback anywhere in this. Decode step is linear in ctx
455|(0.038 ms per ctx token); the prefill quadratic is the KV re-read per 64-row
456|query tile (~M/2 x ctx KV bytes), and the observed 1.38 s at 8k implies ~15 GB/s
457|for those reads against a card doing 500-700 GB/s cold.
458|
459|kbench `attn_paged --bench --timing-only` attributes it exactly. The tile row
460|(attn_paged_bf16, confirmed reached) sustains, at steady deep-ctx cases:
461|hd=128 rows 2.6-15 GB/s; hd=256 rows 0.5 GB/s; and even the small-ctx prefill
462|cases never exceed ~500 GF/s -- ~100x under WMMA peak. Root cause is the tile
463|structure, not a fallback: one block per (head, query tile) -- a whole q_len=1
464|layer runs on only n_head blocks of 96 CUs; 32-key steps with ~4 __syncthreads
465|each; single-buffered K/V staging (load->barrier->mma->barrier fully serialised);
466|at hd=256 the Q+KV+S LDS carve forces one block/CU, matching its 30x deficit vs
467|hd=128. The old "2.1-3.1x from WMMA" claim stands only against the scalar
468|reference; the tile itself is the long-ctx bottleneck.
469|
470|Fix directions (next milestone, ranked): (1) DECODE split-K -- the op schema
471|already carries a `splits` parameter; stream KV chunks in parallel blocks and
472|combine (flash-decode), turning 24 blocks into ~1000 and the 8k step from
473|323 ms toward its ~1 ms bandwidth floor; (2) PREFILL tile redesign -- kBN 128+,
474|double-buffered staging (gemm DBUF lesson applies), fewer barriers, softmax on
475|fragments where possible, occupancy budget set per head_dim.
476|
477|MXFP4 feasibility note while we were in the vocabulary: radiance 1.8.1 loads OCP
478|MXFP4 natively (RAD_FP4E2M1; e2m1 codes + E8M0 scale per 32 K) and serves it at
479|op `gemm_nt_q` encoding `mxfp4a8` (fp8 e4m3 activations). `qwen3.8-27b-mxfp4.rad`
480|(19 GiB) is already on rdna3-dev, and the SHIPPED fixture pack kernels.rkb contains
481|recorded mxfp4a8 + fp8a8 gemm_nt_q cases -- the oracle loop for a Tier-3 tile
482|runs with zero model risk. Boundary-conversion design (e2m1*e8m0 -> bf16
483|fragments per k-slice, exponent add only) fits the existing m64 DBUF structure;
484|e8m0 scales are powers of two, foldable into the per-group accumulator scale.
485|
486|## 2026-10-20: decode split-K shipped — deep-ctx decode 3.0 -> 14.5 tok/s
487|
488|The fix direction above is now the decode band's default. Rows
489|`attn_paged_decode_bf16kv_h{64,128}` (priority 1100) take window 0, q_len <= 8:
490|the two-pass tile body, shared verbatim with the prefill row via
491|`r3d_attn_wmma_body.inc` under a `SPLITK` template parameter, runs once per KV
492|slice across `q_heads * n_seq * splits` blocks, stores un-normalized (O, m, l)
493|partials into the engine scratch arena, and a merge kernel (one block per
494|q head x query row) rescales them into the out row. Shape law on the host, one
495|closed form: `splits = min(192/(q_heads*n_seq), tiles_kv/2, tiles_kv, 64)`, `splits`
496|tunable 0 = the law. Kill switch `R3D_NO_SPLITK=1` runs the SPLITK=0 tail through
497|the same entry; `R3D_FORCE_SPLITS=N` pins the split count for benches.
498|
499|Same-binary serve A/B (pair-differenced decode rate, `tools/decode_probe.py`,
500|co-tenant state shared by both arms):
501|
502|| ctx    | tile arm  | split-KV arm | speedup |
503||--------|-----------|--------------|---------|
504|| 2048   | 10.6 t/s  | 33.5 t/s     | 3.2x    |
505|| 8192   | 3.0 t/s   | 14.5 t/s     | 4.8x    |
506|
507|The tile arm reproduces this file's own 3.1 tok/s measurement to the digit in
508|the same window — the baseline was not a drifted serve. The split arm still
509|runs 12 slices at 8k, not the ~1000 blocks the estimate imagined: the law caps
510|at tiles_kv/2 (a slice must hold >= 2 tiles) and 64, and q_heads*n_seq is the
511|grid floor — but 192 blocks over 96 CUs is already the difference between one
512|wave of latency chains and many. The bandwidth floor (~1 ms/step) remains open
513|ground for a bigger-kBN slice kernel; this is the 4.8x that the structure
514|supports without touching the tile body.
515|
516|Verification, in the order that caught things: `--record-rows`-derived fixture
517|(`tools/attn_rows.rkb`, 16 KiB, regenerate with
518|`rad-kbench --op attn_paged --record-rows ...` from the shipped pack) replays
519|the h128/h64 decode rows at 8-split and passes 8/8 (the SHIPPED pack is
520|head_dim-256-only — every attn case in it, so a pass column from it alone
521|proves nothing about these rows: the vacuous-pass trap, again, and kbench
522|says so itself: `2 of 194 rows ran`). `tools/repro_attn_splitk.hip` then
523|separates mechanism halves: it runs the row entry on a poisoned arena and
524|RE-MERGES the partials host-side against an independent oracle — device-out
525|and host-merge rel_l2 must both be small; which one fails says which half lies.
526|It is what pinned two plane-base bugs the card sweep showed but did not locate
527|(m-plane base omitted the splits factor; l-plane omitted it too; writer and
528|reader had AGREED on the wrong layout, so only a third implementation on the
529|contract caught it). `R3D_FORCE_SPLITS` 1/2/8/64 all green:
530|probe 8/8, kbench rows 8/8 (worst 0.000), full sweep with both fixtures and
531|`--all-rows`: 1083/1083, worst rel_l2 unchanged at 3.135e-03, 24 budget skips
532|(co-tenant serve holds 29 GiB of the 31). Serve now runs from `build6/radiance_home`
533|(build5 retired) with split-KV as the default arm; `/completion` sanity reads
534|coherent.
535|
536|Serve lesson for the day, worth keeping: the probe's first pair after a serve
537|start can read -518 tok/s (t8 > t108 while the engine warms its KV pool);
538|the tool reports every pair, and medians hide the sign of the nonsense rather
539|than deleting it. Discard cold-start pairs by eye, not by averaging.
540|
541|## 2026-10-20: prefill v2 tile schedule — refuted, kept dormant as R3D_ATTN_V2
542|
543|Direction (2) from the ranking above got its first attempt: `r3d_attn_wmma_body2.inc`
544|re-schedules the same tile — 256 threads (8 waves) instead of 128, global->register
545|prefetch of the next K/V slice overlapping the MMA step (the gemm DBUF lesson),
546|barriers cut from 4+6 per two passes to 2+3, bad-byte accounting moved strictly
547|after barriers. The refactor also moved the v1 body into the shared
548|`r3d_attn_wmma_body.inc` so both schedules and both bands (prefill, split-KV
549|decode) instantiate from one source.
550|
551|Arithmetic identity held exactly as a schedule change must: the oracle sweep
552|(`tools/attn_oracle_sweep.hip`, 7 geometries x GQA/window/unmapped-edge variants)
553|prints identical rel_l2 AND identical worst-case bit distances per case on both
554|arms, the split-KV probe passes splits 2/8/64 on both arms, and `tools/attn_bench.hip`
555|hashes v1 and v2 outputs bit-identically on all five serve-shaped geometries.
556|
557|The perf hypothesis did not. Same card, same binary, min over 15 reps:
558|
559|| hd/qlen/ctx    | v1 us | v2 us | v2/v1 |
560||----------------|-------|-------|-------|
561|| 128/64/1024    | 763   | 959   | 0.80x |
562|| 128/128/2048   | 1433  | 1853  | 0.77x |
563|| 128/981/8192   | 12010 | 21065 | 0.57x |
564|| 128/512/4096   | 2930  | 6859  | 0.43x |
565|| 64/512/4096    | 2002  | 2099  | ~1.0x |
566|
567|Read: barrier COUNT is not the serialization term. The v2 block (256 threads,
568|40.75 KB LDS at hd=128) packs 1 block/CU, so every barrier stalls the whole CU
569|with nothing else resident; v1's narrower blocks put two independent barrier
570|chains per CU, and their overlap was worth more than the barriers v2 removed.
571|Register-prefetching the slice also trades LDS pressure for VGPR pressure at
572|256 VGPR — no headroom left to hide latency either way. The occupancy axis is
573|the same one the split-KV win rode; making the BLOCK bigger is the wrong end of
574|it. v1 stays the default; `R3D_ATTN_V2=1` runs v2 in the same .so for postmortem
575|(the promotion pattern inverted: opt-in instead of opt-out).
576|
577|What direction (2) still needs, unchanged by this result: more BLOCKS at deep
578|prefill, not fatter ones — kBN 128+ slices and per-head_dim occupancy budgets,
579|ideally splitting query tiles across blocks the way decode split its KV.
580|
581|Two harness bugs worth remembering, both caught the sweep before the kernel
582|ever ran: absent optional operands are value-initialized `RadTensor`s whose
583|`data` is NULL inside the full positional array (a garbage `rank` from an
584|uninitialized descriptor makes numel helpers walk past the struct — the bench's
585|host-side segfault — and a garbage descale pointer poisons every score, faking
586|a geometry-dependent kernel failure that was really stack reuse); and the sweep
587|gate must be rel_l2-based like kbench's, with bit-distance as diagnostic only —
588|the oracle sums in double, so cancellation-heavy near-zero elements legitimately
589|sit thousands of bit-lines apart.
590|
591|## 2026-10-20: prefill v3 thin block — refuted; staging is instruction-rate-bound (vectorized: 1.2-1.64x)
592|
593|Two experiments rode one commit window. (a) `r3d_attn_wmma_body3.inc` (opt-in
594|`R3D_ATTN_V3=1`) tested the recorded direction directly: 32-row query tiles,
595|64 threads, 24,880 B LDS -> 2 blocks/CU at hd=128 where v1's 41 KB carve fits
596|one. Per-row arithmetic verbatim from v1; the FNV gate confirmed bit-identical
597|outputs on all five geometries. Timing: 0.35-0.80x -- deep cases 2.57x SLOWER
598|(11948 -> 30735 us at 128/981/8192). Reading the columns: v3 moved 2x the KV
599|bytes (half the rows share each K walk) at ~78% of v1's effective byte rate;
600|the occupancy bought nothing. The v2 post-mortem's "occupancy is the axis" was
601|overstated: BOTH fatter (v2) and thinner (v3) blocks leave the binding term
602|untouched, and v3's rate loss per doubled byte shows what it is -- the
603|2-byte-at-a-time gather. 4096 scalar global loads + 4096 scalar LDS stores per
604|K tile at hd=128; the tile runs its deep cases at ~90 GB/s because the copy
605|loop issues 2 bytes per instruction, not because it stalls.
606|
607|(b) `stage_kv_tile` walks EIGHT bf16 per `uint4` load/store when the copy is
608|provably bit-identical: bf16 KV, contiguous rows (`kv_sd == 1`), 16B-aligned
609|bases, 8-aligned live row bases; the original scalar loop serves everything
610|else, and fp8 stays scalar (per-byte decode). Q staging same treatment. LDS
611|pitches D+4 -> D+8 so 16B LDS stores stay legal at every row (41776 B total at
612|hd=128, still under the static ceiling). A copy cannot round differently --
613|the bench's hashes ARE the gate: all five geometries hash byte-identical to
614|the scalar arm's runs, which is also what exempts this from a KLD run
615|(schedule-class change with identity evidence).
616|
617|| hd/qlen/ctx   | v1 scalar us | vec us | speedup |
618||---------------|--------------|--------|---------|
619|| 128/64/1024   | 761          | 510    | 1.49x   |
620|| 128/128/2048  | 1463         | 893    | 1.64x   |
621|| 128/981/8192  | 11948        | 7368   | 1.62x   |
622|| 128/512/4096  | 2935         | 2043   | 1.44x   |
623|| 64/512/4096   | 2020         | 1671   | 1.21x   |
624|
625|Effective KV rates rise to 143-146 GB/s deep-ctx (still ~3.5x under the card).
626|The decode rows share the body and share the win's mechanism: split-KV probe
627|passes on splits 1/2/8/64 (device arm and host re-merge, 0 bad cells, 0 NaN
628|slots), oracle sweep PASS 7/7 including the window and unmapped-edge cases,
629|full kbench sweep (both fixtures, `--all-rows`, coverage ratchet) 1083/1083
630|worst 3.135e-03 unchanged, exit 0 -- with the serve container up and idle
631|(both sweep arms this session ran without the card to themselves and stayed
632|fault-free; contention false-fails remain a reason to distrust FAILs, not
633|passes).
634|
635|Kept: v3 stays behind `R3D_ATTN_V3=1` (same opt-in postmortem pattern as v2).
636|
637|### Serve-side A/B (promotion run, same session, paired by mount swap)
638|
639|build6 (scalar staging) vs build7 (vec), identical container/model/flags,
640|uuid-salted prompts, arms interleaved within one hour window:
641|
642|| probe            | build6       | build7       | verdict |
643||------------------|--------------|--------------|---------|
644|| TTFT @981 (x5)   | 0.255 med    | 0.248 med    | wash (attention share small at short ctx; distributions touch) |
645|| TTFT @1961 (x5)  | 0.485 med    | 0.446 med    | win, non-overlapping (0.484 min vs 0.454 max) |
646|| decode @2k (x1)  | 31.7 tok/s   | 38.8 tok/s   | 1.22x single-shot |
647|| decode @8k (x1)  | 14.0 tok/s   | 22.1 tok/s   | 1.57x -- lands exactly in the tile-bench predicted band (1.44-1.64x) |
648|
649|Decode window = cached-prompt pattern (n_predict=1 warm with cache_prompt=true,
650|then timed generation; `tokens_predicted` is the count field). The 2k first
651|sample early-stopped after 1 token: staging is bit-identical so it is not a
652|kernel signal -- filler text near an EOS-prone shape; decode probes must pin
653|gen length and treat early stop as a spoiled sample, not a rate.
654|R3D_NO_WMMA=1 arm at 8k prefill exceeded a 300 s client budget (minutes-slow
655|scalar attention) -- deep-context necessity of the WMMA row, quantitatively.
656|Promoted: serve now mounts build7 as default (0ce1007); no env flag needed
657|for rollback except rebuild-and-remount of build6 (kept on disk).
658|Open next: the gemm DBUF lesson (prefetch the next slice during this tile's
659|mma) is now the untried axis for attention -- at 143 GB/s there is ~3.5x of
660|card left on the staging path alone.
661|
662|## 2026-10-20: attention DBUF staging shipped — the gemm lesson transfers (1.19-1.27x tile, +21% serve decode@8k)
663|
664|Closed the open item above the same day. `body.inc` gained a `DBUF` template
665|arm: `ld_kv_regs` walks the NEXT k-slice's gather global->registers while the
666|current tile stages LDS and runs mma, so the memory stall shrinks to
667|max(load, stage) instead of load+stage. Two structural consequences:
668|
669|* Row addressing (`RB` row-bases, `CB` col-bad bytes) double-buffers on LDS --
670|  the prefetch must not scribble the addressing the in-flight mmas read. CB
671|  updates land strictly after the barrier that publishes the staged tile.
672|* Barriers drop to a3 in pass1 and a4 in pass2 (v1: a4/a6) -- a side effect
673|  of the restructure, not the point. v2 taught us barrier count is not the
674|  bottleneck; here the win is the overlap.
675|
676|Arithmetic is v1's chain verbatim (same fragment order, same LDS slots): all
677|five bench geometries hash byte-equal across arms, both directions.
678|
679|Paired tile bench (build8, same binary, old arm sandwiched, drift <1.3%):
680|
681|| hd/qlen/ctx | DBUF us | old us | DBUF/old |
682||---|---|---|---|
683|| 128/64/1024 | 368 | 461 | 1.25x |
684|| 128/128/2048 | 688 | 888 | 1.29x |
685|| 128/981/8192 | 6265 | 7512 | 1.20x |
686|| 128/512/4096 | 1698 | 2055 | 1.21x |
687|| 64/512/4096 | 1358 | 1673 | 1.23x |
688|
689|Deep-ctx KV rate 141 -> 171 GB/s (~2.9x of card still left; the next stall is
690|register-prefetch depth: one tile ahead, could be two).
691|
692|Serve A/B (build8 mount, arms differ only by `R3D_ATTN_NO_DBUF=1`, two rounds
693|alternating, uuid-poisoned prompts):
694|
695|| metric | DBUF | old | verdict |
696||---|---|---|---|
697|| decode 8k tok/s | 26.4/27.2/27.6/28.6 | 22.1/22.5/23.9 | non-overlapping, +21% |
698|| TTFT 1961 s | 0.442-0.467 | 0.466-0.507 | ~4% faster |
699|| TTFT 981 s | 0.254 | 0.263 | ~3%, near noise |
700|
701|Gates, both arms: kbench shipped pack 1075/1075 (worst rel_l2 3.135e-03
702|unchanged), attn_rows 8/8, oracle sweep 7/7, splitk probe spl 2/8 identical.
703|Kill switch `R3D_ATTN_NO_DBUF=1`; fp8 KV and odd shapes never enter the arm
704|(vec-path gate). Live serve: build8, DBUF default.
705|
706|Lesson for the skill: when a copy path is vectorized but still slow, the
707|next lever is pipelining the copy across the compute step -- re-tiling
708|geometry (v2/v3) was waste; the same DBUF pattern transferred from gemm to
709|attention staging nearly verbatim.
710|
711|## The 27B MXFP4 target: what the container actually asks for (2026-10-20)
712|
713|`/srv/models/rad/qwen3.8-27b-mxfp4.rad` (18.05 GiB, RAD2, arch `qwen35`, source
714|quant fp8_e4m3, converted from amd/Qwen3.8-27B-Quark-AWQ-MXFP4). Assessed with a
715|recorded fixture, not by reading code: `rad-kbench -m <skeleton> --record` needs
716|no card at all, because the skeleton keeps the tables and holes the data, and
717|recording draws weights from seeds.
718|
719|```
720|rad-kbench -m /models/qwen3.8-27b-mxfp4.rad --skeleton /scratch/skel27.rad   # 7 MB on disk
721|rad-kbench -m /scratch/skel27.rad --record /scratch/mxfp4.rkb --max-ctx 4096  # ~6 min, CPU
722|```
723|
724|**The model is a hybrid, and most of it is not attention.** 64 layers, full
725|attention every 4th (16 layers: head_dim 256, GQA 6, gated q/gate interleave,
726|partial rotary 64/256, MROPE interleaved); the other **48 are Gated DeltaNet**.
727|Drafter scope is five `dflash.blk.*` layers at layer indices 64-68 (head_dim
728|128, plus `attn_conv_base`/`attn_conv_proj` and `ffn_conv_base`/`ffn_conv_proj`),
729|with `dflash.draft_head.weight` at 2 bits (`u2*f16[1x128]-u8[1x128]`, 331 MiB).
730|Trunk `output.weight` is one fp8 tensor with 128x128 block scales (1.18 GiB),
731|not bf16 -- `logits_gemm_bf16` cannot serve it. Weight bytes: 71.5% MXFP4
732|(319 tensors, 12.89 GiB), 20.2% bf16.
733|
734|**Completeness, from the fixture (op x library matrix in the kbench report):**
735|
736|| declared graph | ops | libr3d holes |
737||---|---|---|
738|| drafter + vision tower | 41 | 15 |
739|| `--num-speculative-tokens 0 --mm-max-patches 0` | 31 | 9 |
740|
741|Holes with the drafter off: `gdn_chunk_scan`, `gdn_conv_update`,
742|`gdn_gated_rmsnorm`, `gdn_kkt_solve`, `gdn_recurrent_update`, `gemm_nt_q`,
743|`mul`, `rope_table`, `sigmoid`. The drafter/vision scope adds ten ops, six of
744|them holes (`attn_dense`, `dflash_select`, `gemm_nt_bias`, `grid_embed`,
745|`layernorm`, `scatter_rows`).
746|
747|**The MTP heads are avoidable at runtime, GDN is not.** Speculation off drops the
748|drafter's whole scope, its conv tensors and its 2-bit head from the declared
749|graph -- so the `mtp_num_hidden_layers 1` headers cost us nothing as long as we
750|serve `--num-speculative-tokens 0` (which our fixture serve already does). The
751|gate is the five GDN rows: 48 of 64 layers, and libr3d has no GDN family at all.
752|Two more GDN ops (`gdn_conv_prep`, `gdn_conv_recurrent_update`) are in the
753|program but absent from the matrix, because record declined them -- see below.
754|The graph column reflects ops with recorded cases, so it also under-reports
755|`embed_lookup`/`logits_gemm`, whose cases the memory budget skipped (each needs
756|>7 GiB).
757|
758|**A dtype disagreement worth reporting upstream.** The container stores
759|`ssm_dt.bias` and `ssm_a_log` as **bf16** (48 elements) while
760|`arch/common/rad_block_gdn_fp8.h` declares `w_dt_bias`/`w_a_log` as **RAD_F32**
761|`{n_head_v}`. kbench then declines 40 `gdn_conv_prep` cases ("operand 'dt_bias'
762|is a weight and the declaration bound none to it") and substitutes
763|`ssm_norm.weight` (128) for `dt_bias` (wants 48) on `gdn_conv_recurrent_update`.
764|The check that fires is in `tools/rad_kbench.cpp`, so it is a harness-side
765|substitution failure, and an engine load settled it: against the skeleton the
766|engine accepts the container outright (`model: qwen35 arch=qwen35
767|quant=fp8_e4m3 64 layers`, `declare: 1297 ops, 1600 weights, 64 buffers, 4 KV
768|groups`) with no complaint about those weights. The recipe casts only
769|`ssm_ab.weight`; no f32 cast for the other two is needed to load.
770|
771|That load also names the op list the engine declares for this trunk, which is
772|the authoritative hole inventory: **seven** GDN ops, not the five the fixture
773|matrix shows (`gdn_conv_prep` and `gdn_conv_recurrent_update` are in the program
774|but recorded no case), plus `gemm_nt_q`, `mul`, `sigmoid`, `rope_table`. Run
775|capped (`--memory 6g`, host backend, no device) it reached placement and stopped
776|on arithmetic, not on the model: static weights want 12.99 GiB of the 12.99 GiB
777|claimable, so KV gets nothing. Budget to remember for the real card: 12.99 GiB
778|static + 2.41 GiB activation arena + ~129 MiB step staging -- i.e. the 24 GB
779|gfx1100 holds it alone, not beside the fixture serve.
780|
781|**Feasibility, stated plainly.** On gfx1100 the trunk's shape arithmetic is not
782|the problem -- 15.6 GiB of weights fit the 24 GB card and GDDR6 at ~480 GB/s
783|bounds decode near 30 tok/s. The problem is that two whole op families are
784|missing (GDN chunked scan + triangular solve, and MXFP4 GEMM), and GDN is the
785|largest single body of work attempted in this library. **The same container is
786|already fully served on the R9700 cards**: the stock `stilldeadcode/radiance`
787|image ships `libr4d 0.5.0, 351 kernels, built for gfx1201` (absent from our
788|gfx11 build, which is why it looked missing), covering GDN, h256/gqa6 attention,
789|`gemm_mxfp4a8_{m64,decode,tiled}` and the drafter. Trying the 27B MXFP4 on rdna3-dev is
790|therefore a scheduling question -- one R9700 window on the stock image -- not a
791|kernel question. libr3d's MXFP4 row is still worth doing, but GDN first, or the
792|48 GDN layers run on AVX in host RAM regardless of how good the GEMM gets.
793|
794|## 2026-10-20 (late): the GQA fold at serve, and an A/B that tested nothing for two sessions
795|
796|**Fold serve A/B, the first real numbers** (same .so build10, arms differ only by
797|`R3D_NO_GQA`; pairs alternated twice to cancel drift; MiniCPM fixture, rdna3-dev):
798|
799|| arm | decode @8192 | decode @2048 | TTFT @1961 (min) |
800||---|---|---|---|
801|| plain (per-q-head, spl=12) | 27.0 / 27.2 tok/s | 41–42 | 0.510 s |
802|| **fold (per-kv-head, spl=64)** | **48.9 / 50.3** | **47.6–49.0** | 0.470 s |
803|
804|Non-overlapping distributions at 8k across both rounds; the mechanism is exactly
805|the promised one (KV streamed once per q-head group). TTFT unchanged (prefill is
806|GEMM-bound). Fold is the shipped default; serve now mounts build10.
807|
808|**The fold "serve crash" was a harness illusion.** The A/B's fold phases never
809|started a server at all: `docker rm -f` silently failed (its error was swallowed
810|by `>/dev/null 2>&1`), radiance ignores SIGTERM and its GPU teardown stalls
811|behind the co-tenant (`kfd_process_wq_release` hog warnings in dmesg right at
812|the arm switches), so the kill lands but the reap doesn't -- an EXITED corpse
813|keeps holding the container name and the next `docker run` dies on `Conflict`,
814|whose error the orchestrator's `tail -1` also dropped. The `Connection refused`
815|+ stale-`live step` output that read like a fold warmup hang was the *previous
816|arm's* idle server being torn down. Fixes: rm retry loop with a loud failure,
817|`--stop-signal SIGKILL`, `RUN FAIL` echo on docker run, orchestrator `tail -2`.
818|Rule: an A/B arm that never printed its launch proof tested nothing -- the
819|fold arm had been silently absent from every serve comparison before this.
820|
821|**Probe/gate repairs.** `repro_attn_splitk` (b) re-merged the poisoned arena at
822|SPL=1, where the entry routes to the SPLITK=0 tile arm and no partials exist --
823|a vacuous check that "failed" 1.000 while (a) was passing (the gate's grep
824|pattern dropped the (a) line, which is how a green suite shipped a scary
825|phantom). Now (b) skips at SPL=1 with a comment, every probe line prints (a).
826|`REPRO_CUS`/`REPRO_GQA` exercise the engine's real call shape (cu_seqlens on
827|every decode call); the new fold+cus+gqa8+qlen8 case passes **pre-fix too** --
828|`kBM=64` makes the loose bound exact through qlen 8 x gqa 8 -- but the cus
829|branch of `per_seq` now multiplies by gqa as its own comment always claimed,
830|which is the exact bound only when gqa > 8 (27B-class, and dflash2 at qlen>1):
831|insurance for the 27B track, not a live bug on this one.
832|
833|## 2026-10-20: gemm_nt_q mxfp4a8 shipped -- the quantised GEMM opens on RDNA3
834|
835|`gemm_nt_q_mxfp4` (src/r3d_gemm_mxfp4.hip) answers `gemm_nt_q` at
836|`dtype=mxfp4a8` -- 58 recorded fixture cases from `qwen3.8-27b-fp8.rad`, zero
837|model risk. RDNA3 has no fp8/fp4 hardware, so this is a boundary conversion on
838|the m64 WMMA tile, and the two decodes involved are exact in bf16 (e4m3
839|mantissa <= 3 bits, e2m1 <= 2). The scales never touch bf16 rounding either:
840|
841|* B fragments stage as `e2m1_bits + ((x - eref[n]) << 7)`: bf16 is a biased
842|  exponent, so the add IS the multiply by 2^(x-eref), exact while normal.
843|  `eref[n]` is the row's max non-NaN E8M0 (the `b_ref` role, derived in-kernel
844|  at block entry; the passed `b_ref` is accepted and ignored -- both host
845|  libraries ignore it too).
846|* WMMA accumulates raw products for exactly one 128-of-K activation group
847|  (8 slices of 16; 32 divides 128); the flush multiplies the partial by
848|  `sa[m, grp]` in f32 through LDS (`Acc`), and the store multiplies once by
849|  `2^(eref[n]-127)`. `(sub*2^d)*sa == (sub*sa)*2^d`: one rounding of the same
850|  mantissa, the same class the mma's opaque k-tree already lives in.
851|* NaN scale bytes stage their whole 32-column group as bf16 NaN. The oracle's
852|  `(sub*sa)*NaN` poisons the entire output even for an all-zero-code group
853|  (0*NaN is NaN); OPS.md's "treat as empty" is a permission for zero
854|  activation scales, not for NaN weight scales. Padding (m>=M, n>=N, k>=K)
855|  stages plain zeros -- distinguishable from NaN only because the prefetch
856|  carries a liveness bit.
857|
858|Why `Acc` lives in LDS: this ROCm's rocwmma has no coordinate fragment (no
859|`rocm_t`), so a register-side per-row multiply means hardcoding v_wmma's lane
860|layout. The 42 KB carve is 1 block/CU; the GEMM is weight-bandwidth-bound
861|(M=1 decode reads 64 MB of nibbles per call at N=5120 K=25600), and 80..2560
862|blocks over 96 CUs leave nothing a second resident block could exploit.
863|`R3D_GEMM_NO_DBUF=1` is the schedule arm, `R3D_NO_MXFP4=1` the serve kill
864|switch (the row refuses, the engine answers from libref, same binary).
865|
866|Gates, in order: loader (`rad-info --plugins` with libref co-loaded -- the
867|verbatim libr4d pGemmNtQ/oGemmNtQ dedupes clean); attributed sweep (libavx
868|excluded so counts attribute): **58/58 ok, worst rel_l2 3.21e-04 vs tol 3e-2**,
869|M=1 through 512, ragged M (65/129/258) included, launch proof line present,
870|58 cases guarded, 0 wrote outside; full pack with libavx+libref: **1850/1850,
871|0 failed**, 24 named skips all 7+ GiB embed_lookup/logits_gemm cases over the
872|1.36 GiB co-tenant budget (the same named-skips class as every recent run).
873|Serve build11: MiniCPM never issues gemm_nt_q -- the row is inert there; it
874|ships to be the 27B's door, which the GDN family still has to open.
875|
876|Fixture note (tools/kf_probe_q.cpp, build inside radiance-dev):
877|`g++ -std=c++20 kf_probe_q.cpp kfixture.cpp opshapes.cpp core/format/encoding.cpp
878|core/util/rad_util.cpp -I/opt/radiance-src{,/tools,/core,/abi}
879|-I/opt/radiance/include/radiance`. The recorded geometry spells the contract:
880|a_scale [M, K/128] f32 (sigmoid-filled, so strictly positive), b_scale [N, K/32]
881|e8m0, b_ref present as the full exponent plane (a [N] per-row reduction is what
882|the oGemmNtQ comment means, not the recorded shape), a_sum ABSENT in every case.
883|
884|## 2026-10-20: mul / sigmoid / rope_table / gdn_gated_rmsnorm -- elementwise holes closed, the GDN family opens
885|
886|With `gemm_nt_q` shipped, the 27B trunk's remaining holes were five recorded GDN
887|ops plus three elementwise ones. A correction first: the previous session's note
888|that `quant_act_fp8` (the producer of this GEMM's fp8 activations) "still has no
889|device row" was wrong -- the band has shipped `quant_act_fp8_act` since MiniCPM,
890|and the assessment table above never listed it as a hole. The missing producer
891|never existed; today's sweep confirms the quantizer rows replay clean.
892|
893|Four rows, all fixture-gated against the real 27B pack (995-case nodraft):
894|
895|| row | op | worst rel_l2 (tol) | note |
896||---|---|---|---|
897|| `mul_act` | mul | 0.00e+00 (6e-03) | bit-identical at n=6144 M=1..512 |
898|| `sigmoid_act` | sigmoid | 0.00e+00 (6e-03) | the double-rounded exp pays off exactly |
899|| `rope_table_f32` | rope_table | 1.9e-08 (1e-05) | first row where the sincosf-vs-double delta is *visible*: the store is f32, so the ~1 ULP difference surfaces at 1e-8 instead of hiding under a bf16 narrowing. Still 500x under the case tolerance. |
900|| `gdn_gated_rmsnorm_bf16` | gdn_gated_rmsnorm | 9.9e-06 (6e-03) | tree-vs-sequential row sum, the same rounding class `rmsnorm_bf16` has always traded; 600x margin on real GDN data |
901|
902|Regressions with all libraries: kernels.rkb **1867/1867** (was 1850 — the +17 are
903|the stock pack's 7 rope_table + 10 gdn_gated_rmsnorm cases, which already carry
904|27B shapes there and now run device-side as well as host), mxfp4 full pack
905|(drafter + vision) **4513/4513**; worst rel_l2s unchanged at 3.135e-03 /
906|2.859e-03, the 24 named skips the same 7+ GiB embed/logits budget class. After
907|this batch the full-pack `not reached: libr3d` list is 7 rows, all known classes
908|(budget-skips, cast pairs absent from that pack, M==1 decode rows) — no 27B-side
909|surprises.
910|
911|Design notes that outlive this batch:
912|
913|* **`offlin` is a liar for linear walkers.** `ref_sigmoid` walks numel(Y) through
914|  `offlin`, whose identity branch is contiguity -- a rank-2 with a pitch wider
915|  than n reads *different elements* than a linear walk would. The `sigmoid` row
916|  therefore refuses pitch > n even though `row2_pitch` (and the `add` row, which
917|  walks per-row and agrees with offlin there) accepts it. Same class of trap as
918|  the descale-pointer poisoning in the attention harnesses: matching ref means
919|  matching its *index function*, not just its arithmetic.
920|* **The GDN gate_rowoff is worth keeping as a first-class shape.** The 27B's
921|  gated norm runs against a plain [M, 128] z today, but the engine's fused
922|  [rows, heads*channels] slice walk costs one extra division in the kernel and
923|  buys coverage of the fused epilogue the recurrent fold will produce.
924|* **r3d_gemm_mxfp4.hip was compiled without the house `-ffp-contract=off`**
925|  (CMakeLists source-property line missing). Its chains happen to contain no
926|  contractible mul+add pattern, so nothing was wrong -- but the flag is a
927|  per-TU promise, and one edit away from breaking silently. Line added.
928|
929|Next in the family, in fixture-sake order: `gdn_conv_update` (rolling [.., 3]
930|conv state, one plane), `gdn_kkt_solve`, `gdn_chunk_scan` (both head_k 128, M to
931|512), then `gdn_recurrent_update` last: it carries three encoded weight planes,
932|which a device row cannot consume without a layout hook. The two unrecorded ops
933|(`gdn_conv_prep`, `gdn_conv_recurrent_update`) wait on the dt_bias dtype fix
934|upstream or on `--record-rows` once their siblings' kernels exist.
935|
936|## gdn_conv_update_bf16 -- the second GDN row, bit-exact across the whole 27B decode band (2026-10-20)
937|
938|The decode-band convolution over the rolling window (ref_ops.h spec 10): one
939|block per (sequence, channel), ref's `REF_PAR_FOR(N * conv_dim)` decomposition
940|verbatim. The fixture pack records q_len 1..512 at head_k=head_v=128,
941|conv_width=4, conv_dim 10240 (16 q/k + 48 v heads), all bf16, bias absent,
942|state rank-3 [S, conv_dim, width-2+M] -- a spec-decode band, not just plain
943|decode: `num_accepted` picks the read offset so a rejection is a read-offset
944|change, never a recompute.
945|
946|The arithmetic came out exactly as the gemm doctrine predicts: a width-tap f32
947|chain in ref's ascending-j order over exact bf16 loads (bf16xbf16 products are
948|exact in f32), silu on the double-rounded sigmoid `sigmoid_act` already proved
949|bit-identical. Every check on every sweep: **rel_l2 0.000e+00**, including the
950|rewritten INOUT `conv_state` -- four output operands per case, none diverging.
951|
952|Gates (all on the real packs, serve untouched -- the row is inert for MiniCPM):
953|
954|| sweep | result |
955||---|---|
956|| nodraft 27B pack, `--op gdn_conv_update` | 80/80, worst 0.000e+00, per-case table attributes `gdn_conv_update_bf16 \| libr3d \| device` |
957|| stock kernels.rkb, all libraries | 1877/1877 (24 named budget-skips as ever), worst rel_l2 unchanged at 3.135e-03, coverage ratchet re-locked at 30 libr3d rows |
958|| full mxfp4 pack (drafter + vision) | 4553/4553 -- up from 4513 checked, the +40 being the conv_update cases that now run device-side |
959|
960|Design calls worth keeping:
961|
962|* **ref's op-wide window guards became per-sequence skips.** `off + width - 1 >
963|  state_len` reads device data on the launch side; ref can fail the whole call,
964|  a plugin block can only abstain. A failing guard leaves that sequence's state
965|  exactly as it was -- visible in the INOUT diff if a caller ever trips it.
966|* **`offlin` is the flat identity on DENSE index tensors** -- for a rank-2
967|  [S, M] `state_idx` the ref reads element `n` flat, NOT `n*M`. The row demands
968|  dense index tensors (then agrees with ref by construction) and refuses
969|  strided ones rather than reimplementing the decomposition.
970|* **T comes from `x->shape`, not `cu`.** ref reads `cu[N]` host-side because it
971|  is host code; every valid call has cu's last entry equal to x's row count, and
972|  the tensor shape is the only one a launcher can see without a sync.
973|* The probe `tools/kf_probe_ops.cpp` (dump any op's recorded geometries without
974|  a card, one sample per param digest) is the right first move for any new row:
975|  it cost one container run and settled geometry, dtype, optionality, and
976|  rank before a line of kernel code.
977|
978|GDN holes left for the 27B trunk: `gdn_kkt_solve`, `gdn_chunk_scan`,
979|`gdn_recurrent_update` (three encoded planes, needs the layout hook), plus the
980|two unrecorded ops.
981|
982|## gdn_kkt_solve_bf16 -- the third GDN row, forward substitution bit-exact (2026-10-20)
983|
984|The chunk triangular inverse `A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1`,
985|one block per (chunk, sequence, v head). Geometry pinned by `kf_probe_ops` before any
986|code: head_k 128, chunk 64 recorded ONLY via A's shape `[M, 48, 64]` (the param digests
987|carry `M` and `head_k` alone -- ref's fallback of reading chunk from A's trailing shape
988|is the live path, not the dead one), k bf16 `[T,16,128]`, beta/g f32 `[T,48]`, cu i32,
989|M 1..512.
990|
991|The row is deliberately the SCALAR sibling of libr4d's `k128_c64` WMMA row: ref's inverse
992|is a per-column forward-substitution chain (`A[i][j] = delta - sum_{l=j}^{i-1} M[i][l]
993|A[l][j]`, f32, l ascending), and a thread-owns-column kernel walks that chain in ref's
994|exact order -- bit-exact by construction, which the WMMA blocked merge cannot claim.
995|Chunks are independent by construction, so the chunk index became the grid's free axis
996|(it is what fills the CUs where ref's N*H parallelism would idle the card at prefill);
997|thread ownership makes the substitution barrier-free inside its loops. LDS ~25 KB
998|(strict-lower triangle + per-thread columns padded to 65) -> two blocks/CU.
999|
1000|Gates (all on the real packs, serve untouched throughout):
1001|
1002|| sweep | result |
1003||---|---|
1004|| nodraft 27B pack, `--op gdn_kkt_solve` | 80/80, worst 0.000e+00; all 10 digests attributed `gdn_kkt_solve_bf16 \| libr3d \| device` |
1005|| stock kernels.rkb, all libraries | **1887/1887** (was 1877 -- the stock pack carries 10 kkt cases, one per digest M=1..512, previously host-only, now device-side AND bit-exact), worst rel_l2 unchanged at 3.135e-03, libr3d coverage 31 rows |
1006|| full mxfp4 pack (drafter + vision) | **4593/4593** (was 4553, the +40 the kkt cases), worst 2.859e-03 unchanged, the same 7 known not-reached rows, 1215 guarded-allocation cases 0 OOB |
1007|
1008|Lessons, permanent:
1009|
1010|* **T is `k->shape[rank-3]`, not `[rank-2]`.** The conv row's "T from the tensor shape"
1011|  reading reads shape[rank-2] -- correct for a rank-2 `x`, but for a rank-3 `[T, Hg, K]`
1012|  operand that slot is the HEAD axis. Wrong T mis-derived H/Hg, inflated grid z past the
1013|  real head count, and wrote 256 B past A on the first L==chunk cases -- kbench's guarded
1014|  allocations named the operand and the size on the very first card run. "shape[rank-2]
1015|  is the token axis" is a rank-2 fact, not a house rule. (591de90 caught, a8d69aa fixed.)
1016|* The stock kernels.rkb is not shape-free of the 27B: it now carries gdn cases at
1017|  head_k/head_v 128 (the conv and kkt digests), so a new GDN row shows up in the STOCK
1018|  regression count too. Expect the +N; it is the row proving itself twice.
1019|
1020|GDN holes left for the 27B trunk: `gdn_chunk_scan`, `gdn_recurrent_update` (three encoded
1021|planes, needs the layout hook), plus the two unrecorded ops.
1022|
1023|## gdn_chunk_scan_bf16 -- the fourth GDN row, the serial scan with the state in registers (2026-10-20)
1024|
1025|The chunked delta-rule scan `W = beta(V - e^g K S^T)`, `V' = A W`, `O = scale e^g Q S^T + sA V'`,
1026|`S = e^{g_last} S + V'^T K'`. Geometry pinned by `kf_probe_ops` before code: head_k=head_v=128,
1027|chunk 64 recorded (again) only via A's shape `[M,48,64]`; q/k bf16 `[T,16,128]`, v/o bf16
1028|`[T,48,128]`, g/beta f32 `[T,48]`, h0/ht f32 `[1,48,128,128]`, cu i32, state_idx i32; 10 digests
1029|M=1..512 in BOTH packs. This is the first r3d GDN row whose `scale` parameter is live: every
1030|recorded case carries 0.0883883 = 1/sqrt(head_k), read as ref does (f64 param, f32 use).
1031|
1032|Design calls worth keeping:
1033|
1034|* **The chunk recurrence is serial by nature** -- W reads the state the previous chunk left --
1035|  so unlike kkt_solve there is NO free chunk grid axis. The block is ref's own `REF_PAR_FOR`
1036|  unit: one per (sequence, v head).
1037|* **The transposed f32 state lives in registers**: one thread owns one value column e of
1038|  St[d][e], a 128-float `#pragma unroll`ed array (launch_bounds(128,1) opens the full
1039|  256-VGPR budget). Whole-column ownership is also what keeps bit-exactness cheap: every
1040|  ascending-d chain stays a scalar walk of ref's order; splitting columns across lanes would
1041|  hand chains between lanes just to stay serial.
1042|* `V' = A W` runs **descending i in place** over the W buffer: row i consumes only j <= i, so
1043|  each write lands after its slot's last read. One LDS buffer, not two.
1044|* The sA row and staged g/beta are LDS broadcasts with **barriers outside every predicate**
1045|  (threads owning no column still march through them), and the sA loop carries TWO barriers
1046|  per row: row-ready-before-reads and reads-done-before-the-next-row-overwrites.
1047|* **A negative `state_idx` slot skips the whole (n, h) row -- the o writes included**, exactly
1048|  ref's `continue`; the row pitch may be wider than one slot (`stride[0]`, column 0 is the
1049|  slot); h0/ht strides are READ off the tensor at rank >= 4 (ref's page-padding comment).
1050|* Launcher derived T from `q->shape[rank-3]` from day one -- the kkt lesson applied before
1051|  the card could re-teach it, and this row's FIRST card run was green, launcher included.
1052|
1053|Gates (all on the real packs, serve and co-tenant serve untouched throughout):
1054|
1055|| sweep | result |
1056||---|---|
1057|| nodraft 27B pack, `--op gdn_chunk_scan` | **80/80**, worst 0.000e+00; all 10 digests attributed `gdn_chunk_scan_bf16 \| libr3d \| device`; 0 OOB |
1058|| stock kernels.rkb, all libraries | **1897/1897** (was 1887 -- the stock pack carries its 10 chunk_scan digests too, now device-side AND bit-exact), worst rel_l2 unchanged at 3.135e-03, libr3d 41 rows |
1059|| full mxfp4 pack (drafter + vision) | **4633/4633** (was 4593, +40 from the chunk_scan rows), worst 2.859e-03 unchanged, 1255 guarded-allocation cases 0 OOB |
1060|
1061|The graph ratchet for the nodraft 27B skeleton moves 29 -> **30 of 31**: the last device hole
1062|is `gdn_recurrent_update` (the decode path, with the three encoded weight planes -- the
1063|layout-hook row).
1064|
1065|Lessons, permanent:
1066|
1067|* **`tools/gate.sh` now IS the gate** (`op <name>`, `stock`, `full` on rdna3-dev): the exact kbench
1068|  command lines, the two staged homes (attribution = oracle+libr3d only; regression adds
1069|  libavx), persistent ref caches under /scratch. No more reconstructing gate lines from
1070|  report headers.
1071|
1072|GDN holes left for the 27B trunk: `gdn_recurrent_update` (the last graph hole), plus the two
1073|unrecorded ops (`gdn_conv_recurrent_update` and the conv-prep sibling) and the two fixtures
1074|declined by the ssm-dt dtype quirk.
1075|
1076|## 2026-10-20: gdn_recurrent_update_bf16 -- the last device-graph hole closes, 31/31
1077|
1078|The decode recurrence against the paged state cache: per-token delta rule
1079|`S = e^g S + beta (v - S k) k^T`, the qk l2 norms, the q readout, and the gated RMS norm
1080|folded into the epilogue (ref_gdn.cpp; 48 of 64 layers run it every decode step). Row
1081|`gdn_recurrent_update_bf16`, kernel in `src/r3d_gdn.hip` after the chunk_scan section,
1082|schema `pRecurrent/oRecurrent` verbatim libref (kSchemas-declared so the row stands alone),
1083|constraints `head_k==128, 1<=head_v<=128, state_form in {per_candidate}`.
1084|
1085|**The layout hook was a phantom.** The handover carried `gdn_recurrent_update` as
1086|"needs the weight-layout hook for three encoded planes" -- the probe settled it in one
1087|run: `A_log`, `dt_bias`, `norm_w` are `scheme=plain`, ONE plane each, stored bf16 in this
1088|container. `weight_refusal` declines a hookless device row only above one plane, so the
1089|row reads them AS STORED through `ldt2` -- dtype-driven exactly like the oracle's `ldt`,
1090|which is what keeps bf16-stored weights bit-equal on both sides. The three-week-old
1091|"blocked by the layout hook" note was inherited, probed once, and dissolved.
1092|
1093|**One block per (sequence, v head) -- thread owns one value column.** The same
1094|transposed-state register layout as chunk_scan (thread e owns `St[d][e]`, 128 f32
1095|registers), and the delta rule turned out to be nearly barrier-free BY THE MATH: the
1096|update's two d loops read only `kn[d]` into thread e's own column, so decay, the S k
1097|product, the rank-1 add and the q readout are all register-local per thread. Only two
1098|reductions cross threads, and BOTH walk ref's serial order on purpose:
1099|
1100|* the qk l2 sums -- thread 0 over the staged q/k rows (not two per-thread chains);
1101|* the rms sum `sq2 += orow[e]^2` -- a 128-long ASCENDING-e chain, so `block_sum` (the
1102|  tree) would reassociate it. The squares stage to LDS and thread 0 walks the chain.
1103|  128 cheap LDS reads per token is the price of the contract; correctness trades no chain.
1104|
1105|Barriers: six per token, matched on every path (the `!own` tail marches through two
1106|beside the rms pair). Early returns (`slen<=0`, negative input slot) are block-uniform.
1107|
1108|**The two bugs the first card run found, both inherited-shape traps:**
1109|
1110|* `r3d_softplus` shipped as `log1p(-x)` / `log1p(x)` -- the exp fell out of the port, and
1111|  `log1p(-1) = -inf` sent the gate exponent to +inf and the state to 1e14. Ref's
1112|  act_softplus takes `log1p(exp(+/-x))` in both branches; the fix restores it verbatim.
1113|* `norm_w` was offset TWICE -- `nw + e*w_s1b` AND index `e`. The row landed on element
1114|  2e of the bf16 plane, so every channel got the wrong stored scale while the state
1115|  stayed bit-exact (the weight feeds only the epilogue) -- a failure signature that
1116|  localised the bug to the epilogue in one read of the report. One offset only.
1117|
1118|`state_form=per_candidate` is constraint-gated, and the rule that "a constraint on a key
1119|the query lacks does not hold" makes it exact: the arch block always declares state_form
1120|("anchor when speculating, else per_candidate"), so serve decode and the nodraft pack hit
1121|this row, while anchor batches (the drafter, and every gdn_recurrent case in the FULL
1122|pack) stay with libr4d/libavx untouched -- confirmed by the full-pack report: zero
1123|anchor cases attributed to libr3d. `o_q/o_scale` (the folded out-projection quantiser,
1124|absent from every recorded 27B case) is REFUSED, never silently ignored.
1125|
1126|Gates (all via `tools/gate.sh`, serve and co-tenant serve untouched throughout):
1127|
1128|| sweep | result |
1129||---|---|
1130|| nodraft 27B pack, `--op gdn_recurrent_update` | **64/64**, worst rel_l2 **0.000e+00** (bit-exact, all 8 q_len digests 1..128 attributed `gdn_recurrent_update_bf16 \| libr3d \| device`), 0 OOB |
1131|| stock kernels.rkb, all libraries | **1897/1897** unchanged, worst rel_l2 unchanged at 3.135e-03 (the stock pack's recurrent cases are anchor-form -- correctly not taken) |
1132|| full mxfp4 pack (drafter + vision) | **4633/4633** unchanged, worst 2.859e-03 unchanged, 0 OOB (its recurrent cases are all anchor-form, all libavx/libref as before) |
1133|
1134|**The ratchet: 30 -> 31 of 31.** After staging the new `.so` into the serve home, the
1135|engine restarted clean (the startup refusal it exercised since day one is gone from this
1136|op): `kernel selection libavx (host) 564, libr3d 564 op bands` -- every device band
1137|covered, decode ~60 tok/s on a short request, health 200.
1138|
1139|Lessons, permanent:
1140|
1141|* **A serial reference reduction is the contract, not an implementation detail.** When
1142|  ref accumulates `sq2 += orow[e]^2` ascending, the tree sum is a different number.
1143|  Stage to LDS, walk it with one thread; the cost is bounded and the bit is exact.
1144|* **Early returns in barrier-carrying kernels must be block-uniform** -- guard on
1145|  blockIdx-derived values only (`slen`, `slot_in` qualify; `orow` never does).
1146|* **The probe answers the hook question for free.** Encoded != multi-plane: plain-scheme
1147|  single-plane weights are weight_refusal-invisible and dtype-driven reads on both sides.
1148|
1149|GDN holes for the 27B trunk: NONE -- the graph is whole. What remains in the family is
1150|the two unrecorded ops (`gdn_conv_recurrent_update` and the conv-prep sibling, blocked by
1151|the upstream ssm-dt dtype quirk, harness-side) and the anchor-form row the drafter needs
1152|(already libr4d's; the constraint keeps both libraries honest).
1153|
1154|## logits_gemm_e4m3 — the head GEMV on joined planes; 27B placement chain mapped (2026-10-20)
1155|
1156|The 27B container stores `output.weight` as fp8_e4m3 codes + a bf16 scale grid
1157|(`rad.planes`, block 128x128). No libr3d row could claim it: hookless device rows
1158|are declined for multi-plane weights, and the host kernels get refused placement
1159|for mapped-eligible tiers ("stored form is not the container's"). Startup died at
1160|`logits_gemm -> [host:libavx]` on the 27B config.
1161|
1162|* New TU `src/r3d_gemm_logits.hip` (`logits_gemm_e4m3`): reads the planes header
1163|  verbatim from the ABI (magic PLNS), decodes codes+scale in-register. WMMA
1164|  rejected on math grounds — decoded e4m3 x bf16 needs >11 mantissa bits, an
1165|  f16-rounded product sits ~2e-3 off, and the head is the tolerance-critical op.
1166|  Bandwidth-bound GEMV instead: warp per (row-block, m-group), xs staged per m,
1167|  uint4 codes walk + 8-byte scale walk (br=128) or scalar walk (br=1), f32
1168|  accumulation in the reference's ascending k order. MT in {1,2,4,8} from M.
1169|* Two bugs, both caught ONLY by the standalone unit test
1170|  (`tools/test_logits_e4m3.cpp`, card + CPU oracle, 8 cases: 8/8 after fix):
1171|  1. The block-staged "uniform" scale was thread 0's *warp's* row — every other
1172|     warp silently used the wrong head row's scale (br=1 exposed it; br=128 with
1173|     rblk = r/128 made the whole block genuinely share a scale row, masking it).
1174|  2. Row ordering: declaring this row *before* `logits_gemm_bf16` steals the
1175|     plain bf16 band, because a hook answer of "unsupported" means *store as is*,
1176|     not "decline". It would then refuse those at launch and serve would fall to
1177|     host logits every token. bf16 row first; the encoded band declines the
1178|     hookless bf16 row at declare time anyway.
1179|* Attribution: the 27B serve startup line no longer lists
1180|  `logits_gemm -> [host:libavx]` — the row claims the band. The stock/full pack
1181|  sweeps budget-skip all logits_gemm cases (7.1-14.2 GiB against the co-tenant's
1182|  RAM), so the unit test is the correctness evidence, plus serve adoption.
1183|
1184|## 27B serve placement chain — ONE blocker, and it is the conv pair (2026-10-20, corrected same day)
1185|
1186|An earlier draft of this section blamed an "fp8 GEMM wall". `--debug-selection`
1187|refutes that: the trunk GEMM bands are claimed (995 op bands on libr3d; only
1188|`attn_paged/rope/kv_store/*_table` even print a device+host pair line). The real
1189|chain, from `sel27.log` (kept in `/scratch/r3d/`):
1190|
1191|1. `gdn_conv_prep` + `gdn_conv_recurrent_update` resolve host-only (the two
1192|   unrecorded ops). Their weights (`ssm_inz.weight/scale/ref`, `ssm_out.*` —
1193|   three-plane encodings, plus `ssm_ab`) are then "weights only a host kernel
1194|   can read".
1195|2. Planner rule (engine_bringup, WARN x720): a block whose group contains such a
1196|   weight keeps the WHOLE group in the container's mapping (mapped tier).
1197|3. The group includes `attn_norm.weight`/`ffn_norm.weight`, which are declared
1198|   RAD_F32 and stored bf16 -> `widen=true` at load -> `weight_file_direct` false
1199|   -> the mapped-tier guard in engine_bringup refuses: "stored form is not the
1200|   container's (widened at load)". Startup stops there.
1201|
1202|Why minicpm never hit it: pure-llama, norms declared bf16 (identity, no widen),
1203|so group-pinning to the mapping is legal there. Engine-side fixes exist (split a
1204|widened weight out of a pinned group — it's 512 B/layer; or declare the qwen35
1205|norm operand bf16), but the libr3d-side unlock is in scope and was always the
1206|backlog: **device rows for `gdn_conv_prep` + `gdn_conv_recurrent_update` with a
1207|layout hook for the three-plane ssm weights**. No host-only member -> no
1208|group-pinning -> norms get staged -> startup proceeds. The kbench recording of
1209|those ops is separately blocked by the ssm-dt bf16/f32 dtype quirk — gate via
1210|`--record-rows` or a recorded-rows fix, not on the serve path.
1211|
1212|`--debug-selection` also pinned the VRAM split for the eventual serve: weights
1213|6.67 GiB static, KV 14.44 GiB, mapped 9.67 GiB, host pool 0 B. The card has room.
1214|
1215|Incident: the first serve27 attempt OOM-killed `libr3d-serve` (MiniCPM demo,
1216|exit 137) — 27B host-side mapping on a box holding co-tenant serve + one serve has
1217|~2.6 GB headroom. Do not run serve27 while minicpm serve is up; minicpm serve
1218|was restored after staging (health 200, 564/564 device bands).
1219|
1220|## Whole-GDN serve A/B on minicpm: NULL experiment, closed (2026-10-20)
1221|
1222|`R3D_NO_GDN=1` vs plain, paired same-session decode@8k on `libr3d-serve`:
1223|40.2-41.3 vs 43.9-48.8 tok/s. The gap is noise — MiniCPM5-2B is pure `llama`
1224|arch, **zero GDN ops** (rad-info: 42 layers plain attention, 2 kv heads). Both
1225|arms executed byte-identical paths. The kill switch itself is verified working
1226|where GDN exists (`rad-info --plugins`: 43 kernels / 37 schemas plain, 38 / 33
1227|with the env). A meaningful whole-GDN serve A/B needs the 27B serve, which is
1228|blocked behind the fp8_e4m3 GEMM wall above. The earlier claim that "minicpm
1229|serve exercises GDN" was false — the evidence was `grep -c gdn` on a binary
1230|container (589 hits = quantization noise). `grep -a` on binaries proves nothing;
1231|ask `rad-info`.
1232|
1233|## gdn_conv_prep_bf16 + gdn_conv_recurrent_update_bf16: SHIPPED, byte-exact (2026-10-21)
1234|
1235|The pair that pinned the 27B placement chain (`0d9862f`, cold-fix `f2b1e13`).
1236|Device rows remove the host-only group members, which is what force-mapped the
1237|groups holding widened norms. `conv_prep` is the prefill preamble in three
1238|launches (conv+state-commit, serial-order per-head l2 norms, chunk-reset gate
1239|cumsum) so libref's phase order holds: the norms read the ROUNDED conv output,
1240|the cumsum resets per 64-chunk. The fused row runs conv + recurrent in one
1241|launch per (sequence, v head), staging through LDS rounded by x's dtype to
1242|reproduce the pair's staging round trip; the conv window commits BEFORE the
1243|state_idx skip, as the ref pair defines it.
1244|
1245|Evidence (`mxfp4_nodraft.rkb` and stock/full carry ZERO recorded cases for
1246|these ops — ssm-dt quirk), `tools/test_conv_pair.cpp`: dlopen both plugins,
1247|launch each row through its own `rad_kernel_at` entry (the engine's own table),
1248|seed -> device -> reseed -> ref -> memcmp, all four geometries byte-exact:
1249|prep A (cold+warm slots, chunk-crossing cumsum, gates fused in x, slot order
1250|out of batch order), prep B (V==K==128, separate gate projections, two chunk
1251|resets), fused A (GQA grp 3, V<K idle-tail threads, n_acc rollback, negative
1252|output slot), fused B (V==K==128, z absent, window longer than commit span).
1253|
1254|Gates after the pair: stock 1917/1917 (worst 3.135e-03 unchanged), full
1255|4633/4633 (see below), 0 OOB.
1256|
1257|Two lessons paid for:
1258|- **ref gates the history READ on has_init, not just the commit.** My conv
1259|  kernel read the slot whenever it existed: a cold prefill on a recycled cache
1260|  slot convolved over garbage. prep A caught it (exactly the cold tokens 70..72
1261|  diverged); prep B stayed byte-exact throughout, which is why both belong.
1262|- **a sequence's INPUT state slot must be disjoint from every slot another
1263|  sequence WRITES this step.** The test's fused A first shared slot 0 between
1264|  seq0's commit and seq1's read; BOTH arms raced (diff sets shifted run to
1265|  run). That is a legal-violation detector worth keeping in mind for real
1266|  batch construction, not just tests.
1267|
1268|## gemm_nt_q + gemm_nt_q_gated at fp8a8: FP8 containers place fully on device (2026-10-21)
1269|
1270|`qwen3.5-9b-fp8.rad` (rtn e4m3 128x128, bf16 scales) resolved both rows host on
1271|gfx1100, so the planner put every projection's arithmetic on the host and the
1272|mapped-tier group pin refused startup outright ("'blk.0.attn_norm.weight' was
1273|placed on the mapped tier"). With libr3d mounted, `--debug-selection` at spec 0
1274|+ mm 0 named EXACTLY these two ops as host-only and zero missing device bands —
1275|the scope was the pair, no more.
1276|
1277|`r3d_gemm_fp8a8.hip`: the m64 WMMA skeleton with an exact e4m3→bf16 LUT on both
1278|planes (products of decoded e4m3 are exact in bf16, so only the accumulation
1279|order differs from ref and the kbench tolerance covers it). The scale law is
1280|ref_gemm.h quant_dot verbatim: per 128-of-K group, sub = ascending f32 of exact
1281|products, acc += (sub * sa) * sb — two flush multiplies in that order, sa from
1282|the f32 a_scale plane, sb from the bf16 b_scale grid (128-row block grid OR
1283|per-row grid, make_scales' rblk rule). The weight scale CANNOT fold into the
1284|bf16 fragment the way mxfp4's e8m0 exponent did: rtn scales are arbitrary
1285|floats, not powers of two.
1286|
1287|`gemm_nt_q_gated` composes rather than fuses: the tile writes its bf16 [M,N]
1288|gate_up plane into the launch scratch (hook: 64*N*2, band M <= 64 — recorded
1289|drafter bands are exactly M ∈ {1..64}), the ported gated_quant_fp8 epilogue
1290|closes it into the code pair. Byte-identical to the unfused pair by
1291|construction (the epilogue is that kernel), which is the whole argument for
1292|composing first and fusing later.
1293|
1294|Evidence: fixture `fp8_9b.rkb` (1461 cases, recorded from the container's
1295|skeleton — the skeleton keeps fp8 planes whole, so weights are the real ones);
1296|attribution home (libr3d+libref only): gemm_nt_q 536/536 worst rel_l2 3.7e-04
1297|vs 5e-3 tol; gemm_nt_q_gated 56/56 worst 8.6e-06 vs 3e-2; 0 OOB; stock
1298|1995/1995 and full 4713/4713 unchanged. Serve acceptance: `--placement
1299|all_vram --num-speculative-tokens 0 --mm-max-patches 0`, "9.29 GiB placed from
1300|9.29 GiB of container read", no host-only line, health 200, "capital of
1301|Australia" -> Canberra. Decode 9.8 tok/s = ~98 GB/s effective: the m64 arm
1302|covers ~27% of the card's bandwidth — the decode-shaped tile is the open perf
1303|lever (llama.cpp Q4_K_XL same card: 75.8 t/s no-draft).
1304|
1305|Two schema/pitch traps paid for:
1306|- **operand ROLES differ per op**: gemm_nt_q's a_scale is optional,
1307|  gemm_nt_q_gated's is REQUIRED. Copying the plain table into the gated schema
1308|  made the loader refuse libref entirely at load ("two different schemas ...
1309|  refusing libref") — every libref row vanished with it, not just the one op.
1310|- **band kernels index through BYTE offsets**: the epilogue's gu/ob pitches are
1311|  bytes (the kernel offsets a byte pointer, then indexes elements), while the
1312|  f32 scale pitch is element units. Passing element pitches strides every row
1313|  by half a row — silent, plausible-looking garbage.
1314|
1315|## gemm_nt_q_fp8a8_decode — the decode-band width row, 2.2x fp8 serve decode (2026-10-21)
1316|
1317|Issue #4 closed. The m64 tile at M = 1..8 stages every B row through a 64-row
1318|slab it uses M rows of: 98 GB/s effective on a 368 GB/s card. The new row reads
1319|every weight byte exactly once: warp per weight row, four codes per lane (one
1320|aligned dword — the logits GEMV's load pattern), the MT activation rows decoded
1321|through a loop-free e4m3→f32 into LDS per 128-of-K group, sub ascending per lane
1322|and butterfly-folded across the wave, flush `acc += (sub * sa) * sb` at exactly
1323|the oracle's group granularity and multiply order. Priority 1100 with `M <= 8`
1324|takes the decode band from the tile (the split-KV attention precedent); the
1325|gated row's scratch tile dispatches to the same GEMV at M <= 8, because the FFN
1326|gate_up plane is two-thirds of decode weight traffic and the band would not
1327|reach its 2x bar through q/o/down alone. Ragged N clamps the dead warps' load
1328|row rather than returning early: every warp must reach the block's barriers, and
1329|a returned warp would also leave LDS staging holes. B alignment (base and row
1330|pitch mod 4) is refused at the entry so odd strides resolve to the tile, which
1331|walks bytes.
1332|
1333|Evidence: attribution home, fp8_9b.rkb --op gemm_nt_q: 536/536, the 24 M <= 8
1334|cases attributed to `gemm_nt_q_fp8a8_decode` and all BIT-EXACT (0.00e+00 vs the
1335|5e-3 tol; the tile's worst over the remaining 36 is 3.7e-04, unchanged). Full
1336|fp8 pack 2582/2582, gated worst 8.62e-06 — identical to issue #1, so the GEMV
1337|reassociation left the composed quantiser pair exactly where it was. Stock
1338|2007/2007, full mxfp4 4733/4733, 0 OOB. Serve A/B (qwen3.5-9b-fp8.rad,
1339|--num-speculative-tokens 0, ctx-512 pair-difference probe, arms one env apart):
1340|tile 8.61 tok/s vs decode 18.90 tok/s — **2.19x**, ~82 -> ~180 GB/s effective.
1341|With the container's default MTP drafter on: 26.1 -> 35.2 tok/s.
1342|
1343|Lessons: (1) **an issue-time refusal is an engine abort, not a re-resolve** —
1344|"kernel ... refused op at issue ... selected wrongly". A kill switch that only
1345|checks getenv in the entry cannot A/B anything: the resolver picked the row at
1346|declare from its constraints, the launch refused, the step died. R3D_NO_FP8_DEC
1347|found this on its first serve A/B; R3D_NO_FP8A8 had been carrying the same
1348|latent fault since issue #1 (never exercised under serve). Both levers now
1349|withdraw their rows at the rad_kernel_at registration gate, the R3D_NO_GDN
1350|pattern. (2) This model container enables the MTP drafter by default — an A/B
1351|without --num-speculative-tokens 0 measured 1.35x, because verify/accept
1352|overhead dilutes GEMM bandwidth wins; match the evidence arm's spec setting,
1353|not just the flags you remember passing.
1354|
1355|## v1.11 retarget + the DFlash2 drafter family: issue #9 shipped (2026-10-24)
1356|
1357|`v1.11` fast-forwarded into main (`9d04d8a`..`cd92c7e`); the docs refresh the
1358|branch intentionally deferred ships here: the pin moves radiance
1359|v1.8.1/ABI 22 -> **v1.11.0/ABI 26** (README, `docker/Dockerfile.dev` `RAD_REF`,
1360|`tools/gate.sh` image default). Row count after the chain: **55**.
1361|
1362|Why (issue #9): the published 27B-mxfp4 container bundles a DFlash2 drafter and
1363|the engine defaults speculative tokens to auto, so its default startup crashed
1364|placement on gfx11 — `dflash_conv` / `dflash_select` / `logit_rerank` were
1365|host-only and their weight units force-mapped, which `rad.planes` tensors
1366|cannot survive. v1.11 additionally declares `sample_ngram_ban` a DEVICE band:
1367|weightless ops cannot host-place at all.
1368|
1369|Rows shipped (six), all behind `R3D_NO_DFLASH` (family, including the GEMV) or
1370|`R3D_NO_NGRAM_BAN`:
1371|
1372|| row | op | band |
1373||---|---|---|
1374|| `dflash_conv_t2_g16_bf16` | dflash_conv | drafter conv pass (rank-3 base view arrives unsqueezed — the entry takes both) |
1375|| `dflash_select_bf16` | dflash_select | drafter token select |
1376|| `logit_rerank_bf16` / `logit_rerank_e4m3` | logit_rerank | rerank on the bf16 head and the joined-e4m3 head |
1377|| `sample_ngram_ban_f32` | sample_ngram_ban | engine sampler band: integer compares + idempotent -inf store, bit-identical to libref by construction |
1378|| `gemm_nt_q_w2a8_decode` | gemm_nt_q | w2a8 draft-head GEMV, `M <= 16`, `group == 128` |
1379|
1380|Gates on the retargeted build (radiance-dev:v1.11.0, gfx1100-pinned): stock
1381|kernels.rkb **2011/2011** (worst 3.135e-03), full mxfp4 pack **4737/4737**
1382|(2.859e-03), `--op gemm_nt_q` on the 27b_dflash2 pack **702/702** (9.299e-05),
1383|27b_df3 **768/768** (1.961e-04). Baselines intact. Serve with the DFlash2
1384|drafter ON: zero refusals, positive w2a8 launch proof, `finish=stop` coherent,
1385|**9.5 tok/s vs 3.2** with the GEMV row dead — the draft pass genuinely runs on
1386|the card.
1387|
1388|Lesson (standing rule): the first w2a8 GEMV passed a rel_l2 gate while reading
1389|entirely wrong memory — the fixture's weights were all-zero and zero x garbage
1390|is zero — and one revision before that it had never even launched (stub
1391|signature drift -> `hipErrorInvalidDeviceFunction` -> a silent
1392|`RAD_E_UNSUPPORTED` outside the engine). `tools/repro_w2a8.hip` is the new
1393|model: every new GEMV row gets a **random-operand** numeric gate against
1394|libref, plus an independent double oracle; an all-zero fixture proves wiring,
1395|nothing more. Two more load-bearing fixes rode along: the MT tile rounds UP
1396|(an M=7 pass must not bucket to MT=4 and silently drop rows 5-7), and the w2a8
1397|band is `M <= 16`, not 8 — the drafter's block pass collapses M at 14 (verify
1398|rows join draft rows) and an unfilled bucket sends the WHOLE op host-side,
1399|which its joined scale plane cannot survive. That band gap was the startup
1400|death; GDN's anchor floor moved with it (anchor state is a slot plane: numel
1401|floor = 1 plane, not T).
1402|
1403|
1404|
1405|## dot4 MXFP4 decode GEMV: the q8x arm on `phase-b-dot4` (2026-10-24)
1406|
1407|Hipfire Phase A/B lineage: the Phase A probe proved `v_dot4_i32_iu8` (via the
1408|6-arg `sudot4(1, act_i8, 0, codes, acc, 0)` builtin, the shipped w2a8 pattern)
1409|runs the MXFP4 decode walk as byte dots; the Phase B probe (ARM C: interleaved
1410|byte-plane, codes x2 with woven e8m0 group headers, pure signed sudot4 loop)
1411|measured **2.41-2.57x** over the shipped f32-flush walk at M=1 and
1412|**3.55-5.7x** in the M=4 verify band at model shapes (12288x4096,
1413|24576x4096, 4096x12288, 4096x4096), bit-exact vs float64 requant models. The
1414|binding term was *distinct DRAM streams per warp* (the scale-plane traversal),
1415|not bytes - only at roofline did bytes rank first. `tools/bench_dot4_seed.hip`
1416|is the probe seed.
1417|
1418|Shipped (branch `phase-b-dot4`, opt-in `R3D_MXFP4_Q8X=1`, default OFF): the
1419|decode row `gemm_nt_q_mxfp4_decode` gains a q8x leg. Three new kernels ride the
1420|w2a8 q8x precedent exactly:
1421|
1422|- `m4_repack_q8x_kernel` builds the ARM C interleaved plane (1 i8 per e2m1
1423|  code = value x2, e8m0 scales woven as group headers, 2.06x nibble bytes) -
1424|  plugin-owned, lazily built on the plugin's own non-blocking stream, event
1425|  joined before the engine-stream launch, never freed; alloc/launch failure or
1426|  a VRAM-headroom miss (free-VRAM query, 512 MiB floor) sets a per-tensor
1427|  permanent fallback and the f32-flush walk serves. 27B-class packs land on
1428|  the fallback cleanly.
1429|- `m4_act_i8_kernel` requantizes e4m3 x AS f32 to i8 + one f32 requant scale
1430|  per (m, 32-group) - per-dispatch (activations change per token), chained on
1431|  the engine stream exactly like the issue-#6 A16 f16 prologue precedent.
1432|- `gemm_nt_mxfp4_gemv_q8x_kernel` - the pure signed sudot4 dot loop, flush
1433|  `(dot * AS * (1/act_max)) * 2^(e8m0-127)` preserving the shipped two-multiply
1434|  quant_dot order.
1435|
1436|Band law held: no N/M keys in table constraints (entry re-checks M <= 8 via
1437|numel(Y)/N, K % 256 == 0 - the interleaved group-header stride doubles the
1438|table's K % 128 - at plan time); any rejection serves the old legs IN-PLACE.
1439|The existing `R3D_NO_MXFP4` / `R3D_NO_FP4_DEC` predicates already withdraw the
1440|whole row at registration; no new registration was added, so no new resolver
1441|law applied.
1442|
1443|The probe's "e2m1 code 7 is NaN" was a **table-vs-copy artifact**: the
1444|generator excluded code 7 and the model shared the same table, so it could
1445|never see it. `tools/dot4_fuzz_mxfp4.hip` (new) fuzzes the INTEGRATED constants
1446|against the independent ABI decoder (`rad_fp4e2m1_to_f32`) over random draws
1447|across all twelve valid codes - code 7 is +6.0, code 15 is -6.0 - plus the
1448|signed-sudot4 and device e8m0 rules. GREEN on cat.
1449|
1450|Gates on build_v114 (radiance-dev:v1.11.0, gfx1100-pinned): op `gemm_nt_q`
1451|arm-off 480/480 (7.581e-05), arm-ON 480/480 (worst 6.994e-03 - the
1452|i8-requant drift, inside the 3e-2 mxfp4a8 tol and the 1e-2 doctrine stop
1453|line); stock 2011/2011 (3.135e-03); full 4737/4737 off (2.859e-03) and
1454|**4737/4737 with the arm ON** (6.994e-03). Baselines intact.
1455|
1456|Serve A/B, one binary, env-flipped arms, alternating, spec-3 9b-mxfp4 at
1457|4096 ctx: boot proof identical pair `kernel selection libavx (host) 736,
1458|libr3d 736`, `0 host op(s)` on the recorded draft passes both arms, no
1459|`issued differently` - the prologue chains cleanly through the MTP recorded
1460|pass. Decode medians **32.8 -> 40.5 tok/s (+23.5%)**, every pair positive
1461|(+20.4%..+23.6%); ctx-512 500-token decode medians 27.9 -> 33.4 (+19.7%
1462|clean pairs). The arm's draft record count grows by the prologue dispatches
1463|and replays identically. KLD arm-ON vs arm-OFF: KLD gate (`kldgate.py` base-collect arm-OFF, teacher-forced score arm-ON,
1464|1258 forced positions over the 11-prompt corpus): **FAIL by design of the
1465|class** - aggregate mean KLD 0.00827 (gate 2e-3), p99 0.14714 (gate 5e-3),
1466|top-1 0.975 (gate 0.98), per-corpus means 0.0020-0.0187, worst max 1.806
1467|(prose-long). Same-arm noise floor on this harness is mean ~2e-4: the i8
1468|requant genuinely shifts every decode-row output at 1e-3..7e-3 rel (the
1469|6.994e-03 worst rel_l2 the pack gate tolerated), and ~40x floor KLD is that drift
1470|compounding, not wiring (0 plane fallbacks, 0 launch failures, one-shot
1471|launch proof; +23.5% serve rate is the arm alive). The strict KLD class
1472|belongs to bit-exact arms; defaulting the q8x arm ON is an output-quality
1473|ACCEPT decision, not a gate to weaken - it stays opt-in. (The committed
1474|corpus is 11 prompts x 128 positions; a 20+ prompt corpus would change
1475|the graded fixture, so it was not improvised.)
1476|
1477|Lessons: (1) the probe-era NaN reading of e2m1 7/15 was self-confirming by

## The attention band that never was: splits retune, the live entry band, and the head_dim-256 finding (2026-10-24)

Phase C (hipfire split-K verify attention, PORT verdict) noted a free win: the
decode row's splits law looked mis-tuned for the verify band. The retune landed
first: device-asserted paired sweeps (M 1..8 x ctx 512..8192, 32Q/8KV and
32Q/4KV, L2-cold, 1134 arms) put the fold arm's optimum at 128 blocks - s16
beat the law's 24 by 1.32-1.44x at ctx>=2048 (8KV), s32 beat 48 by 1.40-1.43x
at 8192 (4KV), and the tiles_kv/2 cap pinned ctx-512 1.16x below its best.
Fold arm: wave 128, cap tiles_kv; plain arm untouched (e50b906).

The serve A/B meant to bank that win instead found that the row **had never
been resolved by the engine at all**. Three independent proofs: boot logs
select `attn_paged -> attn_paged_bf16` with no decode-row line; R3D_DEBUG_DECODE
printed zero lines across 10k+ live decode steps on v113 and v115 alike; and
R3D_DUMP captures of the selected entry show the op schema's `q_len` is 2048 -
the chunk capacity, never a step's rows - so the registered bands' `q_len <= 8`
can never match at boot. This is issue #10's schema-vs-geometry trap one level
up: not inside the row's arithmetic, but in the engine's selection of the row.
The gate reports' "Rows nothing ran" list had named `attn_paged_decode_bf16kv_*`
all along - reread: a selection failure, not fixture trivia. Every past
"device-verified" milestone of the split-K family was direct-dlopen bench
evidence, which bypasses selection entirely.

The fix moves the band into the entry the engine DOES select (bce0467):
decode/verify shapes (window 0, rows <= 8*n_seq - the per-launch bound that
survives cu_seqlens, where q_len_uniform is 0 - head_dim {64,128}, bf16 q,
bf16/e4m3 kv) hand the op to the split-K entry, whose own fallbacks land on
exactly the tile the entry would have launched, so only the launch choice can
change, never the math. The decode scratch hook rides the selected row so the
arena sizes for it. Gates: stock 2011/2011, full 4737/4737, worst rel_l2
3.135e-03 / 2.859e-03 unchanged; serve completions byte-identical to build_v113.

The captures' second fact reorders the program: the production family's
attention is **head_dim 256** (8 full-attn layers on the 9B, gqa 4), and every
WMMA body in this library - tile and split-K alike - instantiates only {64,128}.
Production decode attention has been running the scalar v1 reference port on
the card since day one. Phase C's r3d arms were the hd128 WMMA tile; the real
opponent for these models is slower still, and hipfire's kernel was hd256-
native all along. Issue #11: WMMA hd256 attention bodies; the entry band, the
registered bands' widening, and this retune all activate the moment hd256
instantiations exist. Until then the two commits are green-gated and dormant on
this host - stated plainly, not shipped as a win.

Standing rules from this window: (1) "device-verified" must name its call path -
direct-dlopen row proof says nothing about engine resolution; serve-path proof
means attribution lines in the engine's own log. (2) The "Rows nothing ran"
report section is a selection claim; read it at every gate. (3) Serve shape
vocabulary comes from R3D_DUMP captures, never from bench assumptions.

## 2026-10-24 late window: two bit-identical micro-fixes; hd256 sizing (issue #11 milestone 1)

The external kernel review's remaining card-free items shipped: `rope` now
computes `sincosf` once per pair through the token block's LDS (the angle is a
function of the pair index alone - even under imrope's component rule - yet it
ran once per (head, pair) item), and `cast` gained a Contig template arm that
folds away its per-element 64-bit division when both pitches equal n. Both are
bit-identical by construction: same float ops on the same inputs; gates confirm
at the numbers - stock 2011/2011 (3.135e-03), full 4737/4737 (2.859e-03),
27b_dflash2 702/702, 27b_df3 768/768, all unchanged. The stock report's per-case
table proves reachability: 37 rope_bf16 cases (including nontrivial imrope
sections 11-11-10) and the cast_bf16 family all ran on libr3d device rows at
0.00e+00.

The paired serve A/B (one session, container swap, 5 salted TTFT rounds + 3
decode rounds each) measured **nothing**: TTFT medians 22 720 vs 22 717 ms on a
~3k-token prompt, decode 52.09 vs 51.73 tok/s. That is the honest result and it
is informative: at head_dim 256 the prefill is dominated by the scalar
reference attention's quadratic term - rope is not yet a term worth cutting. It
will be when #11 lands; the fix is shipped now so the term is already cheap
when the ceiling moves. (22.7 s TTFT for 3k tokens is itself the loudest number
of the window - the #11 price tag in wall-clock.)

Issue #11 milestone 1 (sizing, from the `Carve<D>` formulas): a literal hd256
instantiation is 66 640 B - over the 64 KiB dynamic-LDS ceiling by 1104 B,
because `total` reserves the DBUF pair unconditionally and the +8 pitch pads on
Q/KV are load-bearing for the uint4 LDS staging. S-pitch kBN+4 + P-pitch kBN+2
+ a conditional DBUF slot fits at 64 944 B; occupancy stays 1 block/CU, which
is what the shipped D=128 tile already runs at 42 064 B. See the issue comment
for the variant table and the split-K carve still owed.

Issue #11 milestone 2 (2026-10-24, branch `hd256-tile`, merged at `a667cdf`): the
fitted head_dim-256 WMMA tile, opt-in via `R3D_HD256` (default OFF). The milestone-1
prediction held exactly: fitted carve PAD0 (S pitch kBN+4) at 65 360 B, PAD1 (S pitch
kBN+1) at 64 592 B — PAD1 became the default after the paired pinned-clock bench put it
1.6-4.1% ahead on all ten geometries at bit-identical hashes. Dynamic LDS via
`hipFuncSetAttribute` once per instantiation (`r3d_launch256`), one block/CU, same as the
shipped D=128. v2/v3 and fp8-KV-256 refuse before any grid selection and the entry's v1
backstop catches every refusal; `R3D_NO_WMMA` still withdraws the whole family including
this path, verified by hash (kill-switch arm output equals v1 on every 256 case).

Numerics: new `tools/repro_attn256.hip` random-operand gate (signed descale fuzz
including negative scales, unmapped block-table entries, sliding window, ragged batches,
rows 1-130, ctx up to 8192): all seven hd256 cases within 2.204e-03 of the libref chain
— under kbench's 2.5e-3 tile gate, and the uniform ~2e-3 floor is the bf16 fragment
rounding against the f32 oracle (the hd128 controls show the same floor). The four-arm
hash matrix proves the tile executed (tile != v1 on every 256 case), the kill switch
returns v1 exactly, and the two pad arms are bit-identical. Gates on build_v117 are
baseline-identical: stock 2011/2011 (3.135e-03), full 4737/4737 (2.859e-03), and the
lever-ON stock replay 2011/2011 — the lever cannot perturb recorded geometry.

Bench (pinned clocks, AB_ROT=12 L2 rotation, one binary, env-lever arms): the tile beats
v1 on every geometry — decode 4.5-8.8x, prefill 53-106x (r512/ctx8k: 841 ms -> 7.9 ms).
Serve proof on qwen3.5-9b-mxfp4 spec 3, same-session lever pair: TTFT medians 13 786 ->
7 866 ms, decode 13.36 -> 27.27 tok/s, non-overlapping distributions, with the on-serve
launch proof captured: `DBG attn-wmma hd256 tile launch pad=1 dbuf=1 smem=64592`.

Honest limits: decode effective ~39 GB/s — 16 blocks over 96 CUs is the geometric deficit
split-K exists to fix. Desk sizing (milestone 3 groundwork) says hd256 split-K fits the
same fitted carve — SPLITK adds no LDS, the merge kernel is LDS-free, arena <= 4.03 MiB.
Implementing it is milestone 3. PAD1's 944 B margin means any future pad/tile growth must
re-check the 64 KiB ceiling. Enabling the tile in production is one env on the v117 build;
defaulting it ON is pending the teacher-forced KLD pair (output-changing at the bf16
fragment class — same gate discipline the q8x arm received).


## The q8x arm graduates: differential-KLD adjudication, default-ON promotion, and the replay law (2026-10-25)

The 2026-10-24 chronicle held the dot4 q8x mxfp4 decode arm opt-in on a teacher-forced
KLD FAIL (mean 8.27e-3 against 2e-3 gates). Re-audited against a same-flow floor: the
committed `compare` flow scores free-running decode trajectories against re-prefilled
forced contexts, and its base-vs-itself floor is mean 8.53e-3 / p99 0.147 / top-1 0.975
— past every threshold by construction. The old FAIL was instrument class, reproducible
from the floor; the arm numbers sat under it. Thresholds untouched (law: never move a
committed gate); the instrument got a differential mode instead
(`kldgate.py diffcompare`, 7f222ac/510d856): two forced artifacts of the SAME base
trajectory, identical conditioning per position — the prefill-vs-decode gap cancels.

Differential results (1258 positions, build_v117, lever arms of one binary): same-arm
floor (two fresh lever-OFF boots) mean 4e-5 / p99 1.5e-4 / top-1 1.000; q8x (ON vs OFF)
mean 5e-5 / p99 5.2e-4 / top-1 1.000. Arm-attributable delta +1e-5, at the floor — the
program stop rule (delta > +2e-3 mean or top-1 drop > 0.5%) not remotely triggered.
Honest caveat kept with the promotion: the score pass is prefill-only (n_predict=1), so
the decode-band arm moves one drafter-prologue launch there; the differential proves no
collateral drift, and the arm's quality envelope is the 480/480 attribution sweep
(worst 6.994e-3, inside the 3e-2 mxfp4a8 tolerance), the random-operand fuzz against the
independent decoder, the serve completion hash matrix, and the paired A/B.
hd256 re-scored with the same tool: mean 9.79e-3 / top-1 0.979 — ~240x the floor, real
drift; tile stays OFF, accept decision still the operator's.

Promotion (b818328, operator-approved 2026-10-25): sibling row
`gemm_nt_q_mxfp4_decode_q8x` (prio 1200) wins the band by default; `R3D_MXFP4_Q8X=0`
withdraws it at `rad_kernel_count`/`at` so the band RE-RESOLVES to the legacy walk —
the issue #4 kill-switch law, not a launch-time refusal. Two build-cycle catches:
the cross-TU helper `r3d_mxfp4_q8x_off` needs `extern "C"` on BOTH sides because
r3d_rows.cpp's body lives inside the header's `extern "C" {"` block — a bare
declaration mangles against the .hip definition and only explodes at dlopen (44b614b);
and the arm's growing activation scratch broke the engine's recorded-pass replay —
`m4_act_i8_kernel` must issue the same scratch pointer every time a recorded pass
replays, but the grow-on-larger-K buffer swapped pointers mid-life, poisoning the drafter
pass (-1) the first time down_proj (K=intermediate) grew it after the pass was recorded
("arguments differ at byte 9" -> engine stop). Fix: one immortal buffer per K (316cd01).
Recorded-pass replay is a NEW gate surface for any plugin kernel holding plugin-owned
scratch: kbench direct-launches and never sees it.

Gates on build_v119 (the scratch fix is post-v118): all baseline-identical — stock
2011/2011 (6.952e-3), full 4737/4737 (6.994e-3) and its lever arms `=0` (2.859e-3) /
`=1` (6.994e-3), attribution 480/480 twice, dflash2 702/702, df3 768/768; dot4 fuzz OK
(all 16 codes, 2^20 sudot4 draws, device e8m0); repro_w2a8 bit-exact at every M;
rows_dump 5-arm kill-switch matrix exact (default/`=1`: sibling at prio 1200; `=0`:
sibling gone, legacy stays; `NO_FP4_DEC`/`NO_MXFP4` cover the sibling too).

Serve A/B on the XTX, sequential pairs, flashnext untouched throughout: v116 medians
31.07 / 30.14 / 31.13 t/s vs v119 35.51 / 34.92 / 36.39 — +14.3 / +15.9 / +16.9%,
non-overlapping distributions (the older-era +23.5% was measured on a different base;
this is the honest current number). Four-arm completion hash matrix: default ==
explicit `=1` (combined sha256 f4b40ce5...), `=0` == the v116 build (aacd15f7...) —
the kill switch restores legacy behavior BYTE-IDENTICALLY, and the default is the arm.
Debug boot: `libr3d: mxfp4 q8x dot4 arm launched (mt=1)`, 0 host ops across 4 recorded
pass classes, 0 plane fallbacks, acceptance 144/171, coherent chat.

Promoted: spec9g renamed to spec9g_v116_rollback (chain v113/v112 kept), fresh spec9g on
build_v119, inspect-faithful (12 GiB cap, host net, --init, spec 3/ctx 4096/4 seqs/
all_vram/mm 0, ENV pin ROCR=1). Card law note: the HOST DRM order moved (XTX is card2
by DRM number) — DRM numbers do NOT index ROCr; the authoritative proof is the serve's
own boot line, verified: `1 device(s) [0:gfx1100]`, `card 0000:17:00.0 rank 0`,
selection libavx 736 / libr3d 736, weights 6.65 GiB VRAM mapped 0 B, 0 real refusals,
67.4 t/s on a live chat completion, 88/99 drafter accepts.

## The review queue opens: objdump-gated closures and the three ledger fixes (2026-10-25)

`docs/REVIEW_QUEUE.md` (external-model passes) arrived with three owed ledger
entries and two objdump-gated closures. All adjudicated against the SHIPPED
v119 code objects (fatbin carve, per-target slice, llvm-objdump) before any
card time was booked — the gpu-kernel-dev law that compiler-codegen claims are
adjudicated by disassembly, and it paid immediately.

**Ledger fix 1 — the `R3D_FP4_DOT2` leg.** Present in `r3d_gemm_mxfp4.hip`
since issue #6 (M==1 only, opt-in, +5% best / 0% at deep ctx, default-off
pending the profiling re-attribution), absent from this ledger until now.
The external reviewer citing "a ledger entry that doesn't exist" was citing
real code through a gap in OUR record. Recorded: the leg exists, the numbers
above are its measured verdict, and it is NOT the shipped path.

**Ledger fix 2 — flat-vs-global load selection: CLOSED, compiler already
optimal.** v119 attention-family slices: 2,902 `global_load_*`, ZERO flat,
ZERO buffer loads; m64 tiles 110 global / 0 flat. There is no load-class
choice left to make. (The queue's 1,192-vs-11 count was a per-slice subset;
the full-family scan is strictly cleaner — the closure stands on better data.)

**Ledger fix 3 — the ds-pipe traffic census (motivates Q6/Q11).** Per-kernel
`ds_bpermute_b32` totals from the shipped objects: the queue's "96/iter in
q8x MT=8" actually describes the W2A8 gemv_q8 family (96 -> 148 across
MT 1..16, x3 instantiations); the mxfp4 q8x production row itself carries 40
at MT=8 — five `__shfl_xor` folds per m row (two integer, three float), so
the Q6 target is half the advertised size on the serving kernel. The
attention WMMA body carries 8,736 `ds_swizzle_b32` aggregate across its
42 instantiations (232-404 per kernel — all from rocwmma fragment staging,
zero from our hand-written code). Dot instructions confirm Phase A/B shipped:
465 `v_dot4_i32_iu8` + 30 `v_dot4_i32_i8` across the family slices.

**Q2 — the `s_barrier`-drains-loads claim — closed WITHOUT card time, the
ISA-first rule applied.** The donor rationale (verbatim in Strata + hipfire:
"s_barrier drains all in-flight global loads") is GCN-era law. The RDNA3
ISA reference (S_BARRIER, p.254) says the opposite: "Barrier instructions do
not wait for any counters to go to zero before issuing." The v119 disasm
agrees with the ISA, not the donors: half the attention barriers are bare
(`s_waitcnt lgkmcnt(0); s_barrier` with in-flight global loads crossing the
barrier unharmed), and the other half carry explicit `vmcnt(0)` because
`__syncthreads()` lowers through LLVM's conservative wait-insertion, which
hoists `vmcnt(0)` whenever any vector global is live near the barrier — that
is a CODEGEN artifact, not a hardware law. The DBUF arm's prefetches are
therefore being drained by LLVM's wait insertion, not by the barrier — which
means Q2's proposed donor idiom has nothing to fix at the hardware level, and
any residual gain is an LLVM wait-placement problem (attackable, if ever, by
narrowing live-global windows, not by porting Strata's wait-idiom). Status:
closed-as-misdiagnosed, with the disasm numbers recorded above as the record.

**Q3/Q4/Q8 batch shipped the same day** (elementwise u32 walks: actmul
Contig/W32 arms, dflash_conv u32 + tap-modulo hoist, band block-uniform split
with the latent grid-x-wrap guard, kv_store VEC uint4 walk; q8x/w2a8
launch_bounds pins), gated on build_v120 in the same window. Rope was already
u32 in place (claim verified-then-rejected).

## The review queue's first harvest: batch built, NT promoted, spec9g on build_v121 (2026-10-25)

**The batch (Q3/Q4/Q5/Q8, commits 7156a03, f5e82dd, a748287).** Elementwise i64
walks narrowed to u32 where provably in range (actmul Contig/W32 template arms
on the cast precedent; dflash_conv's three divisions u32 via the launcher's
existing 0xffff*kThreads cap, with the per-tap modulo hoisted; band's
block-uniform split, which ALSO closed a latent grid-x-wrap the old code could
hit at ngrp > 2^31 — refused at launch now). kv_store gained a VEC uint4 arm
gated on 16B alignment + sd==1 with the scalar walk kept for odd layouts. The
Q5 non-temporal builtin rejected the HIP vector type — the leg loads u64 +
bitcast (same `global_load_b64`), compiled on the second attempt.

**Verification chain, all on-card:**
- Gates build_v120: 6/6 (stock 2011/6.952e-3, full 4737/6.994e-3, attr 480,
  df2 702, df3 768, fp8 536) — and a per-op diff of the full pack, 36 libr3d
  rows v119-vs-v120: ZERO worst-rel_l2 diffs. Every touched row runs device
  per the report's reachability table.
- Kernel-descriptor census (Q3's own gate): the shipped register allocations
  already satisfied both pins (q8x 22-47 VGPR, w2a8 45-62, zero spills) —
  pins closed as codegen-no-ops, KEPT as contracts. Side-bonus measured:
  the u32 rewrite shrank footprints (dflash 21->17 VGPR/34->26 SGPR, band
  38->27 SGPR).
- Serve A/B (same box sequential, port 9022, probe pair-differenced ctx-512,
  3 alternating rounds, every arm boot-asserted gfx1100-rank-0):
  v119 medians 44.73/44.49/44.39; v120 (batch) 44.85/45.19/47.48 — positive
  all rounds, ~+1.6%; v120+NT 46.03/45.96/45.86 — the within-binary NT pair
  is +1.2/+1.7/+1.4 (3/3 positive). The donor's -3% engine warning did not
  reproduce; the +10-12% stream claim was overstated, real is ~+2%.

**NT default-ON (f5e82dd, build_v121).** A cache hint on identical bytes has
no numeric class — the differential-KLD machinery does not apply and was not
needed; the gates carry the proof: build_v121 battery 7/7 with NT on by
default, all worst-rel_l2 identical to baseline, plus the R3D_Q8X_NT=0 lever
arm re-ran the stock gate byte-identically (lever semantics = literal "0",
the q8x law, gate.sh passes it through).

**Promotion:** spec9g -> build_v121 (rename-rollback to v119 kept; chain
v119/v116/v113 deep). Boot assertions: `[0:gfx1100]`, PCI 17:00.0 rank 0,
736/736, 6.65 GiB VRAM mapped 0 B, 0 real refusals; live completion coherent
at 53.05 t/s, drafter 11/21 accepts. flashnext untouched the entire window.

**Harness lessons (both bitten, both fixed in-suite):** the serve-boot gate
`load: rank 0` fires BEFORE the HTTP bind — every arm must poll /health
until 200 before probing (first A/B pass burned 9 probes on this); and a
fresh build dir created by a root container needs container-side cleanup
(rm as host user fails) plus a source-sync check — an identical md5 between
two "different" builds is the tell that the tree did not sync.

## Q7 measured: depth-2 prefetch built, gated, lands NULL on the serving arm (2026-10-25)

The Q2 close transferred its residual (wait placement -> prefetch DEPTH) here.
Built on `61e4432`: `R3D_Q8X_PF=2` opts into a register-staged depth-2 walk on
the q8x decode gemv -- the 136B-group code + scale byte for iteration it+2 load
into a parity ping-pong on explicit VGPR locals before iteration it's dot chain
consumes its operands. Parity ping-pong on named locals (an array indexed by
`it&1` would sink to private memory). Math order verbatim in both arms: pf
moves issue cycles, never bytes-to-instruction mapping -- bit-identical by
construction, no numeric class, no differential KLD. Default-off (unset or any
other value == the shipped walk).

Gates (build_v122, battery 9/9): stock 2011 (6.952e-3), full 4737 (6.994e-3),
attr 480 (6.994e-3), df2 702 (7.306e-3), df3 768, fp8 536 (3.686e-4) -- every
fingerprint identical to the shipped v121 line; the three PF=2 replays
(stock/full/attr) likewise, and per_op_cmp shows ZERO worst-rel_l2 diffs on
v121-vs-v122 (34 ops), stock-vs-pf2 (34), full-vs-pf2full (36),
attr-vs-pf2attr (1) -- the bit-identity claim holds on measured outputs.

Serve A/B (spec9g stopped for the pass -- two all-VRAM serves never coexist on
the 16 GiB card; 6 arms same binary, env apart, 3 alternating rounds, all six
boot-asserted gfx1100 rank 0 + /health-polled): pf1 medians 52.67/52.33/51.50
vs pf2 52.61/52.52/51.59 t/s. Noise band, no signal. The lever was verifiably
ON in the measured arms: live re-check `mxfp4 q8x dot4 arm launched (mt=4
pf=2)` (the boot.log snapshots in the A/B script predate the first request,
which is when the once-flag debug line fires -- harness lesson, move the
capture after the probe).

Verdict: NULL, kept DORMANT default-off (same disposition as the #8 arms).
Depth is not the binding term of the q8x walk -- at 16 waves/CU occupancy the
walk's latency is already hidden, which is the same occupancy answer the #7
PF-window falsification hinted at and the #8 wave study named; this time it is
measured directly on the serving arm, as the schedule-transfer law demands.
The ballot-gated lazy rescale half of Q7 moves to Q6's scope -- the rescale/
fold chain is exactly where the DPP folds live.

Harness note: the deterministic-text probe (`temperature 0` completions) is
byte-STABLE across arms but INCOHERENT on this checkpoint (no
generation_config.json -> whole-vocabulary live, the boot warning says so; the
sample reads "help me" in Chinese at "The capital of France is"). Fine as a
hash matrix, useless as a coherence check -- coherence stays asserted via the
live-chat line on promotions and the battery carries correctness.
