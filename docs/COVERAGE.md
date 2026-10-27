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

## Q6: DPP row_xmask folds — the review queue's first shipped win (2026-10-25)

Mechanism, verbatim from ISA Table 94: DPP_CTRL 0x160|mask (ROW_XMASK)
fetches lane (n ^ mask) within the 16-lane row -- the butterfly partner,
delivered on VALU (`v_mov_b32_dpp ... row_xmask:k`). GCN's row_xor (0xe)
does NOT exist on RDNA3 -- LLVM renders it as a quad_perm and the folds
come out wrong (probe v1: 32/32 lanes mismatched; RDNA replaced the GCN
DPP table with DPP8/DPP16, chapters 15.3.8-9). The shipped `__shfl_xor`
chain lowers to `ds_bpermute_b32` -- DS-pipe traffic that competes with
every LDS read on the machine.

Probe gate (q6probe2, exact q8x fold shape int 1/2 + float 4/8 + float 16):
PROBE-BITEXACT, all 32 lanes vs the shipped chain. xor16 crosses the row
(rows are 16 lanes) and stays shfl; V_PERMLANE16/X16 (opcodes 603/604) are
the cross-row gathers if a later campaign wants them -- deliberately NOT
used yet: 7.2.8 makes PERMLANE adjacent-to-V_CMPX illegal and inline asm
gets no LLVM nop-insertion for it.

Ship path: lever `R3D_Q8X_DPP` on the q8x decode gemv (both walk arms share
the fold helpers). v123 (default-off): battery 9/9 all baseline fingerprints,
per_op zero-diff stock/full/attr lever-vs-off, serve A/B 3/3 positive
non-overlapping (same binary, env apart): base 41.52/42.42/42.11 vs dpp
43.38/43.65/43.66 t/s -- +4.5/+2.9/+3.7%, lever-ON boot-proven live
(`mt=4 pf=1 dpp=1`). Promoted default-ON (`8faec08`, NT evidence class:
bit-identical, no numeric class, gates carry the proof); `R3D_Q8X_DPP=0`
withdraws to the verbatim shfl chain (literal "0" law). v124 battery 9/9:
default (dpp-on) and =0 arms all baseline-fingerprint, zero per-op diffs.
**spec9g -> build_v124** (v121 rollback kept; chain v121/v119/v116/v113):
boot-asserted [0:gfx1100], rank 0, mover 6.65 GiB, 736/736 bands, health
200, coherent chat (reasoning-model trace reads correctly), decode probe
44.18 t/s ctx-512 on the shipped default.

Two same-day results bracket the bound: Q7 depth was NULL (load latency is
hidden by 16 waves/CU occupancy -- the #7/#8 story again) while Q6 folds
bought +3-4% (the DS pipe was the contended resource). A/B absolute values
run lower on some days (52 vs 42 t/s) -- card clock state; only same-session
pairs are comparable.

Harness notes: the once-flag launch proof prints at FIRST REQUEST, after the
boot.log a naive script snapshots -- capture logs AFTER the probe; and this
checkpoint (no generation_config.json) emits `reasoning_content` before
`content`, so a 12-token chat probe reads empty content -- budget >= 200 for
coherence checks.

Unrelated-but-noted: `radiance-flashnext` was RECREATED on cat at 14:10 AWST
with a different model (qwen4exp, 262k ctx) -- operator action, predates this
window's A/B, untouched by it.


## Q9 probed, folded, measured NULL: the bf16 donor instruction does not exist on this card (2026-10-25)

Q9 asked for `v_dot2c_f32_bf16` in the decode GEMVs "probe the 3-operand form first".
The probe is the whole story of the first half, so the record keeps all of it:

**bf16 is hardware-present, assembler-absent, and the ISA explains both.** ROCm 10.1's
gfx1100 inline-asm table rejects `v_dot2c_f32_bf16` ("instruction not supported on this
GPU"), `v_dot2acc_f32_bf16` ("invalid instruction, did you mean: v_dot2acc_f32_f16?"),
and the `_e32` suffix. Cross-checked against the ISA corpus: RDNA3 has NO VOP2 bf16 dot
form (VOP2 dot family is f16-only); VOP3P `V_DOT2_F32_BF16` (opcode 26) exists but the
16-bit VOP3P encoding places src3 in the literal pool only -- it cannot accumulate into
a register; bf16 accumulate exists ONLY as the VOPD dual instruction
`V_DUAL_DOT2ACC_F32_BF16` (ISA p.370/372, "initial value in D is used as S2"), which is
not inline-asm reachable. Donors writing bf16 dot2 proposals for "gfx11" are writing
them for RDNA3.5+, where the VOP2 form exists. (Hand-encoded VOPD stays a theoretical
door; closed as not worth the hazard surface for a fold this size.)

**The fold survives on f16, exactly as the shipped mxfp4 dot2 leg (issue #6) does.**
e4m3 embeds EXACTLY in f16 (<=4 significant bits, |v| in {0} U [2^-9, 448], every value
an f16 normal), products exact in f32. Probe v2/v3 (XTX-asserted): the tie discriminator
says the pair is summed with the adder's guard bits BEFORE the fused add to acc
(1+2^-23, not 1.0); the narrow-exact-band random battery matches none of ascending-fma,
pair-rounded, or exact-pair-single-round models -- the in-lane order is opaque at
maxrel 6.5e-4..1.1e-3 per 128-chain. Carve-objdump lowering proof: the 3-op asm resolves
to VOP2 `v_dot2acc_f32_f16` (16-bit encoding, no clamp bit). This is the same
tol-graded class issue #6 shipped on.

**The arm.** `gemm_nt_fp8a8_gemv_kernel<MT, W, D2>`: LDS 256-entry e4m3->f16 weight LUT
(4 gathers + 2 packs REPLACE the 4x ~10-VALU decode chain, hoisted out of the m loop),
activations stage as packed f16 pairs (xs16, half the LDS), inner = 1 LDS.64 + 2 dot2c
per row per group. Lever `R3D_FP8A8_D2` (literal "0" withdraws; default OFF, this item
never earned promotion). build_v125 (md5 8ebf6f4797296fb13becc08457acfb10), battery
10/10: six default arms byte-identical vs v124 fingerprints (34+36+1+1+1+2 ops zero
diff); four lever-ON arms pass, with the fp8a8 decode ops moving from exactly
0.00e+00 to 1.35e-08 (M=4) / 2.93e-06 (M=1) / 1.89e-05 (M=8) against tol 5e-03 --
order-only drift, three decades under budget, and per-op POSITIVE proof the arm ran
(the per_op_cmp zero-column masked it: its regex compares the tol column, the reports
themselves diff).

**Serve A/B (qwen3.5-9b-fp8, port 9008, spec9g stopped, same binary one env apart,
d2=0/d2=1 boot-proof every arm, refusal lines equal):** off 19.99 / 21.77 / 21.61
vs on 20.41 / 20.19 / 18.49 tok/s ctx-512. One round positive, two negative, r3_on
outside the entire off range; on-mean -6.7%. No positive structure to promote on.
CLOSED NULL, arm stays dark. Issue #8's law re-proven from the other side: the mxfp4
plateau fell to GEOMETRY (dot4 repack), not instruction count -- and the fp8a8 row,
whose dword-per-lane walk was already aligned one-wavefront-per-group geometry, was
never instruction-count-bound, so removing ~30 VALU/lane/group bought nothing while
the LUT gathers added LDS pressure.

**Harness notes this window (now laws):** containers on cat need
`--device=/dev/kfd --device=/dev/dri --group-add video` (hipGetDevice -> count=0
without them, and the failure looks like a card problem); ROCm 10.1 fatbin carve goes
through `--save-temps` + the `*.hipfb.amdgcn.gfx1100.img.lto.o` ELF (clang-offload-bundler
segfaults on the new bundle format); f16 host decoders want the sign/exponent fields of
the RAW 16-bit word, not a shifted-to-f32 pattern (two probe arms burned on that off-by-one);
`pkill -f <script>` on cat self-matches the driving ssh command -- use the [b]racket pattern.
If an fp8 serve model returns to the hot seat, the retry knob is LIBR35D_TRIAGE item 3's
3-instruction arithmetic e4m3->f16 widening in place of the LDS LUT.

Unrelated-but-noted: operator merged PR #14 (docs/LIBR35D_TRIAGE.md + tools/spill_census.py,
the gfx1151 radiance-strix-halo triage). Its "queue deltas" section is intake-logged as
Q17, parked on the operator's "to be looked at later" instruction.

## 2026-10-25 window 7: Q17 classified; spill census verified and upgraded v2; TPR zero-hit

Operator released the libr35d triage for look+classify (explicitly the spill test: "we may
be missing things"). Desk audit of tools/spill_census.py v1 found four blind spots
(SGPR spills unread; VGPR/LDS sizes unprinted so non-spilling pressure is invisible; stale
census commit; open w2a8 private-segment question) and one non-issue (src/ is flat, the
one-level glob misses nothing). Fresh full container build at 75a4eca (26 units, 193 gfx1100
kernels) confirmed the donor census numbers carry (same 25 flagged rows as 61e4432), then the
extended census found what v1 structurally could not:
- gdn SGPR spills 95/79/67/49 on top of the printed VGPR spills; all gdn recurrent kernels
  and every attn kernel2 arm pinned at the 256-VGPR ceiling.
- NON-spilling near-cliff arms: attn_paged_wmma_kernel <256,3,0,0,*,*> at 249/242 VGPR —
  the non-DBUF h256 MT=3 family, one edit from spilling (v1 prints nothing here by design).
- w2a8 q8 GEMV private segment = real traffic: 16x scratch_store_b128 + 1 scratch_load site
  in <8,16> (8 stores <*,8>), the activation-staging b128 window demoted to local memory —
  the donor's LICM trap on our row, not metadata noise. Q18 filed ready.
TPR<=32 butterfly audit CLOSED zero-hit (site-by-site in the REVIEW_QUEUE closed-table).
Census tool rewritten v2 (ssp/near-ceiling/LDS/scratch-flag) and re-verified on the same
tree. Deltas folded: Q1 (rotation gate + MT>=2 gate + rejected-variant skips), Q10 (3-inst
widening constraint), Q12 (L2-convoy risk note), new Q18-Q21, card bench protocol adopted
(verify widths, L2-rotating harnesses, A-B-A-B, rocprofv3 limits, census-after-change).
No kernel code touched; spec9g untouched (build_v124, health 200); no serve ran this window.

## Q18: the w2a8 register window — census trap fixed, MT-gated, shipped in build_v127 (2026-10-25)

The census finding became the fix: the q8 GEMV's prefetch window was a `uint4 bwin[W]`
consumed through the runtime `d` of a `d < rem`-bounded loop — an array with a dynamic
index cannot live in registers, so LLVM demoted the whole window to per-thread scratch
(144/272 B priv on all ten W>1 arms; 16x `scratch_store_b128` per iteration at W=16,
8 at W=8). Fix shape (`a805b84`): BOTH the preload and the consumption loop carry
`#pragma unroll` to the compile-time trip W with a BLOCK-UNIFORM `d < rem` guard
(rem is a function of wbase/nd only — barrier arrival counts stay equal across the
block, the predicated-barrier legality). Every `bwin` index is then static and register
residence is the compiler's only option. Verified at the instruction level: the gfx1100
`<16,8>` disassembly (11,239 lines, 528 sudot4 — non-empty slice, right code object,
co1 = flags 0x41) contains ZERO scratch instructions; the `sgpr_spill_count` 178/439 on
the MT=16 arms are reserved-and-unused slots (metadata only, disasm rules).

The rollout then produced the window's most instructive result: measured WHOLE-TILE,
the register window was a split win (v124 vs v126, same card, same session, alternating
rounds, all five shape classes non-overlapping, log ~/scratch/r3d/q18_ab.txt):
M=1 K=8192 +24.7%, M=1 K=16384 +29.5%, M=4 +15.0%, but M=7 (MT=8) -8.1% and M=14
(MT=16) -11.4% — the W-fold body copies price the wide tiles past what removing the
scratch traffic buys. That is the schedule-transfer law (per-family, once written in the
ledger for sibling kernels) biting inside ONE kernel across its own MT ladder. The ship
(`e78f0f4`) gates the new walk at `MT <= 4`; MT>=8 keeps the pre-Q18 walk verbatim, and
the dword iteration (stage/sync/dot/fold/flush/sync) moves into one `__device__
__forceinline__ q8_dword_iter<MT>` shared by both arms — the bit-identity is now a
SOURCE property (one body, both fetch paths), not a bench finding.

Census before/after (gfx1100, v124 -> v127, per arm MT,W): MT<=4 arms priv 144/272 -> 0
at 85-122 VGPR (all under the 128 cap for W=8; the W=16 opt-in arms at 117-122 are the
only ones near the cap); MT>=8 arms byte-identical to v124 (priv 144/272, 87-90 VGPR).
Gates on build_v127 (e78f0f4): repro_w2a8 BIT-EXACT at every M (rc=0, rel 0.000e+00,
13/13 Ms); battery 6/6 with baseline fingerprints identical to the v125 line (stock 2011
6.952e-3, full 4737 6.994e-3, attr 480 6.994e-3, df2 702 7.306e-3, df3 768 6.994e-3,
fp8 536 3.686e-4); per_op_cmp v125-vs-v127 ZERO diffs on 34/36/1/1/1/2 ops — the
zero-diff class claim measured, per the row's own instruction.
Confirmation A/B v124 vs v127 (3 alternating rounds, ~/scratch/r3d/q18_ab2.txt):
+25.9/+31.7% (M=1, non-overlapping), +15.6% (M=4, non-overlapping), +1.3/+0.2% with
overlapping spreads at M=7/M=14 = parity, exactly the gate's promise.

Row-level tooling born with it: `tools/bench_w2a8.cpp` — repro_w2a8's operand
construction (the joined scale,zero blob no kbench recorder can describe), argv shapes,
plane rotation for the L2 law (q8 plane N*K bytes per launch; >=96 MB at one plane),
median-launch timing as GB/s, and a nonzero-output liveness assertion (the all-zero
fixture lesson armed in the harness itself). Build line in the file header.

Serving context: spec9g (qwen3.5-9b-mxfp4) never issues `gemm_nt_q_w2a8_decode` (zero
w2a8 lines; its drafter is the container's own, chained-launches-off) — the band serves
only DFlash2-style w2a8 draft heads, so no serve swap rode this change: the row ships
for the next drafter deployment on build_v127+, and the live serve stays build_v124.
Residual: Q18b (Wc=4 middle-ground window for the MT>=8 verify band — the scratch
demotion is still paid there, now MEASURED as the cheaper cost, so Q18b needs a
schedule that beats BOTH, e.g. partial unroll or a 4-deep sub-window; gate behind a
fresh bench win). Q12 (compressed-input dot4 on the same row) should re-read this
section before touching the walk.


## Q1 SHIPPED default-ON: b64 staging on the m64 WMMA tile, pad-16 falsified, rotation falsified (2026-10-25, window 9)

The queue asked for attention's vec staging on the m64 tile loader plus the Q17
donor fold-ins (rotation gated K%4096, staging gated MT>=2). All three shipped as
one kernel (gemm_nt_bf16_m64_vec_kernel<ROT,PAD16>, shipped scalar kernel
untouched) and split into three verdicts on the same gates:

THE WIN (shipped default, build_v129): uint4 global loads + b64 LDS stores,
rows keep the SHIPPED pad 4 (a 40B row is 8B-aligned at every 8-col slot, so no
repadding is needed); threads 0-127 stage A, 128-255 stage B (wave-aligned, no
divergence), slot bijection = bit-identical slices. Descriptor census:
b64 arms vgpr 64/66 lds 25088 = BYTE-EQUAL to the shipped DBUF arm (72/25088)
-- occupancy preserved. Kernel A/B (bench_m64, 3 interleaved rounds, same
binary env apart, L2-rotating B planes): vs scalar +6.2% (1024x4096x4096),
+9.0% (1961), +7.5% (1024x11008), +7.0% (K=3584 odd), +18.1% (K=4104 odd),
ALL non-overlapping spreads; MT=1 control rows parity. Bit-identity: md5-equal
Y dumps across scalar/vec16/b64/b64+rot arms at every shape that ran.
Battery v129: 8/8; full pack BYTE-IDENTICAL to v124; stock pack identical
except the gate-budget note (1.45->1.58 GiB is gate.sh, predates this); the
VEC=0 withdrawal arm stock byte-equal too. Serve A/B (v128 shadow container,
bench9b 3281-token prompts): TTFT base med 18.861 vs b64 18.819 -- positive,
non-overlapping by a hair; the bf16 tile is a small slice of this model's
prefill, so the kernel-level SEP wins are the load-bearing evidence (Q18's
promotion class). spec9g now on build_v129 (rollback spec9g_v124_rollback kept).

THE FALSIFICATION (VEC=1, dark): uint4 LDS stores force 16B-aligned rows ->
kVecPad 16 -> LDS/WG exactly 32768 -- two workgroups would need the FULL 64KB
CU bank and the second one falls off. Measured -0.6..-9.1% (non-overlapping
losses) at MT>=2. The queue's "kLdsPad 4->16" was the poison in the donor's
recipe: padding bought the store width at the price of half the occupancy, and
that price dominates. Astra's b64 fallback is what the donor's win actually
needed (their engine's tile was presumably already pad-friendly).

THE FALSIFICATION (ROT, dark): (blockIdx.x+blockIdx.y+ks)%32 row-issue rotation
was a consistent LOSS on top of every staging arm (-2.4..-6.3% off its own
baseline, mostly SEP), never a win. The donor's 60-vs-185 GB/s channel-camp pair
does not reproduce here: the m64 wave already reads 2 rows x 32B contiguous per
instruction (their law bit at 64B/16-step granularity -- skipped donor variants
stayed skipped), so there is no camp to break and the rotation only scatters
the row-walk. L2-convoy thinking (Q17 delta 7) covers this card's camping
regimes; per-tile row rotation does not.

Harness: tools/bench_m64.cpp (arm-expectation line mirrors the launcher gates;
liveness %; Y-dump for cross-arm md5). Runners cat ~/scratch/r3d/q1_ab{,2}.sh.
Shadow-container note: the first one went zombie on teardown without --init (took
a second before the runtime reaped it); q1serve2 ran --init and died cleanly.
Always --init for serve shadows.

## Q10 closed SHIPPED: fp8 KV staging was a byte walk, and it was the whole story (2026-10-25)

The queue's literal premise (donor 3-inst `v_perm + v_pk_ashrrev_i16 + v_and_b32`
e4m3->f16 widening) is FALSIFIED for this kernel family before any card time: the
WMMA attention fragments are pinned `bf16` (the f16 output cannot be consumed), and
the widening trick's exactness rests on the f16 subnormal lattice catching the
shifted e4m3 subnormals arithmetically -- bf16 has no such lattice at that scale.
But the disassembly of the shipped v129 fp8 staging loop told the real story, and
it was bigger than the decode: per-BYTE `global_load_d16_u8`, an exec-mask branch
DANCE around the shipped ternary decode (`v_cmpx_ne` + `s_xor`/
`s_and_not1_saveexec` + `s_cbranch_execz` + re-merge), and per-byte 16-bit LDS
stores. Staging, not attention, was the fp8 cost center.

The shipped arm (R3D_F8ST default 3): a 256-entry per-block LDS LUT built from the
SHIPPED `fp8_to_bf16_w` (identical decode by construction, NaN codes included),
8-byte global gathers (kv_sd==1 + RB[]%8 gates, Q1 pad law honored -- zero LDS
repadding; row starts are already 16B at pitch D+8), and 16B LDS stores. Gate miss
reruns the tile on the scalar LUT walk (still ~+9%). Dark arms: 1 = branchless VALU
decode (the select IS the branch's truth table -- both sides are pure), 2 = LUT
scalar, 0 = verbatim withdrawal. R3D_DBG forces verbatim (the dbg wire bypasses
stage_kv_tile; same law as DBUF).

Identity: battery 10/10 on v130, arms st0..st3 all 160/160 attn cases on the fp8
pack with worst rel_l2 0.00e+00; full/attr/df2/df3/fp8 byte-identical v129.
Promotion battery on v131 (default-3): all six legs green, withdrawal arm
byte-equal to the recorded st0 report, stock/full byte-identical v129.

A/B (v129 vs st0/1/2/3, 3 interleaved rounds, hash-equal on every geometry,
KVH=4 + 12 rotating planes to put decode KV past L2): st3 **+33..+40% hd64,
+75..+154% hd128** (decode and prefill alike), st2 +8..11%, st1 +3..6%, every
spread disjoint; st0 (the codegen-jitter control) +-0.3%.

The st0 control existed because of a NEW TRAP found this card-pass: declaring the
LUT as a dead 1-element `__shared__` array for the F8ST==0 instantiations --
never referenced, `if constexpr`-guarded away -- RE-SCHEDULES the prologue of
EVERY instantiation in the TU (bf16 arms included; instruction-level diff moves
from the 10th line while vgpr/lds descriptors stay byte-equal). Compiler-codegen
claims about UNCHANGED arms must therefore be adjudicated by output fingerprints
(the battery) plus a timing control (st0), not by objdump. Filed for Q19.

Serve: spec9g serves bf16 KV and never issues these arms; v131's bf16
fingerprints are byte-identical v129, so per the Q18 precedent no swap rode the
change (serve stays build_v129; the next fp8-KV deployment mounts build_v131+).

Harness: attn_bench AB_KVDT=f8 (fp8 planes, NaN codes excluded, byte-width
allocs, dtype tag in the print); gate.sh passes R3D_F8ST + R3D_ATTN_NO_DBUF into
the battery env. Runners cat ~/scratch/r3d/q10_gates.sh, q10_ab.sh,
q10_promote_gates.sh. Two host-execution laws re-earned: bench binaries run
INSIDE the device container (host hipGetDevice = phantom run), and scratch paths
mount at /scratch in-container, not /scratch/r3d.


## Q12 SHIPPED default-ON (MT<=8): compressed-input dot4 -- drop the u8 expansion plane, expand inline (2026-10-25, window 11)

The queue's 4x-traffic premise held, and the donor-law risk note (delta7, L2
convoy) fired exactly where it said it would -- as a BENCH-HARNESS trap, not a
kernel defect. The row now walks the engine's packed u2 plane directly: one
dword per lane per window unit (16 codes) expanded in registers to the shipped
`b4` operand, tail one byte per lane. The repack kernel's bit-expansion moved
into ONE `__device__` helper (`w2_expand4/16`) used by the offline repack AND
both C4 fetch sites: "expanded inline" == "repacked offline" is a source
property (the Q18 law again), and `q8_dword_iter` never learns which world the
operand came from. Lever `R3D_W2_C4=0` withdraws to the verbatim repack walk.

Census (save-temps, gfx1100, all 15 C4 instantiations): every C4 arm priv=0 and
the whole TU zero scratch instructions -- the Q18 trap avoided at every tile,
including the MT>=8 arms whose runtime-indexed pwin LLVM kept in registers
anyway (packed = W VGPRs). Pressure at the W=16 ceiling arms DROPPED: 117-122
(C0) -> 70-81 (C1). The 15 C0 descriptors stayed byte-equal to v131 pre-flip.

A/B (same binary v132, lever the only axis, 6 paired samples/shape, EVERY
spread non-overlapping -- ~/scratch/r3d/q12_ab2.txt). Regimes labelled, because
the packed plane is 4x SMALLER and the rot<=4 harness (repack-cache bound) can
leave it CACHE-RESIDENT while the repacked set streams cold:
  cold-vs-cold (both sets past the card cache, K=16384 N=16384 rot=4):
    M=1 +24..30%, M=4 +7..14%, M=7 +12..16%.
  cache-fit regime (packed set resident -- the regime the packed walk EXISTS to
    win for per-token head streams): M=1 K=8192 rot=4 +15%.
  MT=16 wide tile: -1..-3% non-overlapping LOSE -- expansion VALU prices the
  verify tile past the traffic cut. THE Q18 MT-LAW REPRODUCED INSIDE THE NEW
  ARM: the ship gates `c4 = mt <= 8 && !R3D_W2_C4(0) && !R3D_NO_W2A8_Q8X` (the
  family kill-switch withdraws BOTH q8 fast arms, sibling-row law); MT=16 keeps
  the repack walk and its cache verbatim, filled on demand by the first wide
  call. The first (preview) A/B pass was DISCARDED: the v132 rebuild raced the
  bench's dlopen and the unlabelled rot=1/4 shapes mixed cache regimes.

Promotion (v133): repro 13/13 bit-exact on all three arms with per-arm liveness
lines (default=q8c, C4=0=q8, NO_Q8=u2); battery 7/7 with all seven fingerprints
zero-diff; and the new cheap promotion proof: the flip touched ONLY the host
launcher, proven by full instruction-stream diff of the save-temps gfx1100
slice -- 46/46 kernels byte-identical (asm_identity.py; first parse was
VACUOUS, kernels=0/0 -- label lines carry `; @` trailing comments and the
empty-slice trap the ledger warns about nearly caught us again; non-emptiness
assertion now baked in). The 46-way asm identity is what licenses skipping the
second battery pass for the withdrawal arm (the withdrawal shares device code
by object; the 3-arm repro is its row-level proof).

ABI trap, new era-marker: gate.sh now runs kbench from radiance-dev:v1.11.0
(engine ABI 26); a plugin configured inside radiance-dev:v1.8.1 is ABI 22 and
the harness REFUSES the whole library ("built against ABI version 22; this
engine is version 26"). v131 rode that cache unknowingly; build_v132/133
fresh-configured in v1.11.0. BUILD IMAGE = GATE IMAGE ABI, check before any
battery; the first v132 battery died in 60 s on this (all legs refused, the
difflines of that run are garbage, keep only the second log). Also: direct
dlopen harnesses (repro/bench) do NOT check the ABI -- they run green while the
engine refuses, so a green repro is not battery insurance.

Serve: spec9g never issues w2a8 lines (Q18 established); no swap rode Q12,
serve restored to build_v129 home after ~40 min card session, health 200,
boot-asserted gfx1100 rank 0, coherent chat. The next DFlash-style drafter
deployment mounts build_v133+ and eats the +7..30% at the draft head for free;
the repack plane (and its 1.5 GiB ceiling) never allocates for MT<=8-only
processes -- one less side-stream sync at first draft too.

Memory-fit corollary for the serving drafter (untested, flag for the next
drafter deployment): the packed walk cuts the per-token weight footprint 4x,
which for heads whose packed set fits the card cache turns the row's regime
from streaming to cache-fit -- the +15% fit-regime arm is the preview. Q18b
(Wc=4 middle ground for MT>=8) is now also the Q12c candidate: an MT=16 arm
that keeps the packed walk but recovers the wide-tile -2% (e.g. expand into
preallocated regs with the fold shape the wide tile likes); gate on beating
BOTH at M=7/14.


## Q19 count-only compiler-trap audit: the donor traps do not transfer (again) — and the census tool needed its own audit (2026-10-25, window 12)

Scope: every shipped gfx1100 kernel in build_v133. The device objects are
fat ELF binaries (`clang-offload-bundler` target
`hip-amdgcn-amd-amdhsa--gfx1100`, via the radiance-dev container; the host
objdump 21 cannot read the bundle); slices extracted for all 25 `.hip.o`
with zero skips, then parsed by the new `tools/trap_census.py`: 286 symbols,
960,703 instructions, per-kernel counts of the four donor trap families
plus bpermute and ds_swizzle attribution.

Verdicts (all count-only, no kernel code touched, no card time):
1. true16 no-op pairs: ZERO. `v_lshrrev_b32` with shift-imm 16 appears
   nowhere in 960k instructions; `v_alignbyte_b32` nowhere. The d16-class
   traffic (5,858) is all by-design unpack (the Q12 u2 walk, the e4m3/e2m1
   decode paths). The donors' volatile-asm failure mode never existed here:
   `asm volatile` occurs zero times in src/ (both surviving dot2c sites are
   plain asm).
2. Guards inside unrolled K steps: NEGLIGIBLE. Branch instructions within 8
   of a `v_wmma_*`: 143 static across the tree's 1,100 WMMA sites, max 4 per
   kernel — the DBUF page-boundary class, not the per-WMMA exec-mask branch
   the donor fixed.
3. Natural-order LDS row + dword rename: 67 tree-wide, worst 6 (gdn conv).
   Not a cost center on any serving row.
4. Unmerged adjacent `ds_load_b64` pairs (same base, offsets 8 B apart):
   ZERO tree-wide. Q10's widening already removed the class it would find.
5. dead-`__shared__` (the Q10 trap, source sweep): zero declarations without
   uses tree-wide.

The finding that was NOT on the donor list: the tree still pays 6,090
`ds_bpermute_b32`, and the default-path hotspot is the w2a8 q8 GEMV's
kWave=32 butterflies (468 static at <MT=4,W=16>, identical across both C4
legs since the reduction is shared by both input walks). Q6's DPP
`row_xmask` fold was applied only to the mxfp4 q8x row. Spawned as Q22.
Attribution cross-checks: the mxfp4 q8x kernels' 96/48/24 are their
R3D_Q8X_DPP=0 fallback arms compiled in the same function (expected, off at
runtime); every butterfly site is wave-local per the closed TPR<=32 audit.

Bonus attribution for the DS-pipe ledger: the attention family feeds
column-major WMMA B fragments through `ds_swizzle_b32` (13,339 tree-wide;
h256 arms top at 404 per static kernel) — rocwmma's gfx11 lowering, NOT
ds_read. Any future ds-pipe traffic note should count swizzles, not just
loads.

Tooling lesson — the census tool committed the very sin Q19 audits for:
silent zeros. Three RDNA format mismatches, each found only because a
number refused to add up between two independent counts (awk vs python):
(a) GCN mnemonics match nothing on RDNA3: LDS traffic is `ds_load_*`/
    `ds_store_*` (not `ds_read_*`), bpermute is `ds_bpermute_b32` (exact
    match on the stem still works, on `ds_read` it silently died), movs
    carry `_e32/_e64` suffixes.
(b) The container objdump prints UPPERCASE hex in the address comment
    (`// 000000017E24:`) — a `[0-9a-f]+:` regex silently drops every
    instruction whose address contains a-f: measured loss 6.8x
    (140,408 parsed of 960,703 real). First two census tables retracted.
(c) `v_lshrrev_b32` takes the shift imm FIRST (`v, 16, v`) — a
    tail-imm-only regex would "prove" the true16 census zero without ever
    looking.
Countermeasure baked into the tool: a coverage guard — loose-format line
count vs strict parse, refuse to print any table below 95% coverage. The
empty-slice law generalized: a census that reports zero is a hypothesis, a
census whose parse coverage is unproven is noise.

Card state: none spent this window (count-only per Q19's own scoping);
spec9g untouched on build_v129 (health 200 verified at window open).

---

## Window 13 (2026-10-25, evening) -- Q22: DPP butterfly folds on the w2a8 row, SHIPPED

The Q19 census's live finding closed the same day it was filed. Q6's
`row_xmask` primitive (DPP_CTRL 0x160|mask, ISA Table 94) now folds every
w2a8 butterfly: both octet folds (f32 sub[m] in the u2 walk, i32 isub[m] in
the shared q8_dword_iter) and both whole-wave tails (xor16 stays shfl, it
crosses the 16-lane row). Lever R3D_W2_DPP, default ON, literal "0"
withdraws; stubs mirrored for non-gfx11 builds.

Probe-first law earned a corollary: __builtin_amdgcn_update_dpp requires
the control word as a COMPILE-TIME constant -- a runtime mask argument fails
the build ('argument must be a constant integer'), which is why the shipped
helpers template the mask. q22_probe: 4.19e9 lane-fold ops, both arms of all
four fold sequences (f32/i32 x octet/tail), zero mismatches -> bit-identity
is a source property before any card gate ran.

Build-law re-earnings (both cheap only because the gates exist):
(a) An insertion anchor that matches the kernel SIGNATURE but not the
    TEMPLATE HEADER above it lands helpers between `template <...>` and
    `__global__` -- the kernel loses its template, the helpers lose scope,
    and the error cascade names neither cause. Single-TU compile (card-free,
    4 min) is the right catcher; the -j1 full build wasted 22 min on it.
(b) The container image keeps its llvm tools OFF PATH
    (/opt/rocm/lib/llvm/bin): clang-offload-bundler and llvm-objcopy are
    'command not found' there; carve_dis.py needs the PATH prefix. The
    bundler in this ROCm wants --type=archive -unbundle --input/--output.

Gates on build_v134 (SHIPPING artifact; spec9g not swapped -- zero w2a8
lines in its whole log, grep-verified): repro 13/13 exact on four arms
(default q8c+dpp / R3D_W2_DPP=0 / R3D_W2_C4=0 / kill-switch u2) with the
default arm md5-identical to the withdrawal arm; battery 8/8 legs, all
seven standing packs difflines=0 vs v133, plus a full-pack leg on the
withdrawal lever (also 0). Census: v_add_*_dpp present in all 28 q8
instantiations (208 at the <MT4,W16> hotspot); static bperm unchanged by
design -- the runtime-lever shape compiles both chains and branches
uniformly per wave (Q6 precedent, no new law).

A/B same binary, 7 shapes x 5 rounds (two shapes got +2 rounds to chase
outliers): every median positive, +1.5% (M=1) to +8.4% (M=14), the fold
pricing better on wider tiles (more folds per dword fetched). Five shapes
strictly non-overlapping; M=7 non-overlapping once its one noisy sample is
trimmed; M=2's spread stays contaminated by two A-side outliers -- +2.9%
median kept, labelled the softest adjudication of the set rather than
quietly dropping the samples.

Card state: spec9g stopped for the gates + A/B (~50 min), restarted clean:
health 200, boot-asserted gfx1100 rank 0 / 6.65 GiB, coherent chat, still
zero w2a8 lines. Runner artifacts on cat: q22_probe.cu, q22_gates.sh,
q22_ab.sh, q22_ab.txt (144 rows), build_v134.sh.

## Window 14 (2026-10-25, night) -- Q20: the GDN pressure pass, fences transfer, SHIPPED default-ON (build_v136)

Q17-delta2's toolbox finally transferred where its trap list never did: the
sched_barrier live-range cap is the real cure for our st[128]-resident kernels.
Baseline (fresh v134 census, loop-localized -- the severity dimension v2 missed):
all four recurrent arms AND chunk_scan pin at 256 VGPR with vgpr_spill
320/222/189/98(/73), and the scratch ops are INSIDE the hot loops (in-loop
scratch 309/342/374/110/91 of the static totals): every decode token paid the
delta-rule chain through scratch memory. Root cause: the scheduler batches the
128-way unrolled d-chains' LDS/global loads for latency hiding; with the state
array permanently resident, the in-flight window is the marginal pressure and
the allocator pays it in scratch.

The lever: `R3D_GDN_SB` (`42ab8a8` arms, `ead595f` promote), fences
every 16 d via `__builtin_amdgcn_sched_barrier(0)` on chunk_scan + both
recurrent forms (token loops AND the ANCHOR fold loops); slot SB=3 additionally
fuses the token loop's rank-1 add into the q readout (same per-d bracketings,
same d-ascending orders -- bit-identical by construction). The loops MUST stay
fully unrolled (a dynamic `st[d]` index would demote the state array to scratch
-- the Q18 trap, now armed against in both directions).

Census after (`build_v135` measurement build): the four fenced recurrent arms
carry ZERO scratch instructions at 167-187 VGPR (69-89 below ceiling -- the
headroom Q11-class ideas were gated behind); instruction counts DROP (e.g.
conv<1>: 14125 -> 13564). chunk_scan is the exception: fence spills MORE
(189 -> 224) -- its pressure is the resident state itself, no window to cap, so
chunk_scan ships STOCK and its fence stays a dark lever.
SB=0 arms are instruction-for-instruction IDENTICAL to v134 (4/4, after
retiring two false alarms: the disassembler's trailing `...` padding line and
the template-mangling name map; ILb0ELi0E -> ILb0E).

New harness `tools/bench_gdn.cpp` (the first GDN decode bench): decode shapes
(slen 1, anchor fold nacc=2, live quantiser pair), ROTATING DISJOINT slot
windows per launch (aliased windows across sequences RACE -- the fingerprint
wobbles; SLOTS=64 with bases under 16 is the collision-free layout), per-launch
events, medians/p25/p75, and an o+state FNV fingerprint per case.

Kernel A/B (same binary v135, env-selected arms, 3 interleaved rounds x 200
launches x 6 cases, ~/scratch/r3d/q20_ab.txt):
  rec_pc_n1 +11.8%  rec_pc_n8 +17.3/+18.9%  rec_an_n1 +31.0/+31.2%
  rec_an_n8 +44.6/+44.8%  fus_an_n1 +26.1/+24.1%  fus_an_n8 +50.1/+51.3%
medians separated by multiples of the spreads (66 vs 120 us etc.); sb3 (fusion)
is indistinguishable from sb1 -> ship the fences, keep fusion dark.
Fingerprint: ONE hash per case across ALL nine runs (3 rounds x 3 arms, ~1000
launches/case) -- the bit-identical claim measured, not asserted.

Promotion gates on build_v136: battery 12/12 (7 standing packs fingerprint
ZERO-diff vs v134 -- the packs contain the gdn ops, so the shipped arms pass
the oracle where v134 passed; 3 gdn op legs on the kbench oracle default=fenced;
2 withdrawal legs R3D_GDN_SB=0 with ZERO-line report diff vs the shipped legs).
Serve: this lever is the FIRST on the live serving path (spec9g boots 24 GDN
layers: kv_gdn_state/kv_gdn_conv 48 MiB/2.25 MiB per layer). Serve A/B (bench9b,
9 reps/arm + trailing A-confirm 14.5-16.3 == A's cluster, no drift): decode
+2.6% median (15.2->15.6), prefill flat (chunk_scan unchanged -- consistent by
construction), the kernel-level A/B is load-bearing per the Q1/Q18 class.
spec9g RECREATED on build_v136 (API-exact config, only the radiance_home bind
changed), rollback spec9g_v129_rollback prepended to the chain.

Laws earned: (1) the docker ENGINE API create body IS the container Config
(+nested HostConfig), name via query param -- the {Name,Config,HostConfig}
swarm shape 400s with "config cannot be empty"; and bodyless POST /start fails
SILENTLY under curl -s -- give it -d '{}'. (2) `ssh ... setsid nohup docker run`
survives ssh-teardown kills -- the "container restarted itself" scare was
cmake's 3-min HIP configure eating the wall clock. (3) The donor's fence
transfers iff the kernel HAS a schedulable load window: recurrent rows yes,
chunk_scan no (resident-state pressure) -- pressure triage before cures.

## Window 15 (2026-10-25, late night) -- Q23: the attention pressure pass; the fence does NOT transfer (NULL, lever dark)

Triage first, per the window-14 law, and it paid before any card time. The v2
spill census over build_v136 splits the attention family into three stories:
(1) the h256 MT=3 DBUF arms -- the donor cliff rows (88/138 vsp, 324/268 priv,
256-pinned) -- still spill at HEAD, scratch traffic 100% in-loop; (2) the
kernel2<128,MT> staging arms sit pinned at 256 with 2-5 vsp and 192-224 B
private; (3) the h256 MT=3 NON-DBUF arms at 249/242, zero spill, near-cliff.
Placement analysis (`scratch_loc.py`, in-loop by decoded backedges, context by
neighborhood): the h256 DBUF spills cluster around global staging traffic
(32-43 of 84-96 near global loads, 9-12 near WMMA) -- the window signature the
GDN cure rode; the kernel2 spills ride the WMMA chains with ZERO global context
-- resident accumulator pressure, chunk_scan class, no cure built (law, not
measurement debt).

Dispatch audit before promising anything: the h256 row is issued only behind
R3D_HD256 (the operator's open drawer item); spec9g serves head_dim 128 (the
64 KiB/layer/block page arithmetic pins it). So the fenceable arms are a
not-yet-live row -- the fence lands as PREPARED evidence for the hd256
acceptance decision, lever R3D_HD256_SB with UNSET = the historical unfenced
arm (no silent codegen change anywhere), and the fenceable arms are a
dead-today row.

The build: SB template dimension on kernel1 (stub list mirrored per the fold
postmortem -- the mangled-name trap is written in the source itself), three
fence sites after each in-flight `ld_kv_regs`, `r3d_attn_sched_stop` helper,
DBG launch-proof extended with the `sb=` field.

THE NULL, measured two independent ways: the fenced PAD=1 arm carries metadata
identical to stock (138 vsp / 256 vgpr -- not one register freed), and its
instruction stream is identical to stock modulo register renaming (11,174
instructions both, zero mnemonic deltas). No `s_sched_barrier` exists anywhere
in the object -- `__builtin_amdgcn_sched_barrier(0)` is a scheduling
CONSTRAINT consumed before register allocation, never an emitted instruction
(GDN's fenced arm: also zero barrier insns, yet its allocation moved). The
root cause is structural: the DBUF in-flight windows in this kernel are fenced
ALREADY by their designed `__syncthreads`/s_barriers -- the scheduler has no
freedom across them to take away -- and the real pressure is the resident
`oa[D/16]` accumulators (128 VGPR at h256) plus fragment staging, 138 spilled
registers against a 16-register in-flight window. A live-range cap cannot
touch spills larger than the window it caps.

Laws earned: (1) a fence transfers iff the window is UNFENCED -- census the
EXISTING barriers inside the candidate window before building; (2) the cheap
pre-screen: spilled-register count vs in-flight window size (138 vs 16 here)
-- window caps are bounded by the window; (3) the window-7 heuristic
"spill-traffic context names the pressure kind" MISLED on h256 (global-ctx
spills, resident cause): context placement is a hypothesis, only a built arm
or the window arithmetic adjudicates. (4) Retracted mid-window: a hipcc-flag
quick-compile object diffed against cmake-built objects measures the FLAGS,
not the edit (the dual-issue deltas were -mllvm-pipeline noise) -- object
identity gates compare like toolchains with like.

Gates on build_v137 (cmake-like toolchain, the only comparison that counts --
see law 4): stream identity `tools/sym_stream_diff.py` (kept in-tree: it
normalizes the mangled-name drift the 8th template arg adds, asserts parse
coverage, compares operands not just mnemonics) says 31/31 common symbols
BYTE-IDENTICAL streams, zero differing, one addition (the fenced
<256,3,0,1,0,1,0,1> arm). Census: 240 kernels = 239 + fenced; fenced arm's
metadata is stock's 138 vsp / 256 vgpr exactly -- the fence freed nothing,
confirming the resident-cause attribution at the allocator. Card battery
(q23_gates.sh, 11 legs vs the v136 reports): B1-B6 + attn bf16 + four f8st
legs all green, every pack difflines=0 including st1/st2/st3-vs-st0 (Q10's
LUT identity survives the template-dimension surgery); the kernels pack's
22-line delta is the harness's own skip-note prose (it prints the machine's
free-VRAM budget, 1.51 vs 1.50 GiB between runs) -- environmental text, zero
fingerprint lines touched. spec9g was stopped for the card-exclusive battery
and restarted (health 200, still build_v136 -- v137 changes nothing a live
serve issues: unset lever = historical arms, and the arms it does issue are
stream-identical).

## Window 16 (2026-10-25, late night) -- Q18b/Q12c: the wide-tile register window, SHIPPED default-ON at MT=16 (build_v139)

The census trap that started this row finally paid for itself. Since Q18, the
MT>=8 q8 GEMV arms kept the pre-Q18 walk verbatim: the window is preloaded with a
static index but CONSUMED through `for (d = 0; d < rem; ++d) ... bwin[d]`, and a
dynamic array index is precisely what LLVM cannot register-allocate, so at MT=16
the whole window lives in per-thread scratch (144 B private, 9 scratch sites/iter
at W=8, 17 at W=16) on the row's hottest verify tile. Q18 had measured the
obvious cure -- the full W-fold -- as a loss there (-8.1% M=7, -11.4% M=14: MT*W
body copies price the tile past the traffic saved) and left Q18b to find a
middle ground that beats BOTH schedules.

Two middle grounds were built as one new template dimension on the q8 kernel
(`WC`; 0 = verbatim, >0 = sub-window of that depth with both loops unrolled, <0 =
select-window: the runtime consumption loop survives with ONE body copy and reads
the operand through `w2_static_sel`, a static-index select chain). All three walk
the dword indices in the shipped order, feed `q8_dword_iter` the same values, and
keep `rem` block-uniform, so bit-identity is a source property again -- and the
`WC = 0` default is the SAME specialization as the shipped `<MT,W,C4>` template,
so nothing about the existing arms moved.

**The census called the A/B before the card saw it** (`tools/wc_census.py`, card
free, old-vs-new `.lto.s` per arm): WC=2 removes the scratch at MT=16 with the
VGPR count UNCHANGED (122 -> 122) for a 1.46x body; WC=4 removes it but busts the
`launch_bounds(256,2)` 128-VGPR cap at MT=16 (132 VGPR + 46 SGPR spills); and the
select-window -- the elegant one, one body copy -- is WORSE than the disease
(160 B private, 23 scratch sites vs the 144 B / 9 it replaces), because holding
all W `uint4` operands live across the body is exactly the pressure the window
was meant to relieve. The A/B then reproduced all three verdicts (+3.5% / +1.7% /
-60.0% at M=14 cold, all spreads disjoint). A census that predicts the measurement
is worth more than a measurement that only confirms it.

**Q12's MT=16 gate is RE-ADJUDICATED, and that is the real finding.** Q12 gated
the packed-u2 walk out of the wide tile because it measured -1..-3% and attributed
it to the expansion VALU ("the same per-geometry law"). With `WC=2` the packed
plane wins the same tile by +9.9% cold / +15.4% IC-fit. The loss was never the
expansion's instruction count -- it was the expansion riding a scratch-demoted
walk. The packed gate therefore follows the WINDOW now, not the tile size
(`c4 = mt <= 8 || wc != 0`), and the ship is: MT=16 takes packed + depth-2
sub-window; MT=8 stays exactly as shipped (every WC arm measured parity there --
packed staging already keeps its small window in registers, so there is no scratch
to remove and no reason to perturb a shipped schedule on a coin flip).

Confirmation A/B on the shipped build (v139, same binary, `R3D_W2_WC=0`
withdrawal the only axis, 3 interleaved rounds, every spread disjoint --
~/scratch/r3d/q139_gates.log): withdrawal is slower at every MT=16 shape and both
regimes -- M=14 cold 2.551->2.257 ms (+11.5%), M=16 +10.7%, M=10 +14.2%, M=14
IC-fit +16.7%, M=16 IC-fit +14.3%. Gates: repro 13/13 exact on FIVE arms with the
per-tile dispatch line as liveness (`def` = `mt=16, q8c, wc=2`; `w0` = `mt=16, q8,
wc=0` == the v137 dispatch; `c40` = `q8, wc=2`; `kill` = u2), battery 7/7 with
ALL SEVEN fingerprint packs byte-identical to v137, and -- because the change is
additive -- the v138 pre-ship battery also ran 7/7 zero-diff, which is the
evidence that the WC arms are inert until selected.

Tools born this window: `tools/wc_census.py` (per-arm allocator + scratch census
and instruction-stream identity across a template-arg change) and
`tools/ab_parse.py` (A/B adjudicator: medians, spreads, and the DISJOINT verdict
the ledger's adjudication standard demands).

Laws and traps earned:
- A `.lto.s` kernel LABEL line carries a trailing `; @mangled` comment and kernel
  METADATA lines are indented TWO tabs: an anchored `^_Z\w+:$` regex parses zero
  kernels (the ledger's own empty-slice trap, again) and a `\t`-prefixed "instruction"
  filter silently ingests `.amdhsa_*` metadata. Match `^_Z[^:\s]+:` + allow the
  comment; keep only lines matching `\t[a-z_]`.
- Adding instantiations to a TU renumbers module-wide assembler labels
  (`.LBB18_7`->`.LBB24_7`, `.Lfunc_end18`->`.Lfunc_end24`) with every operand
  identical. Normalize the counter before any identity claim or you "find" 30
  differing kernels that are byte-identical code.
- A new template parameter MUST be mirrored in the non-RDMA3 stub -- and the stub's
  own comment said so. The mangled name carries the template-arg list; a stub that
  drifts registers a name the device image does not carry (invalid device function
  on every launch). Caught here by a card-free TU compile in ~60 s.
- `gate.sh` stages `$R3D_BUILD` (default `build13`, an ABI-22 relic). A gate script
  that forgets to export it stages the WRONG build and the ABI-26 harness refuses
  the whole library: seven instant FAIL legs and seven garbage reports. BUILD IMAGE
  = GATE IMAGE ABI, and the staged dir must be the build under test -- export it at
  the top of every gate script.
- `cmd | ssh 'cat > f && ... &'` writes ZERO bytes: the `&` backgrounds the whole
  remote chain and the piped stdin is lost -- and `bash -n` on an empty file says
  SYNTAX_OK (vacuously). Copy and verify byte counts in one step, launch detached
  (`setsid nohup`) in the next.
- Never send a runner's stdout/stderr to `/dev/null`: it hid an awk syntax error
  (`/ROW{...}` -- the pattern needs its closing slash) for 27 silently-lost bench
  runs. Card time is cheap; silent loss is not.
- The regime boundary on this card is the 96 MB INFINITY CACHE, not the 6 MB L2
  (`l2Bytes=6291456` from the device query). The "cache-fit" arm that earned
  +15.4% is IC-fit: 4x16 MB of packed plane resident while the repacked set
  (4x64 MB) streams. Label regimes against the IC on RDNA3.
- The vault's git credential helper failed once with a non-JSON unlock response and
  the push died on `could not read Username`; a plain retry succeeded. Transient
  vault failure is a retry case, not a credential problem -- assert length, retry
  once, and do not go hunting for a broken token.
- A launcher debug line guarded by `static bool once` proves one arm per PROCESS; a
  harness that sweeps M then proves only the FIRST tile's arm, and a lever that
  silently never dispatched would still look bit-exact. Print once per distinct
  (mt, walk, window) tuple instead -- host-side, debug-env only.

Serve: spec9g mounts build_v136 and has logged ZERO w2a8 lines ever (re-verified
this window), so no swap rode the promotion; the row ships for the next
DFlash-style drafter deployment on build_v139+, which collects +10..17% at the
verify tile and never allocates the repack plane at any tile. Card session ran
~25 min card-exclusive (spec9g stopped, restarted clean: gfx1100 asserted, health
200).


## Window 17 (2026-10-26, morning) -- Q17 unparked; Q21 A/B: the WMMA skinny GEMM TRANSFERS at MT>=8 (+46..+211%)

Operator released the Q17 drawer item ("unpark ... it may affect what next to
look at") and the release proved out exactly that way. Queue rows re-stamped
(Q17 released, Q21 parked->ready); the triage doc's remaining open datum --
"i8a8 WMMA rate on gfx1100 not measured by us" -- became a probe, and the probe
became the window.

`tools/probe_wmma.cpp`, one binary, four stages, card session on the XTX:
(1) RATES, liveness-gated per stage (per-stage memset + nonzero count -- the
first pass skipped it and the ledger's dead-row law caught a lie in flight):
WMMA i8 x s8 sustains 59,394 GMAC/s against the shipped dot4 chain's 17,933 --
3.31x, with f16/bf16 in the same 56-62k band (the donor's "iu8 no faster than
f16" holds engine-vs-engine on gfx1100 too, but the transfer comparison is
WMMA-vs-dot4, and the 241 GB/s APU numbers were never the point). (2) The
ACCUMULATOR lane map pinned by manufacturing a coordinate fragment: bf16
A[i][k]=i*16+k, B=identity, dump fragment.x[] per lane -> lane l holds row
2k+(l>>4), col l&15; the donor's `2e+(l>>4)` claim verified on our card; the
i32 accumulator map is IDENTICAL. matrix_a row-major: row l&15, k = reg +
8*(l>>4). This datum was Q11's single blocker -- UNBLOCKED.
(3) THE ROCWMMA 8-BIT LANDMINE: u8 x u8 `mma_sync` compiles, loads run, and the
gemm EMITS NOTHING -- accumulator all-zero under an identity-trick gemm, rate
stage 1442x "faster than physics" (live=0/49152 once the liveness gate existed).
The int8 x int8 path is the live one. Any u8-on-u8 WMMA code on this stack is a
silent no-op; the two liveness stages that caught it (impossible rate, then
zero accumulator under identity) are the detection recipe. (4) B-fragment
contract probe under the skinny kernel's exact staging (col_major, token-major
LDS, ldm=136): 256/256 CONFIRMED.

Q21 itself (`tools/bench_skinny.cpp`, donor r35d_8a8_narrow_kernel pattern --
16-row weight tile as matrix_a, tokens as matrix_b, i32 accumulate, per-128-
group epilogue verbatim so the i32 group sums are order-free exact and the
float tail is bit-identical by construction): SKINNY==NAIVE oracle everywhere;
against the shipped ROW one element in 229,376 differs by 1 bf16 ulp, same
element across shapes -- hypothesis: fma contraction in the library build's
epilogue (NOT yet proven; the integration gate must test it, e.g. -ffp-contract
=off repro arm). Timing (alternating, same process, same planes, IC-fit + cold):

    M=1   -47.5% / -41.4%   (GEMV owns single-token decode)
    M=4    -3.3%  (overlapping spreads: tie)
    M=7   +46.6%  (separated)
    M=14 +208.3% / +211.2%  (separated in both regimes)

The pre-adjudication read it right directionally (the row at M=14 runs at 10.6%
of even the dot4 peak -- staging-bound, and fragments cut LDS bytes per MAC 4x
and remove the cross-lane flush entirely: the accumulator registers ARE the
outputs). Two laws this window: a fragment engine's win can live in the STAGE,
not the MAC ceiling (the M-scaling of GEMV was the xs re-stage, not dot4
throughput); and wave-wide staging loops must be written per-wave -- the first
build gated ws staging on tid<128 (waves 0..3) indexed by the writer's wave id,
leaving every tile quarter-staged and LDS[4..7] uninitialized: near-miss
outputs from stale LDS, the dead-row trap wearing an LDS disguise. A misread of
my own probe output ("lane pairs") sent the first fix down a false path; the
bmap probe, not the panic, found the real bug.

VERDICT: Q21 closed as WIN-with-M-gate; integration (lever R3D_W2_SK, dispatch
MT>=8 -> skinny, repro/battery/withdrawal) booked as Q21b -- it is a kernel
family on a row spec9g never issues, so promotion needs no serve swap.
Serve restored at close: health 200, gfx1100 boot-asserted, on build_v136.

---

## 2026-10-26 window 18 -- Q21b: the WMMA skinny arm ships default-ON (build_v140)

The 1-ulp datum died first, and it died with a control. CMakeLists carries
-ffp-contract=off as a per-unit CORRECTNESS flag (the rad_plugin.h oracle forbids
contraction) -- including on r3d_gemm_w2a8.hip -- while tools/bench_skinny.cpp was
compiled with plain hipcc (contract=fast default). Rebuilding the bench both ways
and running both in-process: fast reproduces the datum exactly (229375/229376, the
same element, 1 bf16 ulp), off is 229376/229376 at M=14 and 114688/114688 at M=7.
Bit-identity of the port is therefore a library-build property, inherited, not
hoped for. LAW: a dlopen bench compiled under different FMA contraction than the
library fabricates ulp drift and misattributes it to the kernel -- benches inherit
the project's correctness flags.

The port: gemm_nt_w2a8_gemv_sk_kernel is the bench kernel made library-faithful
(all six pitches, u4zp nibble zero-points, mpad SUM rows, the shipped epilogue
shape verbatim), 16-row i8 fragment tile per wave, tokens as matrix_b columns,
per-wave staging (the window-17 dead-row lesson kept in the comment), dead waves
EXIT rather than clamp (bar.sync excludes exited wavefronts). Lever R3D_W2_SK
default ON, dispatch gate MT>=8 && M<=16; =0 withdraws to the verbatim v139
dispatch; the kill-switch withdraws it too (sibling-row law); the arm reads the
engine's packed u2 plane and skips the repack cache fill entirely -- the c40
repro leg proves the independence (narrow tiles repack, wide tiles still sk).
gate.sh learned to forward R3D_W2_SK alongside its C4/DPP siblings.

Gates (card session ~7 min, spec9g stopped, operator-granted exclusivity):
repro 13/13 exact on four arms with per-tile liveness lines (def: mt1/2/4 q8c,
mt8/16 sk; sk0: the full legacy ladder; kill: u2; c40: q8 narrow + sk wide).
Battery 7/7 packs zero-diff vs v139 -- the new default is output-identical --
and the SK=0 full-pack withdrawal leg zero-diff vs v139 as well. Library census
on the linked .so: sk body 1212 instructions, 8x v_wmma_i32_16x16x16_i8 (the u8
silent-no-op landmine was NOT stepped on), zero scratch.

A/B on v140, same binary, R3D_W2_SK the only axis, 3 interleaved rounds,
q18_bw2 ROW with full non-zero liveness: M=14 +145.6%, M=16 +165.7%, M=10
+101.5%, M=7 +8.8%, M=14@8k +28.9% -- every spread disjoint (the 8k M=14 spread
is the loosest, 0.403..0.515 vs 0.545..0.556, still separated). The M=7 gap to
the bench's +46.6% is regime, not contradiction: this rot=4 plane set is
4x64 MB >> the 96 MB Infinity Cache (cold, bandwidth-bound), the bench's M=7 row
was rot=1 cache-fit (compute overlap wins there). Serve untouched: spec9g never
issues w2a8; the next drafter deployment mounts v140+ and the verify band
(7-token block pass included) runs ~1.5x on the same weights.

Parsing traps earned (the census family again, all silent-failure class):
- substring-probing mangled names needs a namespace-unique anchor: "sk_kernel"
  is a substring of "mask_kernel" -- the first census hit the wrong code object
  AND the wrong label, and only the asserts caught it (they fired on numbers,
  loudly, which is the whole point of a coverage guard).
- llvm-objdump prints instruction addresses in the TRAILING COMMENT on this
  stack; a `^\s+[0-9a-f]+:` instruction regex matches nothing (the window-12
  uppercase-hex trap's sibling).
- awk: a file path that falls inside the program string becomes a regex
  division -- silent garbage, no error. Quote programs, pass files as arguments.

VERDICT: Q21b closed SHIPPED default-ON, v140 is the drafter-mount build.
Serve restored at close: health 200, gfx1100 rank 0 boot-asserted, coherent
(one-sentence answer, 767 reasoning tokens -- the window-16 probe law respected).

---

## 2026-10-26 window 19 -- Q11 branch 1: fragment-side softmax ships default-ON (build_v142)

The pinned accumulator lane map (window 17) was the key; the door it opened is
the shipped attention body's biggest structural idle cost: both passes paid a
full score-tile round trip through LDS (store S, barrier, read the row back
scalar-per-column) while the data sat in the fragments. The pinned map says a
row's 32 score columns live in ONE 16-lane half-wave (even rows lanes 0..15,
odd 16..31; col c in a0.x[k], col 16+c in a1.x[k], same lane and register k of
row 2k+(l>>4)) -- so the whole-row reduction is a 4-step butterfly whose XORs
1/2/4/8 never cross a 16-lane row: it rides the exact DPP row_xmask primitive
Q6/Q22 shipped. Pass one folds max and exp-sum on the registers with a
designated lane (c==0) owning the (m,l) update; pass two forms P on the
fragments straight into Psp. Ssp never exists on this arm: fewer instructions
(9545 vs 9911 at hd=128), and the l sum is the only thing that changes tree
(max is order-free, so m and every P bit match the sweep arm by construction).

**The hf bug and why only the sweep caught it.** First implementation used
`hf = tid >> 4` -- but the map's lane index l is WITHIN THE WAVE (tid & 31),
so waves 1-3 computed row ownership with hf=2..7 and misowned rows. Every
small-geometry harness passed: the bogus designated lanes' rows overrun
`valid` and get gated, so waves 1-3 silently never wrote -- the dead-row trap
in its third costume (loop, LDS staging, and now lane-partitioned ownership).
The oracle sweep (attn_oracle_sweep.hip, built exactly because "the shipped
pack is head_dim-256-only" -- the window's vacuous-pass trap) was the ONLY
harness reaching geometries with qlen>64: 3 FAILs with rel_l2=-nan and
thousands of outliers. The bisect was two one-line kills + incremental cmake:
pass-1-killed still failed, pass-2-killed still failed -- both arms shared the
broken hf. Fix: `(tid >> 4) & 1`. LAW: a wave-local index must be written
`(tid & 31)`-masked when the block is multi-wave; and a correctness harness
whose every geometry fits in wave 0 / one tile cannot see whole-block bugs --
coverage means ROW count (qlen>64), not just shape variety.

**Gates (v141, lever R3D_ATT_FS dark):** repro prefill 0/16 differ and splitk
bit-identical (hashes EQUAL incl. the REPRO_SPL=4 host-merge) both arms;
oracle sweep PASS and ARM-IDENTICAL post-fix; the seven standing battery packs
zero-diff vs v140 -- AND ALSO under R3D_ATT_FS=1: the battery is BLIND to the
FS bands (its packs land on non-DBUF/hd256-gated rows; attn_bench's header had
recorded exactly this blindness). Two runner self-catches: the first battery
run used an unsynced gate.sh (no R3D_ATT_FS forwarding -> the FS=1 legs
measured the default -- zero-diff was TRUE but MEANINGLESS; the second lesson
being that zero-diff is only evidence when the test can SEE the arm); the
final-log awk had a quote-layer-infected pattern (grep "^geom" became
"\"^geom\"" through three layers -- silent 0-match, the ledger's own awk law
again, 4th repetition). A/B on the fixed artifact, 3 rounds interleaved, same
binary, lever the only axis: 5/5 geometries DISJOINT -- hd128 c1024 +35.6%,
c2048 +54.2%, c4096 +38.3%, c8192 +33.5%, hd64 c4096 +46.6%. Output hashes
differ across arms exactly where l drift is expected, and the sweep's
tolerance evidence bounds that drift.

**Promotion (v142):** default flip is HOST-ONLY -- `.hip_fatbin` of v142 is
byte-identical to the gated v141 (objcopy+cmp), the device image is the tested
one verbatim; three standing packs re-zero-diff with FS now default. Serve
A/B (spec9g 3072-token probes, v136 vs v142, two rounds each + trailing
confirm): TTFT flat (17.28-17.36 s all four rounds, +-0.4%), decode
median-to-median inside the per-eval spread (19.2-24.6 tok/s scatter) --
attention is ~1-3% of a step at this ctx on this model, the same regime law
that kept Q20 at +2.6% serve despite +12..+51% kernels. The kernel-level win
IS the delivered artifact; a longer-ctx serve probe (>12k) is the booked
follow-up if the prefill share is ever questioned. spec9g now lives on
build_v142; spec9g_v136_rollback prepended; health 200, gfx1100 boot-asserted,
coherent (162 reasoning + 7 content tokens, "The capital of Western Australia
is Perth.").

Q11's SECOND branch (VGPR group flushes, Astra-P2) remains open: the flush-
free accumulator-epilogue datum from bench_skinny (lane reg == output element)
applies to the row-tile GEMM shapes, and the attention h256/kernel2 256-VGPR
arms are the flush-flavored pressure -- booked as Q11b.

VERDICT: Q11 branch 1 closed SHIPPED default-ON (build_v142), the eighth
promotion of the arc: fragment-side softmax +33.5..+54.2% on every bench
geometry, arm-identical oracle sweep, host-only default flip over a byte-
identical device image, serve neutral-to-flat with zero regressions.

## 2026-10-26 — window 20: Q11b CLOSED NULL census-first; ChatGPT pass-3 triage; attention-family artifact census

Inbound state: main moved under the session to `f772417` — PR #17 from the
review lane adjudicated ChatGPT pass-3 as Q25-Q28 (all parked) and fixed a
stale DRAFT header on r3d_gemm_w2a8.hip (comment-only, no codegen). Triage
against the integrity record: no conflicts — Q25 explicitly quarantines Q7's
q8x depth-2 NULL from transferring to the attention family (correct, same
conclusion this card family's Q23 reached structurally), Q26 is a genuine
dtype cross Q21 does not cover, Q27/Q28 are ABI-gated. Nothing re-opened.

Q11b was adjudicated entirely with card-free artifact evidence, per the Q23
law (census before card, existing barriers before fences, spill-count vs
window-size pre-screen). The census: every gfx1100 code object carved from
the SHIPPED build_v142 libr3d.so (25 objects), per-kernel allocator fields
from `llvm-readelf --notes` and instruction classes from `llvm-objdump -d`
body slices — 61 attn instantiations, 721,262 instructions parsed, zero
metadata misses (tools live on cat as ~/scratch/r3d/q11b_carve.py).

Serving-arm facts (spec9g env verified R3D_*-free, so the default dispatch is
`k1FS1(128,3,0,1,0,0,0,0,1)`, confirmed against the launcher and container
env): 225 VGPR, ZERO spills, no scratch, 192 DPP ops (the Q11 butterflies are
visible in the shipped artifact: FS0 DBUF carries 160, FS1 carries 192, the
+32 being the 4-step row_xmask folds x two passes x sites). Occupancy:
2 workgroups/CU, and 2 is a VGPR HARD CAP — 3x128x225 = 86,400 register
lanes exceeds the 64K budget regardless of LDS — so any carve-shrink idea
(including the register-P fantasy of freeing Ssp+Psp) cannot raise
occupancy. Q23's family split reproduced exactly: kernel2<128,*> arms still
256-VGPR with scratch ops riding the WMMA chains (resident-accumulator
class, no cure built per Q23), h256 DBUF still 88/138-VGPR spill cliff
(drawer-gated, prepared fence evidence unchanged).

(a) Register-built P fragment: the Psp round trip is WAVE-LOCAL (wave w
writes rows w*16+2k+hf and reads back only its own rows — the producer IS
the consumer), so no cross-wave exchange would be needed... but the pinned
acc-map -> A-map permutation is a full 16x16 lane<->register transpose
(value P[r][c]: source lane (c&15)+16*(r&1) register r>>1 -> target lane
(r&15)+16*(c>>3) register c&7): >= 4 cross-lane butterfly stages over 8
registers (~32-64 DPP/select ops per tile). The shipped LDS path it would
replace measured FAR cheaper in the artifact: the compiler already vectorizes
the b16 round trip (ds_store_2addr_b32 sites; fragment loads come back as
ds_load_b64). ~4 stores + ~8 loads per lane per tile — and the b2 barrier
survives anyway because it gates V(t) visibility, not Psp. Strictly heavier:
NO BUILD. The ledger law: after Q11-b1 ate the big round trip, what remains
in Psp is the cheap half; a lane-permutation never beats a 4-store/8-load
LDS hop on this hardware.

(b) kernel2 direct-O epilogue: not dispatched (R3D_ATTN_V2 env-gated;
container env carries no R3D_* at all), and its pressure is the resident
accumulator (Q23 law: caps and tail-trims cannot touch spills bigger than
the window; 256-VGPR pinned with in-loop scratch). Evidence staged in this
census, nothing shipped, branch closed.

Bonus census rows for the drawer: k3 (v3 THIN, kBM3=32, env-gated dark) is
the cleanest arm in the family — 168 VGPR zero-spill, 24,880 B static LDS at
D=128 — it exists because someone already explored the 2-blocks/CU idea via
thin blocks; it stays dark until a dispatch decision, not until more
evidence.

SPAWNED Q29 (open, desk): the shipped FS=1 arms report EACH OTHER's Carve
totals in group_segment_fixed_size (h128-FS1: 29,776 = Carve<64>; h64-FS1:
42,064 = Carve<128>), verified by two independent parses; FS=0 reports
correctly; the fresh save-temps IR declares [42064 x i8] for the FS=1 h128
instantiation, so the divergence is backend/link-stage. Runtime correctness
is proven at scale, so this is metadata semantics, not a live bug — but it
bit this census mid-flight and the mechanism is now owed to the ledger.
[WINDOW 22 CORRECTION: every swap claim in this paragraph is a PARSE
artifact, not an artifact property — see the window-22 chronicle at the end
of this file; the vgpr/vsp/ssp numbers from this census stand.]

Tool/pitfall chronicle (all four fired before any number was trusted):
(1) `hipcc -S --save-temps` writes the DEVICE `.s` as LLVM IR, not GCN asm —
the save-temps stage files are compiler-stage artifacts; (2) the save-temps
device `.o` is BITCODE, llvm-objdump refuses it ("not a valid object file")
— the assembled truth is only in the fatbin'd host .o or the shipped .so;
(3) the window-12 LDS-mnemonic rename fired again (ds_store_*/ds_load_*,
not read/write — my first table showed a tree of zero LDS ops); (4) objdump
addresses live in the trailing COMMENT — the window-18 trap recurred in the
body-count regex (`^\t[a-z_]`, never the address prefix). The zero-parse
coverage asserts fired loudly on attempts 1-3 instead of shipping tables of
zeros — the pattern the ledger has now saved four times.

## 2026-10-26 — window 21: Q24 CLOSED split: w2a8 permlanex16 tail ships default-ON (v144), mxfp4 lever DARK on serve data

Resume item #1 (Q24 desk step 0) resolved the same day it was booked, and the
window became a full split-verdict promotion cycle on build_v143/v144.

**Desk step 0.** `__builtin_amdgcn_permlanex16` exists in ROCm 10.1 — with SIX
args (the fork carries FI and BOUND_CTRL as explicit bools; upstream has four,
the signature error text told me). Compile probe carved to gfx1100 ISA: ONE
`v_permlanex16_b32`, no DS round trip, and the donor's hazard law ("do not
trust LLVM to fix a V_CMPX adjacency") is OBSOLETE for intrinsics — LLVM's
SIInstrInfo models the permlane hazard itself. The encoding pinned from the
pinned PDF's p.432 pseudocode (fitz-extracted, `docs/isa/.venv`): the x16
reads the ALTERNATE row through 16 shared 4-bit laneSel entries; xor-16
identity = entry i == i.

**THE SILENT-WRONG-ARM FIND.** My first helper passed the identity pair with
the words swapped (0x33221100/0x77665544 — a nibble pattern I'd pattern-matched
from the ISA's rotate example instead of deriving). No build error. No crash.
No fault. 15/16 lanes silently gathered the wrong partner. The q24 card probe
caught it because it adjudicates THREE-WAY — `__shfl_xor(16)` vs an independent
LDS oracle vs permlanex16 — so the dump showed `ref==ora` everywhere and `got`
permuting by the nibble stream of the swapped pair. The entry field reads the
SGPR pair with the builtin's THIRD arg in the LOW word: identity is
{lo=0x76543210, hi=0xFEDCBA98}. LAW: for a permute whose sources are both
"obvious", the probe must carry a source-independent oracle — shfl-vs-permlane
alone would have compared two wrong things and passed. After the fix:
PROBE-BITEXACT at 2.68e8 lane-ops, fold-chain shapes (octet+tail) for f32 and
i32 at lever 0/1/2, 8 seeds.

**Levers (three-state, the R3D_GDN_SB precedent).** w2a8 `R3D_W2_DPP`: 0 =
verbatim shfl chain (the lever law's literal), 1 = the Q22 arm, 2 = +x16 tail.
SHIPPED DEFAULT 2. mxfp4 `R3D_Q8X_DPP`: same states, default stays 1 (see
serve data below), "2" opt-in. w2a8 kernel params went bool→int dpp; stub list
mirrored (the mangled-name law); the debug-liveness tuple grew the dpp axis
into a 64-bit bitmap (mt × walk × wc × dpp) so a leg that silently left the
lever at its default shows up as a never-printed level.

**w2a8 adjudication (same binary, lever the only axis).** M=2 DISJOINT +9.9%
median, M=4 +6.4% trimmed (one outlier round kept and labelled, not pruned),
M=1 +3.1% medians with overlapping spreads (softest — the dominant decode
shape is the honest soft win). The MT>=8 shapes I also benched (M=7/14/16)
DISPATCH THE sk ARM — which has no folds at all: their dpp=1-vs-2 rows were
no-ops, and the liveness line made that visible in the harness output before I
adjudicated them as noise. The zero-diff-visibility law in its A/B costume:
an A/B row can only adjudicate the arm the lever actually reaches.

**mxfp4 adjudication (serve, same binary, env the only axis).** Seven rounds
on spec9g's card, boot-liveness `q8x dot4 arm launched (mt=1 pf=1 dpp=2)`
proven per d2 round post-probe (the arm dispatches at FIRST DECODE, not at
boot — my first runner grepped liveness at readiness and reported NONE, a
false alarm of placement, not of code): base {22.10, 21.84, 22.04} + confirm
{21.28} vs d2 {22.01, 22.85, 20.04} medians t/s at ctx 3072, full overlap —
FLAT, and by Q6's non-overlap law the lever stays DARK. A one-instruction
swap per fold is below the serve noise floor where the fold is one bperm; the
w2a8 kernel-level effect (~3-10%) does not survive into serve decode share.

**Gates.** v143 (armed dark): repro 13/13 exact on dpp arms 0/1/2, per-arm
liveness printed every level; battery 11/11 zero-diff vs v141 (both lever-2
legs included — after catching that gate.sh never forwarded R3D_Q8X_DPP: the
mxfp4 half of my kernels_d2 leg would have been a vacuous pass; caught PRE-run
by re-reading the forwarder list, fixed in-tree). v144 (host-only default
flip): FATBIN-IDENTICAL (16,044,752-byte .hip_fatbin cmp v143 vs v144), repro
13/13 exact with DEFAULT-dispatch liveness dpp=2, census 423 permlanex16 sites
in the w2a8 q8 TU / 30 in mxfp4 q8x (bperm counts unchanged by design — the
withdrawal arms live in the same binary). spec9g never issues w2a8: no swap,
v144 is the drafter-mount build (Q18 precedent); serve restored on its v142
home, health 200, zero restarts, chat coherent.

**The 48/58 difflines that were nothing.** The first v144 default battery
looked dirty vs v141 (48/58 difflines, checked counts down 12/45) on a build
whose device image is byte-identical to a zero-diff build. A same-build control
run reproduced the SAME 4725/4-skipped twice, and the "Not covered" section
named it: dflash_select needs 727.51 MiB vs the harness 512 MiB free-VRAM
budget — deterministic per RUN (card state), not per build; the surviving diff
lines were the skipped cases' result rows + their derived not-reached entries.
Zero numeric diffs. LAW (sharpens window 9's budget-noise note): budget skips
are deterministic per card-state, so "deterministic" does not discriminate
code from state — re-run the same build as a control before believing a
coverage-count delta.

**Errors I made, on the record.** The swapped-selector constants (above) — the
near-miss of the window; a host/device variable-name slip (`dpp` vs `w2_dpp`
in the liveness block — the four-minute TU compile caught it before the full
build wasted 15 minutes… actually the full build caught it, because I launched
it without the TU gate; 6 minutes of compile to find a two-character error);
a vacuous `grep -o 'md5='` on repro outputs where the md5 law belongs to the
battery packs — the non-emptiness assert caught it, `exact=YES rel_l2=0.000e+00`
is the repro's real identity gate; an amend-push that diverged cat (fixed by
`git reset --hard origin/main`, cat keeps no local commits by law); `docker
rm -f` racing the daemon's cleanup into a create conflict — the runner failed
closed, the second template retries create; the serve A/B runner's liveness
grep at readiness instead of post-probe; and two ssh-quote/awk path slips that
garbled the census `total` field (the non-zero permlane/bperm asserts held).

**State.** main == GitHub == local == cat @ 6f1be03 (promotion) — all trees
clean. spec9g: v142 home (its own rollback slot chain: v136/v129/v124/v121),
health 200, gfx1100 rank 0, zero restarts. build_v143 = the fully gated
measurement build; build_v144 = the drafter-mount shipping artifact (permlanex16
tail default-ON for w2a8; nothing changes for spec9g's issued rows). Runners:
q24_gates.sh, q24_serve_ab.sh + q24_serve_ab2.sh (create-retry template),
q24_probe.hip (three-way probe).

---

## Window 22 (Q29 closed: the metadata "swap" was our parse — shipped artifact cleared)

Desk-only window, zero card time. The question owed to the ledger: do the
shipped FS=1 attention arms really report each other's static-LDS totals?
Answer: no — they never did. The window-20 census table was lying, and the
window now belongs to the parse-collision class.

**Native decode (association by structure).** The ROCm 10.1 notes are
MessagePack, not YAML: an `AMDGPU`/tp32 note per code object whose per-kernel
map contains `.name` and `.group_segment_fixed_size` together, so a correct
decode cannot mis-associate. Native read of the shipped v142 fatbin:
k1FS1(128,...)=42064, k1FS1(64,...)=29776 — both correct per the source's
Carve mirror; every h256 arm reads 0 (dynamic `extern __shared__`, exactly as
designed; the source's own static_assert keeps the launcher formula pinned).

**Second association (kernel descriptors).** Each kernel's `.kd` symbol blob
begins with the group-segment size (u32) and stores the kernel-code pointer as
the fn_va minus kd_va delta — verified present per kernel. First-u32 vs
msgpack gsf: 61/61 across both gfx1100 attn objects (29+32 kernels), and
h256's zeros match too. Two independent paths, zero disagreements: the
artifact is internally coherent.

**The culprit, reproduced.** Re-running the actual window-20 scratch script
(q11b_carve.py) on the same libr3d.so still prints the swap — so nothing
about the artifact changed or drifted. Mechanism: llvm-readelf renders the
metadata with `.group_segment_fixed_size` BEFORE `.name` in each kernel block
(legacy AMDHSA YAML field order), and the script attributes every field to
the last-seen `.name` anchor. Each kernel therefore inherits the NEXT
kernel's gsf. The collision is invisible exactly where adjacent kernels
share a value — which is precisely why the FS=0 pairs looked fine, why the
FS=1 h128/h64 adjacency was the only attn pair with differing neighbours,
and why the whole ldsB column is gsf-shifted (last h256 arm printed 42064
where truth is 0; k2(128,*,0) rows shifted). Post-`.name` fields
(vgpr/vsp/ssp/priv) were always anchored correctly: the occupancy ruling and
the donor-cliff spill numbers all stand, and the serving-arm re-read from
msgpack matches the census exactly (225 VGPR, zero spills). The two
"agreeing" parses shared the same anchor defect; agreement between parses
that read the SAME rendered dump is not independence.

**Repo audit.** spill_census.py splits per-kernel blocks delimited at
`.args:` so its gsf read is within-block (its window-7 numbers stand);
wc_census.py reads only post-`.name` fields from pre-assembly IR (safe); the
anchor pattern existed only in the scratch script, never in a committed tool.
The "FS-arm LDS numbers unreliable" hold is retracted — the numbers were
always the good ones.

**Retraction before truth (mine, on the record, twice this window).** My
first KD read applied the HSA ABI offsets (source_size u16 @0, gsf @4,
kernel_code @0x30) and printed 61 DISAGREEs with srcsz==gsf — garbage: this
ROCm's `.kd` blob is not the legacy 64-byte descriptor layout at the symbol
VA. Had I trusted the first table, I'd have "found" a second artifact bug.
The fix was to read the bytes before the ABI: dump the blob, see 0xa450
(=42064) as its first word, and see the fn delta inside it.

**Fatbin forensics recipe (now in the ledger body):** carve = raw scan for
`\x7fELF` with EM=224 in the objcopy'd `.hip_fatbin`; TRIM by max(PT_LOAD +
PT_NOTE filesz, e_shoff + shstrtab end) or the section table/note segments
get sliced; each TU ships as gfx1100 + gfx1101 + gfx11-generic triplets
(filter on `amdhsa.target`; the generic stubs report all-zero resource
fields); notes are msgpack with 4-padded owner names — walk nameships, decode
with a STREAMING unpacker (the padded blob ends in alignment zeros, which a
strict unpackb() rejects as ExtraData — the failure that initially swallowed
every big object, including the attn ones, and made a 12-object fatbin look
like an 11-object one without them).

**State.** main == GitHub == local == cat @ (this commit), docs-only. spec9g
untouched on its v142 home, health 200. No lever moved, no build cut. The
Q29 line in the queue carries the full closure text and the three new laws.

## Window 23 (Q25 closed: attention depth-2 prefetch — NO-BUILD on three grounds)

Q25's queue row demanded desk-first work and a v142 rebaseline. Both were done
for ~3.5 minutes of card time; the build never became interesting.

The premise was written pre-Q11. Rebaseline on the shipped v142 (q11_ab binary
against build_v142's libr3d.so, card-exclusive spec9g stop, 3 rounds/arm,
hashes stable per arm): deep-ctx 128/981/8192 = 215.4/216.8/216.0 GB/s on the
serving DBUF+FS1 arm, 135.2/133.7/134.1 with R3D_ATTN_NO_DBUF=1 (which also
falls off FS — the static assert pins FS to the DBUF schedule), h64 136.9 vs
76.6, small 32.0 vs 18.3. The ledger's "141 -> 171, ~2.9x of card left"
(COVERAGE:690) is what DBUF bought in the build8 era; Q11 branch 1 has since
carried the same geometry to ~217 (window 19 measured 217.6 — agreement across
weeks and card states is itself a datum: the metric is stable). Remaining head
to the ~500 GB/s card ceiling is 2.3x, and it is not prefetch-depth-shaped:

- register wall: the DBUF in-flight buffer is KNU uint4/lane per tensor
  (kBN*D/8/128; 8 at h128 = 32 VGPR, 4 at h64 = 16). Census: k1FS1(128) = 225
  of 256 VGPR — a second in-flight K array is 257, architecturally impossible.
- LDS wall: a second KVp buffer is +17,408 B at h128 / +9,216 at h64 — both
  cross 64 KiB/2 and drop to 1 WG/CU (the Q1 repad law).
- dispatch law: h64-FS1 (215 VGPR) WOULD absorb the +16, but no model on this
  host dispatches h64 — Q11b's no-undispatched-arms rule holds.

The objdump says the design already got the effective depth: the two staging
windows (K(t+1) at loop top, V(t+1) mid-loop) are 8x global_load_b128 each,
full-drain gated at the consumption store and in flight ACROSS the mma phase —
per-tensor depth is 1 but in-flight is 2 tiles/iteration, 256 B/lane. Depth-2
per tensor would demand 4 tiles and the registers do not exist.

Three self-inflicted disassembly-matcher traps, all silent-zeros caught by
non-zero-hit asserts (the Q19 census discipline working as designed): the wait
mnemonic is TWO tokens — `s_waitcnt lgkmcnt(N)` — so matching token 0 alone
("s_wait") or "lgkmcnt" against the first token both matched nothing; LDS ops
here are ds_store/ds_load (Q20's rename law, third sighting, mine again); and
llvm-objdump prints the 16-BIT SIGNED s_cbranch offsets UNSIGNED (65448 is
-88) — the first script found "zero backward branches" in a loop kernel, which
is never a fact about kernels and always a fact about parsers.

Recipe kept: tools/spill_census.py runs against any build_vNNN tree directly
(it wants CMakeFiles objects, not the .so); carve_dis.py + a per-symbol slice
gives per-kernel disassembly from the shipped .so.

---

## 2026-10-26 — window 24: Q26 CLOSED measured (skinny owns fp8a8 M>=4); issue #18 filed (MT-ladder row truncation)

Q26 asked whether Q21's WMMA skinny win transfers across the dtype boundary to
fp8a8 (e4m3 x e4m3, 128-of-K block scales). It transfers, the crossover sits
LOWER than w2a8, and measuring it honestly surfaced a shipped-contract bug.

**The bench** (`tools/bench_fp8_skinny.cpp`, window-17/18 harness laws baked
in: project `-ffp-contract=off`, regime labels, same-process alternating A/B):
arms are the two dlopen'd shipped entries (plain tile, decode GEMV), an
ascending-f32 independent oracle, and the skinny prototype — 16-row weight
tiles per wave as matrix_a, tokens as matrix_b, bf16 fragments filled from a
per-block e4m3->bf16 LUT (the exact-embedding property Q9 pinned: e4m3 needs
<=4 significant bits, every code is an exact bf16), f32 accumulators, per-
128-K-group flush applying `(ps * sa[m]) * sb[n]` in the oracle's order — the
m64 tile's own flush semantics moved into registers. Because this family is
tol-graded (3e-02, order-opaque dot legs — issue #6 law), bitwise equality is
NOT the bar: rel_l2-vs-oracle is, applied symmetrically to the shipped arms.

**Verdict table** (K=N=16384; rot=4 cold / rot=1 cache-fit; medians of 25;
all deltas same-process alternating):

    M      GEMV(ms)  TILE(ms)  SKINNY(ms)  SKINNY vs incumbent
    1      0.654     (alias)   1.267       GEMV owns (-48%)
    2      1.048     (alias)   1.277       GEMV owns (-18%)
    3      1.038     (alias)   1.319       GEMV owns (-21%)
    4..7   1.87(MT4) (alias)   1.33        +41..45% BOTH regimes
    8      4.04(MT8) (alias)   1.326       +205%
    9..16  --        2.45-2.5  1.33-1.41   +75..87%
    (alias = f8_launch_tile routes M<=8 to the GEMV for BOTH entries — see
     the label-liveness finding below; real tile data exists only for M>=9)

No interior crossover: SKINNY is FLAT ~1.33 ms across 4..16 (B-stream bound
~200 GB/s, matching the attention row's rebaselined 216), GEMV grows per-MT
ladder step (MT4 1.87, MT8 4.04 — the xs[MT][128] f32 re-stage is the
per-token tax Q21 predicted, doubled here by the per-element decode), and the
64-row tile is flat 2.45 because it stages B for rows it never uses more of.
Correctness: SKINNY vs oracle rel_l2 3.3e-05 (the shipped tile itself shows
3.4e-05 on the same fixture); SKINNY-vs-TILE 1-ulp agreement 229323/229376;
the deterministic UNIT mode (row-class codes 1.0/2.0/3.0, scales 1.0) passes
229376/229376 ANALYTIC — every output element at its hand-computed value.

**Issue #18 (shipped contract bug, evidence-backed).** The decode band picks
`mt = M>=8?8 : M>=4?4 : M>=2?2 : 1` — rounding DOWN — while the kernel
computes `live = min(M, MT)` rows and the entries accept every M <= 8. M in
{3,5,6,7} therefore silently never writes Y rows past the ladder: M=7 wrote
4/7 rows (rel_l2 0.655, exact sqrt(dropped/M) signature), M=3 2/3, M=5 4/5,
M=6 4/6; M=1,2,4,8 clean. The flip side matters for every future reading of
GEMV timings: any M=5..7 measurement is MT=4-work timing, not service time —
mine included, and the bench found it because GEMV@M=7 timed equal to GEMV@M=4.
Suggested fix in the issue: round the ladder UP (kernel guards make MT > M
safe by construction) + battery rows at odd M. Q26b must land the same fix.

**Laws earned this window**

- **A vector-store idiom transfers only after re-deriving COVERAGE, not
  width.** The first skinny build copied the w2a8 bench's per-lane uint4 LDS
  store — in w2a8 a packed dword EXPANDS to 16 codes so a uint4 fills 16 LDS
  slots; in fp8a8 a dword DECODES to 4 codes, so the same store covered 4
  slots while claiming 16: every tile row staged only its first 32 of 128
  slots AND each store wrote 8 bytes of adjacent stack garbage over the next
  chunk. Unit mode localized it exactly (LDS row clean to index 31, garbage
  after, untouched beyond). This is the dead-row family's fifth costume.
- **A tol-graded family needs a deterministic unit mode before any timing
  claim.** Random fixtures cannot separate "slightly wrong" from
  "tolerance-class different"; row-class codes through a LUT give an analytic
  expected value for EVERY element, and it was the only tool that could prove
  staging exactly. Keep it in the bench (F8SK_UNIT=1).
- **Liveness applies to bench labels, not just zero-diff packs.** The bench
  arm named TILE was the GEMV for every M<=8 — the band routing lives inside
  the shared launcher, below both entries. The claim "+215% vs TILE at M=8"
  was really "vs GEMV-MT8" and would have shipped as a tile verdict if M=8's
  4.18 ms had not sat absurdly above M=14's 2.45. Benches with band dispatch
  behind an entry must print what dispatched (R3D_DEBUG_GEMM) or assert the
  monotonicity their labels imply.
- **f32 debug channels carry values, never bits.** Packing two 16-bit LDS
  halves into one float slot (low | high<<16) exceeds float's 24-bit exact
  integer range AND the multiply-shift loses the second half: the first
  rowdump "found" ws[0][0]=0x0000 while the slot next to it read 1.0. Count
  and first-bad-index as small ints instead.
- **Debug slot maps collide unless centrally allocated** (second sighting,
  window 19's costume): lane16's fc dump at dbg[16..23] silently overwrote
  the row-check results at dbg[20..23]; the "garbage" row-check output was
  the fc dump. Allocate a slot table before writing any of it.

Errors on the record: the uint4 staging (above); `__shfl_down(0, v, off)`
argument order (width is not the first arg — the wave-reduce checks read
garbage until fixed, and the one "28/32" it printed after fixing is an
unresolved SCAFFOLD oddity, kept on the record: the analytic per-element gate
that supersedes it passed 229376/229376 and no shipped artifact is affected);
`std::span` under -std=c++17; the dbg-slot collision; the lossy bit-packing;
and reading my own mislabeled A-rows as tile data for one full sweep before
the non-monotonicity caught me.

State: no kernel touched, no build cut, no lever moved; v144 remains the
drafter-mount artifact; spec9g never touched (card used ~55 min under the
operator's card grant, XTX never left idle-duty state — spec9g container
untouched, GPU1 shared per grant). tools/bench_fp8_skinny.cpp committed.
Q26 closed measured -> Q26b ready (port with issue-18 fix); drawer unchanged.

## Window 25 — Q26b shipped (fp8a8 skinny band) + issue #18 fixed

Ported the bench's skinny row into `r3d_gemm_fp8a8.hip` behind `R3D_F8_SK`
(default-ON 4<=M<=16 && K%128==0; literal 0 withdraws to pre-Q26 routing;
`R3D_NO_FP8_DEC` family kill withdraws too). Three deliberate deviations from
the bench: the m64 pitch/rblk contract (BOTH scale grids — `F8SK_RBLK=128`
added to the bench so the checkpoint grid is gated, not assumed), no dbg
scaffolding, and dead waves participate in the barriers (`alive` guard) —
the bench's early `return` at tile>=N never fired on its all-alive grid,
but the threads it kills own slices of the block-shared xs staging; at
ragged N it would have left holes. Issue #18 (ladder rounds DOWN, rows past
the ladder silently never written) fixed unconditionally in the same commit:
withdrawal returns the old routing, never the old bug.

Honest cost on record: the corrected withdrawal GEMV at M=5..7 is slower
than v144 ever measured there (M=7 2.9->4.1 ms) because it now computes the
rows it used to drop — every historical GEMV timing at odd M in this row was
broken-work timing. The skinny band replaces it (+207% at M=7 vs the
CORRECT incumbent).

Gates. Unit: 229376/229376 analytic at M=14; the M=16 leg's 229376/262144
is the bench's EXPECTS formula not honoring the fixture's zeroed rows 14/15
(the real-fixture M=16 legs pass) — second scaffold oddity beside the
persistent wavemma_ok=28 suspect; artifact gates are the analytic + oracle
arms. Correctness: 16 legs, both grids, both arms, worst rel_l2 4.7e-05,
all verdicts ok, NaN-free. Battery 14/14: withdrawal 7/7 zero-diff vs v143
(reachability stated: pack fp8a8 rows are M=4/8/16 only); default 5/7
zero-diff, kernels+fp8 drift ONLY in-band rows (incl. gated composition),
verdicts hold, 536/536 + 2011/2011. Census: sk arm 159 VGPR / 0 spill /
0 priv / LDS 39680 exactly equal to the hand-computed LUT+xs+ws — a free
source-layout audit (my head-math had halved ws; the compiler disagreed and
the compiler is the artifact). A/B cold, same binary, liveness every leg:
+46.2/+207.2/+207.1/+79.7/+75.2% at M=4/7/8/14/16, 5/5 DISJOINT spreads.

Runner post-mortem (both dead sections were harness, not card): (1) `$r_$1`
inside a script running `set -u` expands variable `r_` — bash errors, SKIPS
the docker+grep pipeline segments, and the empty-input pipeline TAIL still
echoes liveness-shaped lines: the A/B section "ran" 30 times in zero seconds
and looked superficially alive until the timestamps collided on one second.
LAW: assert each harness section's expected ROW COUNT against its
expected-leg count before adjudicating; a section that ran empty is
indistinguishable from a section whose legs died, unless you count. (2) the
rebuilt script silently dropped its body (`+ body` missing from a header
concat) and both background runs "completed" exit-0 having done only the
header — same law, same guard: count, or diff the script against intent.
(3) battery report stems ARE filenames: passing one stem to all seven legs
collides them onto one file (q18_batt's per-leg suffixes existed for a
reason).

State: build_v145 shipped default-ON; v144 superseded as the drafter-mount
build. spec9g never issues fp8a8 (zero log lines, re-verified), restored
after the card session: health 200, gfx1100 boot-asserted, 6.65 GiB,
restarts=0, Perth-coherent. Residual booked: M 17..64 tile-band crossover —
the skinny row needs a grid.y token-slab dimension before it can even serve
M>16 (xs is 16 rows), so the tile band is untouched by design this window.

## Window 26 — Q26c measured: skinny wins the whole M 17..64 band (port justified, booked)

The booked job was the Q26b residual: does the skinny row beat the m64 tile
past M=16, and is the port worth a token-slab dimension. It is — measured,
bench-only, nothing shipped this window.

Design choice over the handover's sketch: the handover asked about a grid.y
slab dimension; that re-walks the weight plane ceil(M/16) times — the tile's
own staging law, inverted against us. The committed extension (6b1127f) puts
the slab loop INSIDE the K-group loop instead: ws stages once per group,
16-token slabs stream through xs, accumulators become y[4][8]. One weight
pass (same as the tile), LDS unchanged, and M<=16 reduces to the exact v145
arm — same order, same barriers, so window 25's numbers stay comparable.

Gates, all green (q26c_run.log):
- Unit regression M=14 IDENTICAL to the v145 baseline — wavemma_ok=28 is not
  a regression: it is the fixture's deliberately-zeroed padding rows 14/15
  making lanes 14/15/30/31 fail a nonzero expectation, and it reads 28 in
  every v145 log too. The probe itself is now gated (zero-fill only when
  M<=14, where those rows are actually dead), which also fixes the old
  u16-formula artifact for good. New slab path at M=32/64: wavemma_ok=32,
  ROWCHECK clean, ANALYTIC exact 524288/524288 and 1048576/1048576.
- Correctness vs oracle, both scale grids, real fixtures: rel_l2
  3.6e-05..4.4e-05 — the shipped tile's own class on the same fixtures
  (it reads 4.1e-05), the tol-graded family bar is 3e-02.
- Timing (spec9g stopped per the exclusivity protocol, restored after:
  health 200, RestartCount 0): 10 legs, M 17/24/32/48/64 x rot 4/1, 30
  reps, tile-dispatch liveness per leg. EVERY spread disjoint, both
  regimes: +45.9/+47.1 (17), +44.8/+44.8 (24), +43.3/+43.9 (32),
  +33.5/+34.0 (48), +14.1/+14.1 (64).

Datum retired: the "tile flat 2.45 ms" extrapolation only held in the M<=16
padding regime. Past 16 the tile climbs 2.62->3.49 ms while skinny walks
1.76->3.06; the win shrinks monotonically (slabs add token staging; the
tile's padding waste shrinks with M) but stays disjoint at M=64, where the
tile pays zero padding at all — so skinny's edge is instruction path and
staging width, not padding avoidance alone. That is the stronger claim and
the reason M>16 is worth the port instead of a skinny/tile crossover.

Booked next (Q26c port): same slab loop into the shipped sk kernel, band
4<=M<=16 -> 4<=M<=64 (tile keeps M>64), stub mirror, full ceremony — and
one new battery expectation to state up front: standing packs at M<=16 must
stay ZERO-diff (the M<=16 arm is unchanged by construction), while any
battery rows past 16 drift in-band like window 25's did.

State: local == GitHub == cat at 6b1127f, trees clean; spec9g restored
mid-run by the runner itself (health 200 in 20 s). Card session ~10 min
under the operator grant. No lever moved, no build cut; v145 remains the
drafter-mount artifact. Drawer unchanged: hd256 accept, RDNA3.5 guard.

## Window 29 (2026-10-26) — Q26c blocker: the tautological store guard — fixed at the desk, v148 gate-green

The v146/v147 battery fp8 leg (14 deterministic FAILs + 62 guard flags) fell
to desk work with zero card time, once the right discriminators were ordered.

Root cause (b4d2fdb): 9666f99's SLABS==1 store replaced the Q26b predicate
`tok < live` (live = min(M,16)) with `tok < 16` — a TAUTOLOGY (tok = lane&15).
The commit's lemma "under the 4<=M<=16 launcher gate, tok<16 == tok<live"
holds only at M==16. At M 4..15 the kernel zero-wrote rows M..15 past the
M-row Y: the flush guard (tok < live_s) kept those accumulators 0.0f and xs
is zero-staged, so every stray byte was a ZERO. The sk band precedes the gemv
band, so decode-shape M 4..8 rode sk<1> too (v145 was safe only because its
guard was the real predicate).

Why every probe missed it (LAW): a zero-fill OOB into zero-filled probe
padding is invisible to pad-region checks — probing for zeros with zeros. The
w28 arena sweeps (32 decode cells, gated-mode hits identical across builds)
were blind by construction. kbench's GUARDED allocator saw it immediately and
named the exact shape: 48 decode 'y'-past-end + 12 'b_scale'-adjacent + 2 'a'
= the 62 flags; the stomped neighbours were other cases' inputs, which is why
14 FAILs cascaded on arms untouched since v145. The fused gated row hid it
twice: the 64-row scratch arena absorbs rows 4..15 and the epilogue reads only
rows < M (gated-only passed 56/56; M<=16 fused dumps were byte-equal).

Booked discriminator outcome: paired battery dumps (dv145 vs dumps2, diff via
dumpdiff.py, card-free) exonerated the fused path — all fused pre.bins EQ, all
M<=16 post EQ, M=32/64 post differ on codes+product only within the tol-graded
family. That redirected from y_pitch suspicion to the kbench OOB table, and one
line of desk arithmetic on the guard closed it.

## Window 30 (2026-10-26) — v148 PROMOTION CEREMONY: v148 replaces v145 as the drafter-mount artifact

All five booked gates green on build_v148 (fresh, radiance-dev:v1.11.0),
runner cat:~/scratch/r3d/q26c_v148_promote.sh (log v148_promote.log), card
exclusive (spec9g stopped by the runner, restored at close).

1. Full 7-pack battery, canonical leg defs: 7/7 OK. fp8 536/536 with 0 wrote
   outside guarded allocations (v147: 14 FAIL / 62 outside); stock 2011/2011
   (24 budget skips), full 4737/4737, both 0-outside.
2. Stream gate (q26c_stream.py): 264 non-sk symbols BYTE-IDENTICAL vs v145;
   sk<1..4> present. sk<1> is NOT asm-identical to v145: +16 instructions,
   mnemonic-histogram-immune check shows all of it is control-flow skeleton
   (the SLABS==1 arm keeps the slab-loop skeleton) and the v146→v148 delta is
   EXACTLY the guard encoding (cmpx+exec-restore → cmp_e64+s_and_saveexec, +2
   for the live computation). RETRACTED expectation "restored guard returns
   the Q26b stream exactly": semantic restoration is proven by the battery and
   bench CHECKs, not by asm identity.
3. Fingerprint rows vs v145: attr/df2/df3/full/attn_bf zero drift; kernels 48
   and fp8 36 difflines are 100% fp8a8 rows at M 18/32/36/64 (the newly
   skinny-routed band), verdicts ok, rel_l2 <= 2.35e-05 vs bars 5e-03/3e-02.
   Every M<=16 row byte-stable.
4. Liveness A/B (v146 datum re-adjudicated at one geometry): M=32, 3 rounds x
   {rot4,rot1}, 6/6 spreads disjoint, t_med 1.947..1.987 (sk) vs 2.776..2.806
   (tile), +39.7..+43.1%; slabs=2 liveness line every sk leg, withdrawal live
   shape identical to the v146 window's.
5. Census from the shipped fatbin (q26c_notes.py; self-verified — v145 re-reads
   exactly 159/0/0 VGPR/spill/priv, LDS 39680 = the Q26b authoritative table):
   sk<1..4> = 152/162/178/178 VGPR, ALL zero spills, priv 0/0/128/0, LDS 39680
   uniform. sk<3> carries 128 B private segment already in v146 (SLABS>=2
   streams untouched by the fix) — flagged for the next allocator change.

Correctness legs at the fix band: M=8/15 slabs=1, M=32 slabs=2 — ROW==oracle
== SKINNY rel_l2 4.5e-05..5.4e-05, cross-slab bitwise probe 524288/524288.

VERDICT: v148 = the drafter-mount build; v145 superseded (v146/v147 never
promoted). spec9g rides v142-era unchanged and never issues fp8a8. Serve
restored and verified: tokenize 200, gfx1100 boot-asserted (PCI 17:00.0),
RestartCount 0, chat probe finish=stop.

State: local == GitHub == cat, trees clean. Drawer unchanged: hd256 accept,
RDNA3.5 guard.


## Window 31 (Q31 CLOSED desk — WGP store-law census of the shipped artifact + toolchain compliance check; queue hygiene)

Desk-only, zero card time. Two booked items from the operator's queue review:
the Q31 census and the stale Q26c queue row.

Q31 part 1 (census): the shipped v148 gfx1100 code objects are not on disk as
images — they ride inside libr3d.so's `.hip_fatbin` (75 embedded ELFs = 25 TUs
x 3 ISA targets; roc-obj-extract is deprecated and refuses the .so). Repo tool
`tools/q31_barrier_census.py` carves the section, filters per-kernel
`.target`, disassembles with llvm-objdump, and scans per function for pending
GLOBAL stores at every `s_barrier` (full drain = `s_waitcnt_vscnt null,0x0`).
Result on the live build_v148 artifact: 223 functions with barriers, 1945
barriers, ZERO outstanding global stores at any barrier. Two traps earned on
the way: (1) scratch_store must NOT count — the first pass flagged 17
functions and all of them were per-thread spill storms (sk<3>'s 32
`scratch_store` preamble from its 128 B private segment; thread-private,
lgkmcnt-counted, invisible to the law); (2) a census loop that appends its
row only at EOF reports one function per object — the co_28 barrier-count
mismatch (80 vs 96 raw) is what caught it. The sk kernel's real global stores
(Y epilogue) all sit AFTER the last barrier: kernel-exit stores, drained by
kernel-completion + stream order, no in-kernel consumer.

Q31 part 2 (toolchain law): src/ carries NO hand-written barrier asm — every
shipped barrier is LLVM output from `__syncthreads()`. The ROCm-6.x wait
generator complies with the WGP store law: 118 sites of
`s_waitcnt lgkmcnt(0); s_waitcnt_vscnt null, 0x0; s_barrier; buffer_gl0_inv`,
with the drain emitted exactly where a global store is pending (canonical:
quant_act_fp8 `global_store_b32; ...; vscnt null,0; s_barrier; gl0_inv`) and
omitted where none is — i.e. it is a real dataflow decision, not unconditional
padding. The barrier lowering itself (dscnt/lgkmcnt + s_barrier + gl0_inv)
carries NO vscnt wait: a hand-written or intrinsic-level publish MUST carry
the explicit triple; the compiler cannot see through it. The C++ reference's
inheritance assumption (Q31's open question) HOLDS for this toolchain.
Gate implication: nothing standing blocks Q27/Q28/Q14-class fusion on this
law.

Queue hygiene: Q26c row brought forward (was "port booked" while v148 was
already promoted windows 27-30 — status now CLOSED shipped with the
tautology/zero-fill chronicle); header date 2026-10-26.

State: local == GitHub == cat, trees clean. Q30 (card session) is now the ONLY
ready item in the queue.

## Window 37 (2026-10-27) — issue #11 milestone 3: hd256 split-KV decode arm (R3D_HD256, code + gates + bench)

The two booked items the operator greenlit together: build milestone 3, then the
teacher-forced KLD pair as the hd256 accept-evidence. This window closes the build;
the KLD pair rides at the end.

Code (commit 4ea299e, local == GitHub == cat). The decode row's split-KV family
gains head_dim 256 behind the SAME lever the m2 tile uses:
- body.inc: the m2 placeholder assert (`D != 256 || SPLITK==0`) becomes
  `D != 256 || FOLD==0 || SPLITK==1` -- the SPLITK=1 paths were already D-generic
  by construction (integer slice cut; partial store reuses the proven 32-row O
  staging through the Q+S alias, 33792 B fits the 42240 B span at 256).
- decode entry: gate `d256 = lever && hd==256 && !v2 && bf16-KV`; arena hook claims
  256 partials ONLY while the lever is set (an unset lever must not cost KV budget);
  the slice arm is a new R3D_DK_L256 macro carrying the m2 dynamic-LDS law into the
  decode row (attribute raised per instantiation, r3d_attn256_smem(PAD) passed).
  Eight literal instantiations (DBUF x PAD x FOLD); FS stays un-instantiated (m2
  law). splits=1 or short arena fall back to the m2 tile through the existing path.
- tools/bench_attn_decode.hip: the Phase C decode bench forked with BENCH_HD env
  (default 128 -- every v148-era command reproduces verbatim).

Gates (cat, build_v149 fresh): lever-OFF 7-pack OK with fingerprint difflines=0 on
ALL SEVEN packs vs v148; lever-ON stock+attn replay OK; repro_attn256 tile arm
byte-stable (in-run hash diff vs v148 = 0). Split probe (repro_attn_splitk,
REPRO_HD=256): 28/28 arms bad_cells=0, nan_slots=0; device-vs-oracle band is flat
across SPL=1..64 (max 2.688e-3 vs the SPL=1 tile arm's own 2.656e-3 -- the band IS
the bf16 fragment class, splitting adds no error class); host re-merge is always
LOWER than the device arm (partials obey the contract exactly); SPL=64@ctx300 is
bit-identical to SPL=13 (inert-slice skip = exactly the same live-slice sum), and
fold/plain arms agree to the last digit. My adjudicator's first pass printed 14
FAILs -- both halves of that were the adjudicator, not the card: a noise-token
regex that matched the substring inside `bad_cells`, and a 2.5e-3 threshold
imported from the kbench prefill-tile gate that the known-good SPL=1 tile arm
itself exceeds on this double-precision oracle. Adjudicated on raw (a)/(b) values.
Spill census (card-free, build_v149 local): slice arms 167-179 vsp / 268-284 B priv
(same h256 resident-accumulator family as the m2 tile's 88-138, slightly heavier
for the partial store); the merge kernel is spill-free; v148 rows unchanged.

Bench (exclusive, pinned, 3 interleaved rounds, 243/243 arms clean, tile column
lands on the m2 recorded data -- 3364 vs 3363.7 us):
  M ctx | tile us | law us | speedup
  1 512 | 576 | 133 | 4.33    1 2048 | 1007 | 318 | 4.25    1 8192 | 3364 | 722 | 5.66
  4 512 | 571 | 143 | 4.01    4 2048 | 1011 | 340 | 4.05    4 8192 | 3380 | 758 | 5.40
  8 512 | 585 | 154 | 3.79    8 2048 | 1013 | 366 | 3.80    8 8192 | 3404 | 808 | 5.12
Effective KV bandwidth M1/8192: 9.9 -> 46.5 GB/s. The law default is within 20% of
the swept best everywhere (p48 wins only at 8192 -- a 256 wave-target retune is a
follow-up candidate, same shape as the 128-era finding, does not gate the merge).
Runner law honored again the hard way: the first sweep "completed" in 90 seconds --
`bash -c "$script" "$R"` binds $R to $0, every arm died on an unbound $1, and only
the row-count assertion (81/round) caught the empty run. Fixed with SWEEP_ROUND env.

KLD accept-evidence pair (teacher-forced, differential law, v149 serve on
qwen35-9b-mxfp4 = the hd256 serve geometry 16Q/4KV): see next entry.

KLD accept-evidence pair, window 37 continued (teacher-forced, differential law,
v149 serve on the hd256 model 16Q/4KV, spec3, 1281 positions):
- Boot arithmetic: flashnext's deploy took the host's third-serve headroom
  (avail 3.3 GB, serve wants ~4.5 with the engine's default 4.0 GiB
  --host-headroom-mib). The guard aborted clean; the session ran with the
  knob at 256 over a TEMPORARY 8 GB swapfile (removed after -- host back as
  found). PORT LAW (new): llama-swap silently owns 9009/9010 here; a serve
  whose bind fails into /dev/null still prints its card line, so the m2-era
  log-grep check cannot tell whose /health you are answering -- v149 flow
  binds 9013 with a pre-flight port guard and a post-boot identity check
  (health body must carry radiance's in_flight field).
- Differential ON vs OFF: mean 2.557e-2, p99 0.359, top-1 0.969 -- same FAIL
  class as m2's tile-only measurement (9.79e-3, 0.979); the exact value is
  not cross-session comparable (each session scores its own fresh free-run
  trajectory, and the drift is tail-dominated: prose components sit at the
  floor -- prose-long 0.0018 -- while math/code carry 0.04-0.08 means).
  Nothing about it is decode-split attributable: FORCED SCORING IS PREFILL-
  ONLY (the slice arm is never invoked by the score pass; slice launch-proof
  = 0 with tile = 1 is the instrument's design, not a defect -- the proof
  lines are static-once per process, so tile=1 is the expected positive).
- Decode-split serve-level attribution is therefore structurally unavailable
  from this instrument, and the kernel-level numerics ARE the attribution:
  flat band across SPL=1..64, fold==plain, SPL=64==SPL=13 bit-exact on the
  inert-slice path, and the host re-merge strictly below the device arm.
  The accept decision for the hd256 FAMILY (fragment-rounding class, ~240x
  floor, established m2) remains the operator drawer item it always was --
  milestone 3 does not move it and this run did not need to.
- Session arms verified at the serve level: OFF ran ZERO 256 WMMA paths
  (tile=0 slice=0), ON ran the tile, gfx1100 adoption asserted per boot,
  spec9g restored healthy after the window (qwen35, in_flight 0).
