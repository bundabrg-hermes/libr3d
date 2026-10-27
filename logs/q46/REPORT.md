# Q46 adjudication report — host-side split-count policy + merge-kernel U-ahead batching
(subagent card window, 2026-10-27, manager/subagent mode. Build: **v154**. Branch: `card/q46-split-policy` on cat @ commit d2bcd6c (+tools/logs follow-on). Serve closed: spec9g on build_v153, health 200.)

## VERDICT: **PROMOTE-MERGE-ONLY**
- Merge kernel change (bit-identity class): **PASS** — battery legs M and W zero-diff vs v153 standing packs across all 7 packs; kernel-level identity matrix 32/32 cells byte-identical; spill census byte-identical.
- Split policy (precision class): tolerance grading **PASSES** (all drift within bars, confined to attention rows) but the serve decode A/B shows **0/3 non-overlapping positive rounds** (rounds overlap at every comparison) → the mechanical PROMOTE-BOTH gate fails → policy stays behind `R3D_SPLIT_POLICY=0` (both arms live in the binary; no default flip).
- Bonus datum (recorded, see §7b/§8): the wave rule at the PHYSICAL CU divisor (96) wins disjoint at h256 geometry ctx≥1024 — a retune direction for a future row, not this bundle's gate.

## 0. Scope and verdict rubric (from the brief)
(1) merge-walk bound + U-ahead loads = bit-identity class (fingerprint packs ZERO-diff vs v153). (2) wave-efficiency split policy behind `R3D_SPLIT_POLICY` = precision-visible (split counts change → partial-combine order): battery tolerance grading + decode hash sweep; drift must be confined to attention partial-combine rows or withdraw. (3) p48 anomaly datum recorded under the new policy. PROMOTE-BOTH iff merge zero-diff AND policy tolerance-pass AND serve decode ≥2/3 disjoint positive rounds.

## 1. Premise re-verification (line evidence, at d4c0a3d)
- Merge kernel `attn_splitk_merge_kernel` at src/r3d_attn_paged_decode.hip:156; walked ALL splits unconditionally (max loop :181-184, combine loop :191-197), inert skip AFTER the m load (`if (m == -INFINITY) continue;` :193). ✓ premise.
- Inert-sentinel law :28-29. ✓
- Split formula `decode_splits()` :214-225 (fixed closed form `(fold?128:192)/units` + caps), shared by arena hook (:277/:279/:293/:295) and launcher (:364/:371); `decode_partial_bytes(...)` :282 as cited. ✓ (premise "splits from arena bytes + fixed formula" = this closed form).
- p48 anomaly: COVERAGE window 37 (h256 bench, ctx 8192, ~20%). ✓
- CU count measured on-card: `rocminfo -a` (ROCR_VISIBLE_DEVICES=1) → gfx1100 "Compute Unit: **96**" (physical). The **HIP runtime answer** — queried exactly the way the policy queries it — is **48** (`hipGetDeviceProperties: multiProcessorCount=48, blocksPerCU=64, warpSize=32`). RDNA3 HIP counts 16-lane SIMD pairs. The policy takes the device's runtime answer; `R3D_SPLIT_CU` overrides; sweep A/B's BOTH divisors (cu48 = binary default, cu96 = physical fill). The brief's "0.8 of 256 compute units (XTX: 32 CUs)" matches neither measured count; the queue row's rule (0.8 × device CU count) is what was implemented.

### 1a. CRITICAL premise finding: the donor's n_act PREFIX walk bound is UNSOUND here
The Strata qsa merge_v2 donor assumes contiguous fixed-size chunks (live partials = prefix [0,ceil(width/slice))). Our shipped slice cut is the **floor cut** (`r3d_attn_wmma_body.inc` :445-456: `a=t0+nt*sp/spl; b=t0+nt*(sp+1)/spl; empty ⇔ b≤a`). Under the floor cut live splits are NOT a prefix: nt=3,spl=8 → live={2,4,6} (split 0 itself inert; split spl-1 ALWAYS live). A prefix-bounded walk would silently DROP live splits. And for nt≥spl every split is live — no inert tail to bound. **The "bound the walk at n_act" half of part (1) is dark against this law and was NOT shipped.** The sound half was: the donor's kMergeU=16 batching with partial loads hoisted ahead of the skip-select. (A prefix bound would first require changing the cut to contiguous chunks — that reorders partials = part-(2) precision class, not bit-identity.)

## 2. Implementation (branch card/q46-split-policy, commit d2bcd6c)
- **Merge kernel**: two passes (max, then combine) over batches of `kMergeU=16` splits; per batch, loads (m; in pass 2 also l and O) issue UNCONDITIONALLY into register arrays ahead of any select; only the ACCUMULATE gates on `m != -INFINITY`. Accumulate set and order byte-for-byte v153's (ascending sp, live-only; inert contributes nothing) → bit-identity is a construction property. Garbage O rows behind inert sentinels land in dead registers. Behavioral delta = arena READ traffic only, and only where inert splits exist (ctx < splits·tiles).
- **Policy** `decode_splits`: arm 0 = v153 closed form verbatim; arm 1 = FA2 rule per the row: first s with `units·s ≥ 0.8·CU` short-circuits (units alone ≥ fill ⇒ s=1 ⇒ no split arm); else smallest s ≤ smax with `wave_eff(s) ≥ 0.85·best`, `wave_eff(s)=blocks/(CU·ceil(blocks/CU))`. Caps identical (tiles_kv, tiles_kv/2 plain, **64** → arena law structural, one law function for launcher + arena hook). CU from `hipGetDeviceProperties.multiProcessorCount` once per process; `R3D_SPLIT_CU` override; fallback 48 = the device's live answer. `R3D_SPLIT_POLICY_DEFAULT=0` as built — default flip is the manager's act.
- **Liveness**: `DBG attn-decode split-policy arm=wave|fixed cu=... units=... s=...` under `R3D_DEBUG_DECODE`; `R3D_SPLIT_POLICY` + `R3D_SPLIT_CU` forwarded by gate.sh (w43 phantom-pass law).
- **tools/**: `q46_policy_check.cpp` (host differential: 1408-row grid, arm0 ≡ v153 exactly, arm1 inside the old envelope, fails=0 — log `~/scratch/r3d/q46_policy_check.log`); `bench_attn_decode.hip` parity mode gained FNV-1a `yhash=` over raw output bytes; runners `q46_id.sh`, `q46_sweep.sh`, `q46_sweep2.sh` (contain radiance-dev, pin `ROCR_VISIBLE_DEVICES=1`, HARD-assert `1 device(s) [0:gfx1100]` per the gate.sh phantom law), `~/scratch/r3d/q46_serve_ab.py` (serve matrix), `q46_spec0_probe.py`, `q46_serve_verify.py` (serve liveness).

## 3. Card/serve state at open→close
Open: spec9g up (4h, build_v153 mount, restarts 0, /health 200); XTX free of squatters; radiance-flashnext on the RDNA4 pair untouched; `spec9g_v151_rollback` never started. Build v154 in radiance-dev:v1.11.0 (gate image = build image), BUILD_RC=0.
CLOSE (verified): q46_* containers removed, `docker start spec9g` → **/health 200**, RestartCount=0, bind `build_v153/radiance_home` (md5 inside container `ad0ac9fb…` == build tree). Test build v155test deleted.

## 4. Spill census (v153 vs v154) — PASS
`diff ~/scratch/r3d/q46_census_v153.txt ~/scratch/r3d/q46_census_v154.txt` → **empty** (280 gfx1100 kernels, 41 spilling/private, same near-ceiling set, same LDS budget). The merge kernel appears in no spill/near-ceiling row in either build: kMergeU=16 batching introduced **no spill, no near-ceiling movement**.

## 5. Battery (final binary, solo runner — first parallel attempt voided, see D4)
Legs × 7 packs (kernels, full, attr, df2, df3, fp8, attn_bf): **M** default env; **W** `R3D_SPLIT_POLICY=0`+debug (withdrawal arm); **P** `R3D_SPLIT_POLICY=1`+debug. Log `cat:~/scratch/r3d/q46_batt2.log`.
- **M: 7/7 OK, ZERO-DIFF** vs v153 packs (merge patch rides the default path; attn_bf's h256 decode q_len 1..8 rows ride split+merge → the packs are NOT blind to the merge kernel).
- **W: 7/7 OK, ZERO-DIFF**; liveness `arm=fixed cu=48` lines present in leg logs (3).
- **P: 7/7 OK** (kbench gate = tolerance grader: all 4737 rows incl. 1689 attn rows within tol); liveness `arm=wave cu=48` ✓. Moved packs: ONLY `full` and `attn_bf`, 16 difflines / 8 attn-touching each; kernels/attr/df2/df3/fp8 ZERO-DIFF → **drift confined to attention rows ✓**.
- Drift rows (all 4 = h256 decode, in both packs), rel before→after vs bar 2e-02, status ok→ok: q_len1 2.25e-03→2.15e-03; q_len2 1.94e-03→1.97e-03; q_len4 2.28e-03→1.93e-03; q_len8 1.96e-03→2.10e-03. Same-magnitude reorder drift, ~10× bar margin. (P-leg battery ran cu48 = binary default.)

## 6. Decode identity + hash sweep (final binary)
- **rsk merge-identity matrix: all 32 cells ZERO-DIFF** (hd {128,256} × spl {1,2,4,8,16,32,48,64} × ctx {300,4096}; v153 vs v154 byte-identical, incl. ctx=300 = the interleaved-inert floor-cut class of §1a; and policy0-vs-policy1 equal under forced splits = force-precedence proof). Log `cat:libr3d/q46logs/q46_id.log`.
- **Natural-split parity sweep** (h128 32Q/8KV, ctx 512..8192): **v154/p0 yhash == v153 yhash at ALL five widths** (f8a92421…, 53c8ea0f…, 45945a81…, c25641dd…, fcd8c143… — bit-stable rows did not move at all). v154/p1(cu48): spl 16→5, rel_l2 drift ≤ +0.0001 (max 0.002221), all PASS the 0.01 band. h256 8192: p0 spl=64 rel .001989 / wave-cu48 spl=19 rel .002079 / forced-48 spl=48 rel .002016 (f48 yhash identical across arms — same splits, same order ✓) — all PASS.

## 7. Kernel-level A/B, h128 32Q/8KV (exclusive card, one binary, alternating arms, 3 rounds, liveness per leg ✓)
med_us (fixed s=16 | cu48 s=5 | cu96 s=10):
| ctx | p0 r1/r2/r3 | cu48 | cu96 | overlap? |
|---|---|---|---|---|
| 1024 | 113.1/113.3/113.3 | 171.7/172.1/171.8 | 132.1/134.9/132.2 | DISJOINT — p0 wins |
| 4096 | 223.4/223.5/223.9 | 435.1/435.7/435.5 | 280.6/280.2/280.5 | DISJOINT — p0 wins |
| 8192 | 368.0/368.5/368.6 | 734.0/733.0/735.6 | 482.8/483.6/483.8 | DISJOINT — p0 wins |
At h128 geometry the closed form (128 fold blocks ≥ 0.8·96 physical) already clears the FA2 fill target; the wave rule cuts splits below optimal and loses at every width, disjoint.

## 7b. Serve-geometry sweep, h256 24Q/4KV (battery attn_bf family; qwen3.5-9b decode row)
med_us (fixed s=16@512, 32@≥1024 | cu48 s=10 | cu96 s=14@512, 19@≥1024):
| ctx | p0 | cu48 | cu96 | overlap? |
|---|---|---|---|---|
| 512 | 131.1-132.4 | 157.6-157.7 | 157.7-158.1 | DISJOINT — p0 wins |
| 1024 | 230.3-236.4 | 217.1-217.6 | **168.0-168.3** | DISJOINT — cu96 wins −27% |
| 2048 | 307.5-307.8 | 309.7-310.0 | **225.9-226.8** | DISJOINT — cu96 wins −26% |
| 4096 | 453.6-453.9 | 492.5-494.7 | **339.6-340.9** | DISJOINT — cu96 wins −25% |
Recorded for the ledger: the FA2 rule at the PHYSICAL divisor wins disjoint at the serve geometry for ctx≥1024 (short-circuit s=19 ≈ the sweep-observed optimum region); at the runtime divisor (48) it picks 10 — flat-to-losing. A future h256-targeted retune row should start here. Out of scope for this bundle's gate.

## 8. p48 retune datum under the new policy (h256 16Q/2KV rows=1 ctx 8192, 3 interleaved rounds)
| leg | picks | med_us | vs fixed |
|---|---|---|---|
| p0 fixed | s=64 | 464.5/463.2/463.3 | baseline |
| wave-cu48 | s=19 | 529.4/530.3/529.4 | +14% LOSES (disjoint) |
| wave-cu96 | s=38 | 346.4/346.5/346.1 | −25% WINS (disjoint) |
| forced | s=48 | 322.3/321.9/321.7 | −30% WINS (disjoint) |
The w37 anomaly reproduces and is stronger (~30% vs ~20%). FA2 direction validated at h256; the row's rule as written lands near-optimal at cu96 (38 vs the 48 optimum, ~7% short) and below par at cu48. No single divisor satisfies both h128 (loses with fewer splits) and h256 (wins with fewer) geometries.

## 9. Serve decode A/B (spec9g stopped; v154-mounted arm containers from spec9g's own live spec; env-only arms; strict alternation + trailing baseline confirm; /tokenize-200 readiness; uuid-salted gen=512 rounds, temp 0; rates from the container's own done lines)
Arms: `q46_pol0` (R3D_SPLIT_POLICY=0) vs `q46_wave` (POLICY=1, SPLIT_CU=96 — the better kernel-level wave sub-arm). Rounds:
| round | arm | tok/s |
|---|---|---|
| r1 | p0 | 70.7 |
| r2 | wave | 63.9 |
| r3 | p0 | 63.6 |
| r4 | wave | 65.9 |
| r5 | p0 | 62.3 |
| r6 | wave | 65.8 |
| CONFIRM-A | p0 | 63.5 |
p0 {62.3,63.5,63.6,70.7} vs wave {63.9,65.8,65.9} — **spreads OVERLAP** (p0 max 70.7 > wave max 65.9; wave min 63.9 inside p0's range); confirm-A 63.5 sits inside p0's first-half cluster (no box drift). **0/3 disjoint positive rounds → mechanical PROMOTE-BOTH gate FAILS.** Dilution explains flatness: 9 full-attn layers of a hybrid model, short probe ctx (≤~532, the ≤512 regime where fixed already wins at kernel level), attention a minority of step time.
**Serve liveness resolution (the w43 law, upheld the hard way):** the arm containers initially logged ZERO DBG lines — investigated to root cause: **the radiance engine swallows plugin STDOUT** (printf); plugin stderr reaches docker logs. Proven with a throwaway stderr-DBG build (`v155test`, deleted after use, NOT a promotion artifact):
- q46_spec0 (spec-off, wave default): 16 `splitk` lines + `arm=wave cu=48 … spl=10 fold=1 hd=256` — serve DOES dispatch the split entry (n-gram spec decode was never the cause).
- q46_vfy0/q46_vfy1 (prod Cmd, port 9009): vfy0 → `arm=fixed cu=48`, `spl=32 fold=1`; vfy1 → `arm=wave cu=96`, `spl=19 fold=1`. **Per-arm liveness directly observed under the serve engine; the matrix arms were genuinely different code paths.**
Law recorded: serve-side liveness requires the debug wire on STDERR (or engine-logged stats); printf-only DBG is invisible under serve and would have produced phantom PASSes in earlier windows too.

## 10. Mechanical verdict walk
- merge battery zero-diff (M and W, all packs) + rsk 32/32 + hash sweep p0-equal → merge PASS. No drift anywhere in the identity class → inertness claim stands → WITHDRAW not triggered.
- policy tolerance PASS (§5 P-leg confined + in-bar; §6 in-band) → not "out of tolerance".
- serve decode A/B: 0/3 disjoint positive → NOT promote-both. Not "flat at every width" in the kernel sweep (h128 loses, h256≥1024 wins at cu96), but FLAT at serve level (overlapping spreads at the only serve-reachable comparison) → **PROMOTE-MERGE-ONLY**; default stays fixed (arm 0); manager holds the flip decision and the §7b cu96-h256 follow-on datum.

## Deviations & lessons
- (D1) n_act prefix bound NOT shipped — unsound vs the floor cut (§1a); the bit-identical bundle ships U=16 hoisted-load batching only. A prefix-bound wish requires changing the cut law first (precision class).
- (D2) `rocm-smi --showproductinfo` doesn't exist on cat's rocm-smi; CU facts from `rocminfo -a` + `hipGetDeviceProperties` on-card.
- (D3) Brief's "XTX: 32 CUs / 256 compute units" contradicted by measurement (runtime 48, physical 96); implemented the queue row's rule (0.8 × the device's CU count) and swept both divisors.
- (D4) First battery/id evidence run VOIDED before use: a pkill'd chain wrapper left an orphaned battery runner racing the fresh one on identical pack stems (the interleave produced a phantom gdn_chunk_scan FAIL — race, not a regression; md5-verified .so untouched by it). Rule: exactly one runner per stem namespace; kill whole process groups (`setsid` orphans survive parent kill).
- (D5) Serve-runner plumbing fixes (raw docker.sock keep-alive → curl; API-created containers need Config.ExposedPorts for port mappings; radiance /completion carries no id → tie done-lines by before/after diff; rate regex). No scientific impact; fixed scripts live in `~/scratch/r3d/q46_serve_ab.py`.
- (D6) **Engine swallows plugin stdout; serve liveness must ride stderr** (§9). Prior serve-path windows' log greps of printf-DBG would have been blind — check when next touching those ceremonies.
- (D7) radiance-flashnext restarted (RestartCount=1, OOMKilled=false) at 11:17:53Z, minutes after my first arm container started. For the record: my containers are pinned to the XTX via inherited `ROCR_VISIBLE_DEVICES=1` and never touched flashnext; kernel log shows no amdgpu reset/OOM; its restart cause is app-level on the RDNA4 pair. Flagged for visibility, not fought (per brief).

## Artifacts
- Repo (branch `card/q46-split-policy` on cat): code commit d2bcd6c (src/r3d_attn_paged_decode.hip, tools/gate.sh, tools/bench_attn_decode.hip, tools/q46_policy_check.cpp, tools/q46_id.sh, tools/q46_sweep.sh) + follow-on commit (tools/q46_sweep2.sh, tools/q46_serve_ab.py, tools/q46_serve_verify.py, tools/q46_spec0_probe.py, logs/q46/* evidence pack incl. this report).
- cat:~/scratch/r3d/: q46_batt2.log, q46_census_v153/154.txt, q46_policy_check.log, q46_serve.log (+ attempt1), q46_sweep2_wrap.log.
- cat:libr3d/q46logs/: q46_id.log, q46_sweep.log, q46_sweep2.log + per-leg raw logs.
- v154 = the gated build (build_v154/, mount-ready). Rollback container never started; v155test deleted.
