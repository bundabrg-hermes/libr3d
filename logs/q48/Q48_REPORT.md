# Q48 — Attention-side score reductions onto the shipped DPP/permlanex16 kit

Window: 2026-10-27, subagent card window on 'cat' (RX 7900 XTX, gfx1100, ROCr idx 1 = GPU1).
Branch: `card/q48-dpp-attn` (local only, NEVER pushed; main untouched; REVIEW_QUEUE.md / COVERAGE.md untouched).
Baseline: main @ 9c500b3; serve artifact build_v153 (spec9g live).
Head commit: **e2a05b3** (tools + logs only; no kernel edit).

## VERDICT: DARK (stale premise — time-box early-stop clause)

**There is no ds_bpermute butterfly anywhere in the attention family, and the
serving decode fold path's score reductions ALREADY ride the shipped Q6 DPP kit**
(Q11 fragment-side softmax, `R3D_ATT_FS` default-ON since build_v142 — landed
between the Q48 row's census datum and its booking). The queue row's premise
("fold path decode.hip:366 region predates the Q6 kit") is falsified at
source AND in the shipped v153 disassembly. Nothing was left to fold as a
pure instruction substitution, so no lever, no build, no battery, no A/B —
recording the census numbers and the DS-pipe datum instead, exactly as the
time-box clause prescribes: "if the premise is stale (butterflies already
folded or not dispatched by the serving arms), that is a legitimate
blocked-or-dark finding — stop early and report it precisely."

## 0. Card state at open / close
- Open: `spec9g` Up 4h (serve on build_v153, GPU1), `radiance-flashnext` Up (GPU0/GPU2 — RDNA4 pair, NOT the XTX, per window-35 attribution law, never touched). `spec9g_v151_rollback` Exited — left alone, never started. rocm-smi: GPU1 17.9/25.7 GB = spec9g model; no foreign squatter; nominal.
- Close: spec9g **never stopped** (no timing legs were run, so the stop/restart protocol never triggered); health 200, status running, restarts=0. Card left exactly as found.

## 1. Premise re-verification (grep/line evidence)

### 1a. The cited anchor exists — and points at the fold LAUNCHER, whose kernel body already rides the kit
- `src/r3d_attn_paged_decode.hip:365-366` (verbatim): `/* GQA fold ... one block per KV head streams the slice once for the whole q-head group` — the queue's ":366 comment verbatim" check out.
- The launcher dispatches the fold path via `R3D_DK_LAUNCHFS` (`:568`: `if (dbuf_ok && fs && d128) R3D_DK_LAUNCHFS(attn_paged_wmma_kernel, 128, 1);`), template slots `<D, BF16, SPLITK=1, DBUF=1, FOLD=1, PAD=0, F8ST=0, SB=0, FS=1>`.
- `R3D_ATT_FS` defaults ON (`:444-452`, "DEFAULT ON since build_v142"); spec9g's env is R3D_*-free (window-20 verification), serving shape is bf16 KV, head_dim 128, gqa 4 → `dbuf_ok && fs && d128 && use_fold` → **the FS arm IS the serving decode fold arm**.
- The kernel bodies live in `src/r3d_attn_wmma_body.inc`, #included by BOTH TUs (`r3d_attn_paged_decode.hip:125`, `r3d_attn_paged_wmma.hip:125`).

### 1b. The fold path's score reductions ARE the shipped Q6 kit, verbatim
`r3d_attn_wmma_body.inc:42-53`: `fs_rowmax` / `fs_rowsum` = 4-step ascending butterfly on `__builtin_amdgcn_update_dpp(p, p, 0x161/0x162/0x164/0x168, 0xf, 0xf, 0)` — literal control words (compile-time mask law honored), i.e. exactly `0x160|mask` ROW_XMASK from Q6/Q22. Used at `:569` (`rmax[k] = fs_rowmax(fmaxf(v0, v1))`) and `:578` (`rsum[k] = fs_rowsum(s0 + s1)`); pair-first a0/a1 combine is intra-lane; fold completes inside the 16-lane DPP row (the pinned acc lane map puts one tile row's 32 columns in ONE half-wave), so **no xor-16/permlanex16 tail even exists in this fold**.
The FS=0 withdrawal arms are NOT butterflies either: they are per-thread serial LDS row sweeps over Ssp (`:600-620`: `store_matrix_sync(Ssp...)` then thread-tid walks `srow[jj]`, jj ascending) — converting those to DPP folds would CHANGE the summation tree (window-19: "only the sum's tree differs"), i.e. precision-visible class — not Q48's FNV substitution class, and the FS=0 arms must stay verbatim as withdrawals anyway.

### 1c. Disassembly census on the shipped v153 artifact (tool: `tools/q48_census.py`, coverage-guard laws baked)
- **Tool validation (discipline f):** reproduces the authoritative Q19/Q22 numbers EXACTLY on v153 — `gemm_nt_w2a8_gemv_q8_kernel<4,16,...>`: 468 ds_bpermute (Q19: 468), <2,16>: 234, <4,8>: 244, <16,*>: 148; DPP v_mov: 208 at <4,16> (Q22: 208); permlanex16 present at the Q24 tails. Per-class non-zero hits asserted before trusting the attention zeros; a first parser revision hit the window-18 trailing-comment-address trap and was caught by the <SMALL-BODY>/PARSE-BUG guards (zero-instruction rows flagged, never silently trusted).
- **Attention family (69 instantiations across BOTH TUs): ZERO ds_bpermute in every single one** — including the serving decode fold arm `<128,BF16,1,DBUF1,FOLD1,0,0,0,FS1>`: 12,540 instrs, **0 bpermute, 192 v_mov_b32_dpp** (matches Q11's shipped "192 DPP butterflies" datum), 212 ds_swizzle. Also 0 in kernel2 (v2, env-gated, not dispatched), kernel3 (v3 thin, not dispatched), the 8 hd256 slice arms, the merge kernel, and the v1 SIMT fallback `attn_paged_kernel` (chain-sum, thread-per-row — never had a shuffle; `grep -rn shfl src/` shows zero hits in every attention file).
- **Tree totals (v153, 25 objects, 332 symbols):** 8,504 ds_bpermute — 100% in non-attention families (w2a8 q8, mxfp4 fallback arms, bf16 gemv, fp8a8, rmsnorm, sample/argmax/topk/logprobs, quant, qk_norm_rope_gate). 18,507 ds_swizzle — **17,892 (96.7%) in the attention family = the rocwmma column-B fragment feed** (Q19's attribution, confirmed at the larger v153 tree).

### 1d. The second Q48 target does not exist
"any Q40 Arm-B SIMT score loops": Q40 is status `gated`, never built — no Arm-B code in the tree (only the pre-existing v1 SIMT fallback, which has no cross-lane reduction, and it is dispatched only on foreign chips / `R3D_NO_WMMA`).

## 2. Three-way lane-exact probe (law a — done on card BEFORE any A/B consideration)
`tools/q48_probe.hip` (in branch; compiled hipcc -O2 -ffp-contract=off -Wall --offload-arch=gfx1100, 0 format warnings — Q33 lesson applied; run ROCR_VISIBLE_DEVICES=1):
- Shapes: EXACT shipped Q11-FS reductions — ascending ROW_XMASK 1/2/4/8 fmaxf-fold and += fold, plus the composed pair-first shapes (`fmaxf(a,b)` then fold; `(a+b)` then fold); three-way = shipped-form shfl chain vs independent LDS oracle vs DPP kit.
- Result: **PROBE-BITEXACT, 2.684e8 lane-ops per shape** (8 seeds × 4096 waves × 32 lanes × 256 iters). Gated legs (rowmax raw-bit 3-way; rowsum + composed production-class 3-way): zero mismatches. Raw-bit chain-vs-chain leg (DPP vs shfl, sNaN torture included): **0 mismatches** — informational, kept ungated per the law below.
- Forensics (`tools/q48_probe_dbg.hip`, kept with both lessons documented): (1) an early revision's ref leg shuffled the wrong accumulator → a vacuous zero-diff, the "third costume" of the vacuous-harness law; fixed and documented. (2) The LDS-oracle add path diverges on ~1.2e-4 of raw-bit iterations, only where running-value AND partner are both sNaN (the three lowering forms of `x += partner` emit different v_add_f32 operand orders; NaN payload preference follows the emitted order). The DPP-vs-shfl pair — the only pair the FNV substitution claims — measured bit-exact everywhere tested. Production rowsum inputs are exp2 sums (finite by construction, ≤ 32 terms of ≤1.0 each), so the FNV class of any such substitution is intact; re-check after toolchain bumps (codegen-contract class, Q42 lesson).
- permlanex16 was NOT re-probed: the attention fold never crosses the 16-lane DPP row (no xor-16 stage exists in fs_rowmax/fs_rowsum), and the Q24 kit leg is on record bit-exact at 2.68e8 on this toolchain (window 21, window 42).

## 3. Build
**No build cut (no v154).** There was no kernel edit to justify one: every candidate fold site is either already the kit (FS arms), a withdrawal arm that must stay verbatim (FS=0), an env-gated non-dispatched arm (kernel2/kernel3), or an arm no serving model reaches (fp8-KV fold arms — spec9g is bf16 KV; hd256 slice arms — spec9g is h128, and their non-FS softmax is a tree-changing port that belongs to Q40's step-0 numeric-class decision). Building a byte-clone v154 to run a "zero-diff battery" would be phantom-pass theater, explicitly refused.

## 4. Battery verdicts + per-leg lever liveness
Not run — no binary change existed to battery. No lever was added (`R3D_ATT_DPP` not created), therefore gate.sh needed no forwarding edit and no leg had an arm to prove. Stated here rather than faked, per the phantom-pass discipline (window-43 finding).

## 5. Census before/after (ds_bpermute / ds_swizzle) — the committed datum
| scope | ds_bpermute before (v153) | ds_bpermute after | ds_swizzle before | ds_swizzle after |
|---|---|---|---|---|
| attn family (69 insts, both TUs) | **0** | **0** (nothing changed) | 17,892 | 17,892 |
| serving decode fold arm `<128,3,1,1,1,0,0,0,1>` | **0** | 0 | 212 | 212 |
| tree total | 8,504 (all non-attn) | 8,504 | 18,507 | 18,507 |

Raw tables: branch `logs/q48/q48_census_v153.txt` (attn family, per-instantiation), `logs/q48/q48_census_tree.txt` (all 25 objects), `logs/q48/q48_census_w2a8.txt` (tool validation).
**Datum for the ledger (the dark-row datum): the DS-pipe suspect for the attention family is NOT butterflies — it never was on gfx1100 — it is the rocwmma column-B `ds_swizzle` fragment feed (96.7% of tree swizzles), which the DPP/permlanex16 source kit cannot replace: it is library lowering inside load_matrix_sync. The donor's RDNA2 "score loop bound by the LDS pipe" law does not attach to our mix at the butterfly level.**

## 6. A/B tables
None run (no arm pair exists; a same-binary A/B of identical code measures noise, which is exactly the "no-op arms adjudicate nothing" law from Q24). No serve decode A/B (spec9g untouched).

## 7. Mechanical verdict per the thresholds
- PROMOTE: impossible — no change, no A/B wins.
- WITHDRAW: no FNV drift was introduced (nothing changed); not the right word either.
- BLOCKED: probe, build machinery, and card state all worked; nothing mechanical failed.
- **DARK** per the time-box clause (stale premise: "butterflies already folded" for the dispatching arms — the FS fold path rode the kit since v142; "not dispatched by the serving arms" for the remainder). The row's own text admits this outcome: "Flat result is informative: DS-pipe suspect weaker for our mix than the donor's." Here it is stronger than flat: the suspect's butterfly mechanism does not exist in our attention code objects.

## 8. Deviations
1. **Stopped early at premise re-verification** — explicitly sanctioned by the time-box clause; effort went into making the falsification airtight instead (source grep + per-instantiation disassembly + tool validation against the authoritative Q19/Q22 numbers + on-card probe).
2. **No build/battery/A/B** (see §3/§4/§6): mechanical DARK preconditions (bit-identical battery, disjoint-flat A/B) are inapplicable with zero binary delta; the report carries the census + probe evidence the dark row owes instead.
3. Probe engineering: first census parser hit the known trailing-comment-address objdump trap (caught by its own guards); first probe revision contained a vacuous ref leg (caught desk-side via the dbg forensics tool); both lessons documented in the committed tools.
4. permlanex16 leg not re-probed (no xor-16 stage in the attention fold; Q24/Q33 record covers it on this toolchain).
5. Card protocol deviation: spec9g stop/restart never invoked (no timing legs). Card left as found; rollback container untouched.

## 9. Next steps for the manager
1. Queue: mark Q48 closed-dark with this datum; cite the v153 census (0 attn bpermute; swizzle feed attribution) — the row's premise "fold path predates the kit" needs the correction note (Q11-FS shipped it in v142; the row was booked in window 43).
2. If attention DS-pipe pressure is ever proven worth attacking (rocprofv3 ds-vs-vmcnt attribution — the Q50 counter pattern, never run on attn arms), the target is the rocwmma col-B swizzle feed (e.g. hand-laid V/K staging to feed mma operands without the fragment transpose) — a different item, not a kit transfer.
3. Q40 step-0 already has a free datum from this window: the hd256 slice arms carry the Ssp sweep softmax (no FS instantiation); an FS port there is precision-visible (l-tree differs), not FNV — Q40's numeric-class decision owns it.
4. Toolchain-bump regression knob: re-run `tools/q48_probe.hip` after any LLVM bump (the sNaN operand-order finding is a codegen contract, harmless today, contract-class per Q42's law).
