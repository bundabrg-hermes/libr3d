# External-model review queue

Canonical backlog of proposals from the external-model review passes defined in
`docs/review-prompts/`. This file is the ordered work queue: COVERAGE.md records
what has been *tried*; this records what is *queued*. Future review-pass models
should treat anything listed here as already identified (see the coverage_check
rules in the prompts).

Statuses: `ready` (validated at desk, awaiting card time), `gated` (needs a
specific check before it can move), `parked` (dependency or low priority),
`closed` (resolved without benchmark, with reason).

Card order below reflects the state as of 2026-10-25. Re-ordered as responses
land; each reorder cites the evidence that moved it.

| # | Item | Source | Status | Notes |
|---|------|--------|--------|-------|
| Q1 | m64 GEMM vec staging (uint4/b128 + kLdsPad 4→16) | QFN-P1 + Astra-P3 (independent convergence) | ready | Ports the shipped attention vec-staging pattern into the m64 tile loader; ledger says scalar staging deliberate only as fraction-of-tile-traffic argument. Astra b64 variant = fallback arm. |
| Q2 | DBUF barrier wait-idiom (`s_barrier` drains all in-flight global loads) | QFN pass2-P1 | **closed** | Gate ran on the v119 shipped slices: RDNA3 `S_BARRIER` does NOT drain counters (ISA p.254 verbatim: "Barrier instructions do not wait for any counters to go to zero before issuing") — the donor law is GCN-era, and half the attention barriers cross live global loads untouched (lgkmcnt-only waits). The `vmcnt(0)` before the other half is LLVM wait-insertion, not hardware. Donor idiom has no hardware target here; the residual (prefetch drained too early by the compiler) is a wait-placement problem whose only real lever is PREFETCH DEPTH — transferred to Q7. COVERAGE 2026-10-25 holds the census (8/16 DBUF barriers full-drain, 8-13 loads caught each). |
| Q3 | launch_bounds pins: mxfp4 (32,16), w2a8 (256,2) | QFN pass2-P5 | **closed** | Landed 7156a03 (+twins, gate.sh). Kernel-descriptor census (v119 vs v120 metadata): q8x family 22-47 VGPR, w2a8 gemv 45-62, zero spills in both builds — both pins were ALREADY satisfied by the shipped register allocation, so the pins are codegen-no-ops as claimed by their own arithmetic. KEPT as contracts (they forbid a future edit from silently spilling past the occupancy the launcher's geometry assumes); zero cost, gates v120 6/6 baseline-identical. |
| Q4 | Kill i64 div/mod in elementwise kernels | QFN-P3 | **built** | Landed 7156a03: actmul Contig/W32 arms, dflash_conv u32 (provably in range via the launcher's 0xffff*kThreads cap) + tap-modulo hoist, band block-uniform split + latent grid-x-wrap guard. ROPE SITE REJECTED: `w / half` was already u32 in place — the claim conflated rope with band. |
| Q5 | `__builtin_nontemporal_load` on q8x weight legs | QFN pass2-P8 | **built** | Opt-in `R3D_Q8X_NT` weight-leg-only (activations inline, cold-plane law); same-binary paired bench pending. |
| Q6 | DPP/permute folds replacing `__shfl_xor` chains | Astra-P5 + QFN pass2 (independent convergence) | ready | Attribution CORRECTED from v119 disasm: the 96/iter@MT=8 figure is the w2a8 gemv_q8 family (96->148 across MT, x3 insts); the mxfp4 q8x serving row carries 40 (5 folds/m x 8). Attention swizzle aggregate is 8,736 across 42 instantiations (rocwmma fragment staging — not our code). Q6's true target is half the advertised size on the serving kernel. |
| Q7 | Depth-2 prefetch + ballot-gated lazy rescale | QFN pass2-P2 | ready | RESEQUENCED from parked: the Q2 close transfers its residual here. Depth is the only lever that survives LLVM's barrier wait-insertion (issue #8 schedule-transfer law applies — A/B the mxfp4 q8x family separately, template the depth). |
| Q8 | 16B elementwise walks (copy-style kernels) | QFN-P4 + Astra-P6 | **built** | kv_store VEC uint4 arm (16B alignment + sd==1 gate, byte-identical copies, scalar kept); lands with the Q4 batch on build_v120. |
| Q9 | `v_dot2c_f32_bf16` in decode GEMVs | QFN-P2 | ready | 3-operand form via `dot2c_f16_m4` asm precedent (4-operand rejected by this ROCm's assembler — probe first). fp8→bf16 LUT already exists. |
| Q10 | fp8 attention / KV staging widening campaign | Astra-P4 + QFN pass2-P3/P4 | ready | Merged campaign; hipfire leg of QFN-P4 is the load-bearing donor cite (`load_int_b2` anchor at pinned llama.cpp commit was stale). |
| Q11 | Fragment lane-map epic: fragment-side softmax (QFN pass1-P7) + VGPR group flushes (Astra-P2) | both models | gated | Single blocker: pin WMMA accumulator lane map on-card (no rocWMMA coordinate fragments; ledger :858). Two branches, one probe. |
| Q12 | w2 compressed-input dot4 (drop u8 expanded plane + repack cache) | Astra-P1 | ready | 4x weight-traffic cut on cold passes; bit-expansion trick hand-verified at desk. |
| Q13 | `v_cvt_off_f32_i4` unpack | QFN-P6 | parked | Real instruction, small payoff. |
| Q14 | MUBUF `buffer_load_*` hardware clip for ragged/tail loads | QFN-P8 | parked | Load-class half closed by Q2-gate objdump (flat already lowers to global; 1,192 global vs 11 flat in attention, m64 all-global). Residual: predicate/tail-loop deletion + removal of 3 hand-rolled OOB mechanisms. Needs per-tensor SRD cache (ptr/pitch/extents). Bug-surface value > perf value. |
| Q15 | hd256-milestone item | QFN pass2-P9 | parked | Rides hd256 milestone 3. |
| Q16 | Lowest-expected-gain pass2 item | QFN pass2-P7 | parked | Rated last by its own author. |

## Closed without card time

| Item | Source | Closure |
|------|--------|---------|
| flat-vs-global load selection | QFN-P5 | Objdump of v119 build, MAINTAINER-RE-VERIFIED on the full family (not a slice subset): attention family 2,902 `global_load_*` / ZERO flat / ZERO buffer; m64 tiles 110 global / 0 flat. Compiler already optimal. Entered in COVERAGE.md 2026-10-25. |
| Q2 DBUF barrier idiom | QFN pass2-P1 | ISA p.254 + shipped-slice disasm (see queue row above and COVERAGE chronicle). |

## Ledger fixes owed to COVERAGE.md

ALL THREE LANDED 2026-10-25 (COVERAGE.md, "The review queue opens" section):
the `R3D_FP4_DOT2` leg record, the flat-vs-global closure, and the ds-pipe
traffic census (with the Q6 attribution correction).

## Submission integrity record

| Model | Pass | Citations | Defects |
|-------|------|-----------|---------|
| QFN | 1 | 14/15 | `buffer_load_dword` mnemonic slip (fixed in full version); missing `validation_notes` + `no_op_findings` blocks (silence carries no weight) |
| Astra | 1 | 12/12 (cleanest) | Reviewed snapshot 44b614b, one commit behind; anchors verified |
| QFN | 2 | 15 donor files at pinned SHAs, rationale comments verbatim | P5 w2a8 mapping corrected (kThreads=256); P4 `load_int_b2` anchor stale at pinned llama.cpp commit; q8x issue count 5 not 7; some line anchors drifted 5–20 lines (HEAD moved) |
| maintainer | desk | — | Desk-verify of pass-2 against the tree + shipped disasm: P1 donor law false on RDNA3 (closed above); P3 rope site already u32 (rejected); P1/P6 "96 @ q8x MT=8" belongs to the w2a8 family, mxfp4 q8x row carries 40; pass-2's flat/global closure re-verified with strictly cleaner numbers |

Pending: Astra pass 2. When it lands, cross-verify its donor reads against QFN pass 2 (especially the Q2 barrier idiom — a second gfx11 sighting would downgrade the objdump gate to a spot check).
