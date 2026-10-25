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
| Q2 | DBUF barrier wait-idiom (`s_barrier` drains all in-flight global loads) | QFN pass2-P1 | gated | Gate: objdump the v119 DBUF attention body; count `vmcnt(0)` before `s_barrier`. If bare, closes as no-op. Donor rationale verbatim-verified in Strata + hipfire. Attacks the 171 GB/s binding term; 10 sites in DBUF arm, 16 more non-DBUF. |
| Q3 | launch_bounds pins: mxfp4 (32,16), w2a8 (256,2) | QFN pass2-P5 | ready | Target arithmetic corrected from donor (w2a8 GEMVs use kThreads=256, not wave-per-row). Grouped with Q4 as the cheap batch. |
| Q4 | Kill i64 div/mod in elementwise kernels | QFN-P3 | ready | 5 sites (actmul, dflash_conv decode x3, kvstore, rope, band); cast Contig precedent in-tree. |
| Q5 | `__builtin_nontemporal_load` on q8x weight legs | QFN pass2-P8 | ready | Donor: +10–12% stream, −3% in-engine (cited honestly). q8x repack layout already satisfies precondition. |
| Q6 | DPP/permute folds replacing `__shfl_xor` chains | Astra-P5 + QFN pass2 (independent convergence) | ready | CONFIRMED by disasm: `__shfl_xor` lowers to `ds_bpermute_b32`, 96/iter in q8x MT=8; attention body also carries 5,184 `ds_swizzle_b32` from rocwmma. Donor measured −27% with XOR tree preserved (bit-exactness law satisfied). |
| Q7 | Depth-2 prefetch + ballot-gated lazy rescale | QFN pass2-P2 | parked | Sequenced after Q2: the cheap barrier is what lets the deeper prefetch survive. |
| Q8 | 16B elementwise walks (copy-style kernels) | QFN-P4 + Astra-P6 | ready | kvstore scalar walk confirmed. Same class as Q4; merge bench arms. |
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
| flat-vs-global load selection | QFN-P5 | Objdump of v119 build on cat: attention emits 1,192 `global_load_*` vs 11 stray flat; m64 all-global; `global_load_b128` present. Compiler already optimal. Enter in COVERAGE.md. |

## Ledger fixes owed to COVERAGE.md

1. Record the `R3D_FP4_DOT2` opt-in leg (present in mxfp4.hip, absent from ledger) — QFN cited a ledger entry that doesn't exist; the code leg is real.
2. Record the flat-vs-global closure above.
3. Record the ds-pipe traffic census (96 `ds_bpermute`/iter q8x MT=8; 5,184 `ds_swizzle_b32` attention body) — motivates Q6/Q11.

## Submission integrity record

| Model | Pass | Citations | Defects |
|-------|------|-----------|---------|
| QFN | 1 | 14/15 | `buffer_load_dword` mnemonic slip (fixed in full version); missing `validation_notes` + `no_op_findings` blocks (silence carries no weight) |
| Astra | 1 | 12/12 (cleanest) | Reviewed snapshot 44b614b, one commit behind; anchors verified |
| QFN | 2 | 15 donor files at pinned SHAs, rationale comments verbatim | P5 w2a8 mapping corrected (kThreads=256); P4 `load_int_b2` anchor stale at pinned llama.cpp commit; q8x issue count 5 not 7; some line anchors drifted 5–20 lines (HEAD moved) |

Pending: Astra pass 2. When it lands, cross-verify its donor reads against QFN pass 2 (especially the Q2 barrier idiom — a second gfx11 sighting would downgrade the objdump gate to a spot check).
