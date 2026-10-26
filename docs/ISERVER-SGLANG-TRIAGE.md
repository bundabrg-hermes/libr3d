# iserver sglang `decode-gfx1100` triage (donor: miversen33/sglang, branch `decode-gfx1100`)

Donor: https://git.iserver.me/miversen33/sglang — SGLang serving Qwen3.8-27B with DSpark
spec-decode on 2x→4x RX 7900 XTX (gfx1100, wave32, 96 CU). Branch tip `3d3d952` is a
`.knowledge/` decision ledger (46 files: decisions / gotchas / conventions / playbooks)
built alongside ~40 `[AMD]` kernel commits. Read the whole ledger before trusting any
single claim — the revert catalog (`gotchas/reverted-approaches.md`) is the most valuable
part and follows the same discipline as our COVERAGE.md.

Portability: same physical card as our bench host, same ISA family. Their engine-side
machinery (HiCache, DSpark wiring, all-reduce, radix cache, launch-count engineering,
per-shape variant tables) is SGLang plumbing — not portable, not attempted. What ports
is the hardware-law layer. Dispositioned below; two items became queue rows (Q30, Q31).

## Porting items

### P-A (Q30) HSA scratch-reclaim env — free, measured, not yet on our deployment

Their measured datum: `HSA_NO_SCRATCH_RECLAIM=1` + `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0`
(image-level env) gave **46.0 → 51.3 tok/s at 22k and 29.4 → 31.9 at 112k** (one-run A/B
against a reverted image, per their own benchmarking rules). Mechanism: when a
register-spilling kernel launches inside HIP graphs, the runtime's scratch-reclaim path
stalls the round; the switches disable it.

Why this lands on us: our shipped build still has in-loop spilling (h256 DBUF arms ~100+
VGPR spills, per Q23's census — resident-accumulator class, no cure built). We probed
the live `radiance-flashnext` container on cat 2026-10-26: **zero `HSA_*` vars in its
environment** — the fix is unapplied. Zero-code, zero-binary change: env on the serving
container, serve A/B. Risk: scratch memory is never reclaimed, so peak VRAM grows by the
spill high-water mark — our free-VRAM bars (512 MiB gate, window-9 budget-noise class)
are the guard. If the engine OOMs at KV sizing, the env loses.

### P-B (Q31) WGP store-visibility law — new to our ledger; census our sites

Their measured bug (cause analysis in `fused-kernels-not-bit-exact-by-construction.md`,
`div_probe.py`): in WGP mode a workgroup spans two L0 caches; **global stores are
counted by `vscnt`, not `vmcnt`**, and the plain barrier idiom (`s_waitcnt vmcnt(0);
s_barrier`) does not wait for them — a store→barrier→read pattern across threads read
stale data on ~5% of values. Correct fence: `s_waitcnt_vscnt null, 0` → `s_barrier` →
`buffer_gl0_inv`. (gfx9's vmcnt covered stores; this trap is gfx10+.)

Ledger check 2026-10-26: `vscnt` / `gl0_inv` appear **nowhere** in COVERAGE.md,
REVIEW_QUEUE.md, or LIBR35D_TRIAGE.md — genuinely new law. Our shipped LDS staging is
immune by mechanism (`ds_store` is lgkmcnt-counted and s_barrier orders LDS), so the
exposure is only GLOBAL store→barrier→global-load sites. Q31: census the shipped
objects for that class (and confirm the compiler's `__syncthreads()` lowering inserts
vscnt where the C++ memory model demands it). Matters most to any future cross-kernel or
cross-WG scratch fusion (Q27/Q28-class, Q14 SRD work).

### P-C fp32 division datum — ledger line

They probed (cause-1 analysis): Triton's fp32 `/` on gfx11 is **IEEE in every form
tested** — division itself is not a rounding source; only `a * (1 / b)` differs. This
closes a cheap class of "replace division by reciprocal-multiply" ideas that a future
reviewer will propose: on this ISA it is precision-visible (and our `-ffp-contract=off`
law + Q4's u32 rewrites already own the actual div-cost lever, which is integer).

### P-D fusion-placement evidence — strengthens Q27/Q28 design choice

Donor measured BOTH placements of the same fusion: activation-quant fused INSIDE the
consumer GEMM regressed the verify path 32.35 → 61.02 ms and served decode −22%
(reverted); fused at the PRODUCER (norm / SiLU returning the quantized tuple) shipped
with −2 to −6 us per call and a bit-exactness battery. Same law we chose for Q27
(rope∘kvstore, producer-side) and Q28 (GEMV+actmul — note Q28 fuses into the GEMV's
own epilogue, which is consumer-side; their data says the epilogue is only safe where
it replaces the final store, not where it re-reads under the main loop). Recorded on
the Q27/Q28 rows as donor evidence; the ABI graph-gate is unchanged.

### P-E changing-inputs verification law — one-line bench/verification addition

Their all-reduce incident: a probe with CONSTANT inputs "passed" while the kernel
returned stale peer data, because a stale read of unchanged data equals the right value.
Fix: dependent-chain probes with values that change every iteration (`chain_check`).
Generalize to our verification: any battery leg whose expected output is bit-identical
across iterations cannot detect a stale/skipped read — our fingerprint packs vary
inputs per case, but single-shape liveness probes should be chain-shaped. Law adopted
into the card protocol.

## Independent confirmations (no action; raises donor trust)

- **Skinny WMMA beats row-GEMV at verify widths, GEMV owns M=1** — same crossover
  shape our Q21/Q26 measured (their keeper: 16-key blocks; register-pressure-tuned).
- **fp8: staging beats in-place conversion.** Byte loads 2.6x slower than bf16;
  bit-trick upcast in attention measured 16 → 11.9 tok/s decode then reverted;
  staging-to-bf16 won (~3x). Validates Q10's shipped design and the 8-byte gather
  choice; also matches Q9's "no hardware fp8→bf16" finding (their wording) — our
  v_perm trick is ALU-side, they never tried it; their byte-load datum is the
  constraint it must respect.
- **In-order load completion; branch-free clamped prefetch** — Q25's law, donor-side.
- **`sched_group_barrier` speeds isolated math, slows the full kernel; micro-bench
  is not evidence** — Q23's fence-by-16 lesson and the Q26c harness laws, donor-side.
- **Wave-half register duplication on matrix ops; spill cost is real (~13% class)** —
  the Q17/Q18/Q20 census family, donor-side.
- **Rotated-weights bench protocol (96 MB IC)** — adopted already (LIBR35D delta6);
  donor independently lists it as rule #2 for the same reason.

## Rejected / not applicable

- Triton-vs-CK prefill attention (22-28 vs 76 TF/s): radiance's prefill path is not
  libr3d's surface; note for the engine team only.
- Custom PCIe all-reduce, HiCache, DSpark/ReplaySSM, expandable segments, model-folder
  shape fixes, `HIP_FORCE_DEV_KERNARG` (they measured: no effect on launch gaps):
  SGLang plumbing.
- `page-size disables gfx11 attention` and other engine gotchas: no libr3d analogue.
- Producer-side norm+quant fusion kernels themselves: radiance's dtype universe has no
  i8-activation-of-norm edge today (i8a8 quantization happens at load/repack); revisit
  only if Q27/Q28-class ABI work lands.

## Queue deltas from this triage

- Q30 (new, ready): HSA scratch-reclaim env serve A/B on cat (P-A). No build.
- Q31 (new, desk): global store→barrier→load census + `__syncthreads()` lowering
  check on shipped objects (P-B). No card.
- Laws added to the card protocol: chain-check probes (P-E); reciprocal-multiply
  precision-visible on gfx11 (P-C).
