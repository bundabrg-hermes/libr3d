# libr35d triage (radiance-strix-halo, gfx1151)

Source: https://codeberg.org/testeddoughnut/radiance-strix-halo (cloned 2026-10-25;
libr35d README + TUNING.md + `src/r35d_gemm_bf16_nt_m64.hip` read in full; kernels not audited
line by line). Maintainer describes it as AI-ported libr4d (RDNA4) with no careful stewardship;
the tuning ledger, however, carries before/after numbers and an unusual number of rejected
experiments, and its fragment-layout claims match our hardware-verified gfx11 facts (no AGPRs,
v16h operands, accumulator row `2e + (l>>4)`, wave32 lanes 16-31 mirroring 0-15). Treat
mechanisms as trustworthy leads and every number as gfx1151-only.

Portability: LLVM puts gfx1100-1103 and gfx1150-1154 in one amdgpu11 family. LDS/VALU/WMMA
issue behaviour, the compiler traps below, and the layout facts carry to gfx1100. What does not
carry: their constants (20 WGPs, ~241 GB/s APU bandwidth, 32 MB MALL) vs ours (96 CUs, GDDR6,
96 MB L2); their WMMA-shim machinery (we are native); their non-temporal null result (APU L2
semantics, our Q8 stays).

## Queue deltas (merge into REVIEW_QUEUE.md ordering)

1. Q1 (m64 vec staging) gains a donor and two refinements — `src/r35d_gemm_bf16_nt_m64.hip`:
   - Rotated load issue at power-of-two strides: K % 4096 == 0 (16 KiB rows) camps DRAM
     channels; each wave issues its slice's 32 steps from `(tile + split) % 32` into fixed
     registers (sums unchanged, bytes unchanged). Their N 1024 K 8192: 60 GB/s rotated-to vs
     185 at K 8448, then 0.28 -> 0.11 ms. Genuinely new: our queue has vec staging but no
     rotation, and our own ledger has no channel-camping entry. Gate on stride; they measured
     the rotation costing a few % at odd K.
   - Staging is gated to MT >= 2 row tiles: at one row tile the same staging was 3-10% slower
     (their rejected list). Our Q1 bench must split one-tile vs multi-tile shapes accordingly.
   - Rejected donor variants to skip: 64-byte windows, 16-step groups, register-held groups
     across the start switch.
2. New cheap item — spill census landed (see below): gdn recurrent kernels spill 320/222/189
   VGPRs and the attn_paged_wmma h256 MT=3 arms spill 88-138. Register-pressure work on those
   two files precedes any new attention scheduling idea (Q11 epic), since their donor traps
   (sched_barrier(0) live-range caps, tied-asm operand temporaries, LDS-order staging) are
   exactly the tools that fixed their analogues.
3. Q4 / fp8-KV staging (Claude-P4) should use the donor's 3-instruction E4M3 -> f16 widening
   (`v_perm_b32` + `v_pk_ashrrev_i16` + `v_and_b32 0xbfffbfff` = value x 2^-8 exactly, checked
   over every code; fold the 2^8 back into softmax fma / output scale), not a LUT. Their fp8
   tiles gained 6-11% over the 4-instruction form; LUT gathers lose to VALU when the widening
   happens once per staged block. Note their fp8-decode spill cliff: a software E4M3 decode
   inlined in an unrolled loop hit 1400-2800 spilled VGPRs; widening belongs at tile staging.
4. New desk item — compiler-trap audit of our hot loops against their traps (all family-level):
   - true16 no-op pairs (`v_lshrrev_b32` + `v_mov_b16.h.l` around f16 WMMA operands): count
     them per kernel in our disassemblies; their fix is issuing the WMMA as plain (not volatile)
     inline asm; `volatile` was slower, hoisting/LICM spilling was the failure mode.
   - Guards inside unrolled K steps become per-WMMA exec-mask branches; keep fragment-0
     unguarded, drop guards on always-live rows.
   - Reading a natural-order LDS row then renaming dwords costs a v_mov each; stage rows in
     operand order instead.
   - `__shfl_xor` on gfx11 lowers to `ds_bpermute` (we already confirmed 96/wavefront in q8x);
     their DPP8 fold (Q-DPP) has a second independent donor now (their MoE GEMV uses the
     butterfly, with the TPR<=32 law: at TPR 64 the halves were never summed — a 20%-faster
     wrong variant; check our GEMV reductions for the same trap).
   - Adjacent `ds_load_b64` merge into b128 only outside inline asm.
5. New candidate (parked, medium) — WMMA skinny decode GEMM vs GEMV at M 1-16: their
   `r35d_8a8_narrow_kernel` (16-row fragment tile a block, tokens as B operand) streams
   206-236 GB/s where the register-widening GEMV got 174-228 on the same part. Our q8x default
   is a GEMV; worth one A/B at serving shapes before Q8-class micro-tuning closes that space.
6. New bench-protocol items adopted into our card windows:
   - Time decode kernels at verify widths (M up to draft-window+1), not M=1 alone; their
     logits-gemm story (scalar chain flat-to-4 vs WMMA flat-to-16) is a whole-decode gap class.
   - Rotate weight copies past L2 in harnesses (XTX 96 MB L2 makes this more important than
     their 32 MB MALL: a non-rotated bench of a <=96 MB weight times L2, not GDDR6).
   - A/B alternating runs (A,B,A,B) as the default comparison, never two far-apart runs.
   - rocprofv3 runs on harnesses only, at most four counters; server dies under it (their
     experience; assume ours similar until tested).
7. Their entry-major-dispatch / L2-convoy law (blocks sharing a stream must walk it together;
   a latency-hiding prefetch can DECREASE L2 hits 90->30% by letting the convoy drift) is a
   caveat for any Q2-DBUF prefetch work and for MoE-shaped ops: register as a risk note on both.

## Not portable, recorded so nobody re-imports

- WMMA shim + gfx12 layout machinery (we never had gfx12 rows).
- "Non-temporal made no difference" (APU; our Q8 cites hipfire's discrete-card +10-12% instead).
- Their 54 TFLOP/s WMMA sustained and 241 GB/s (gfx1151 constants; ours need their own
  microbenchmarks — their `tools/isa` suite pattern is worth mirroring if we formalise ours).
- i8a8 "exactness not speed" claim: their iu8 WMMA measured no faster than f16 on gfx1151;
  gfx1100 rate not yet measured by us — the claim is an open question, not a refutation, since
  our dot4 integer path already exists for the GEMVs.

## Spill census (fresh container build of main @ 61e4432, 2026-10-25)

Tool: `tools/spill_census.py` (runs inside radiance-dev; reads the AMDGPU metadata notes of the
gfx1100 code objects in each unit's fatbin). Result: 181 gfx1100 kernels, 25 with spills or
private segments. Top of the cliff, VGPR spills (private bytes):

    320/708  gdn_conv_recurrent_update_kernel<1>
    222/692  gdn_recurrent_update_kernel<1>
    189/560  gdn_chunk_scan_kernel
     98/360  gdn_recurrent_update_kernel<0>
     88/324  attn_paged_wmma_kernel<256,3,0,1,0,0>
    138/268  attn_paged_wmma_kernel<256,3,0,1,0,1>
     73/284  gdn_conv_recurrent_update_kernel<0>
      0/272  gemm_nt_w2a8_gemv_q8_kernel<*,16>   (private without spill: check dynamic stack)
      0/272  ... all five <*,16> arms; 0/144 all five <*,8> arms
      5/224  attn_paged_wmma_kernel2<128,4,1> (decode)

Read: GDN is our worst unit by register pressure and it is on the serving path for Flash-Next
hybrid layers; the h256 MT=3 WMMA attention arms (prefill family behind the DBUF work) spill
before any new scheduling idea is attempted on them. The w2a8 q8 GEMV private segment with zero
spill looks like stack/LDS-path traffic, not register spill — verify before chasing.
