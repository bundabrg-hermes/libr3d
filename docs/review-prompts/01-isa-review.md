# Pass 1 — ISA Optimization Review Prompt

Paste everything below the line to a fresh model session that has access to
this repository (clone of `bundabrg-hermes/libr3d`).

---

You are performing a **read-only ISA optimization review** of a GPU kernel
library. Do not write tests, do not compile, do not run or benchmark anything,
and do not propose work that requires execution to describe. Your deliverable
is research and code review only.

**The project.** libr3d is an out-of-tree HIP kernel library for the Radiance
LLM inference engine, targeting AMD RDNA3 (gfx1100/gfx1101, e.g. RX 7900 XTX).
Repo: https://github.com/bundabrg-hermes/libr3d — kernels live in `src/`
(GEMM/GEMV: `r3d_gemm_*.hip`, paged attention: `r3d_attn_paged_*.hip`, plus
rmsnorm, rope, actmul, kvstore, embed, sample kernels).

**Mandatory first step.** Read `docs/COVERAGE.md` in full. It is the ledger of
techniques already implemented, already tried, or deliberately deferred. Any
proposal that duplicates an entry there is rejected unless it is a concrete
refinement of the tried approach — say so explicitly if so.

**Instruction-set reference.** The repo contains a full extracted corpus of the
AMD RDNA3 ISA manual in `docs/isa/` (one markdown file per instruction, plus
narrative chapters). Prefer it. The official manual is AMD document 69267,
"AMD RDNA 3 Instruction Set Architecture" (amd.com), if you need wider context.

**Ground truth (verified on hardware, do not "correct" these):**
- gfx1100 has `v_dot4_i32_iu8` / `v_sudot4` (verified working). It has **no**
  AGPR/accvgpr accumulators; WMMA on RDNA3 uses plain VGPRs.
- Decode is memory-bandwidth-bound; deep-context KV attention runs
  ~215-217 GB/s effective on the shipped serving arm (v142-era rebaseline,
  queue row Q25; practical ceiling ~500 GB/s).
- Dtype universe is closed: {bf16, f32, fp8a8 (128-block scales), i8a8,
  mxfp4a8 (E8M0/32-block), w2a8}. Proposals for dtypes outside this set are
  phantom work.
- Caller contracts: all tensors passed to r3d kernels are >=16B-aligned and
  row-padded per the conventions in the headers; the engine never mutates
  buffers a kernel is reading.

**What to look for.** Where the current kernels spend instructions the ISA can
do cheaper or in fewer ops. Examples of the kinds of things worth checking:
`v_dot4`-style integer dot products replacing scalar unpack-and-multiply chains
(e.g. in the mxfp4/w2a8 GEMV paths), `v_permlane*`/`v_perm_sv`/`ds_swizzle`-class
data movement vs the current shuffle or LDS round-trips, `v_cvt_*`/
packed-conversion instructions replacing manual bf16/fp8 unpack,
`ds_read_b128`/`global_load_dwordx4` wider accesses, SMEM/LDS bank layouts,
`s_waitcnt` scheduling density, buffer (`buffer_load`) addressing vs flat
loads, `v_mad_mk_*`/literal-operand forms, and anything in the manual's
encoding chapter that cuts VALU/SALU/LDS instruction count in hot loops. Trace
each hot kernel's inner loop, count the ops, and only then consult the ISA.

**Output format.** Respond with exactly one fenced YAML block, no prose outside it:

```yaml
proposals:
  - id: P1
    title: short name
    target: "file.hip:function or line range"
    status_quo: what the code does now (1-2 sentences, cite the actual instructions/idiom)
    proposal: the ISA-level change you recommend (pseudocode or small diff sketch welcome, full implementation not required)
    isa_basis: ["exact instruction names", "manual chapter/section"]  # REQUIRED - no citation, no proposal
    expected_gain: why it is faster (instruction count delta, wavefront occupancy, LDS traffic, etc.) and rough magnitude if estimable
    coverage_check: why this is not already covered by docs/COVERAGE.md
    risks: correctness/precision/portability concerns
    effort: S | M | L
    confidence: high | medium | low  # high only if you verified the instruction exists in docs/isa and applies to gfx11
validation_notes: how the maintainer should benchmark each proposal (which kernel geometry to A/B)
no_op_findings: areas you reviewed where current code already looks optimal (be honest, not exhaustive)
```

Rules: every proposal must cite exact instruction names that exist in
`docs/isa/` — do not invent mnemonics or import CDNA/ROCm-server-GPU features
(no matrix cores beyond RDNA3 WMMA, no warp operations that don't exist on
gfx11). Order proposals by expected gain. If you're unsure an instruction
exists on gfx11, mark confidence low and say what to check — an executor will
verify everything on real hardware, so speculative-but-cited ideas are more
valuable than safe vagueness.
