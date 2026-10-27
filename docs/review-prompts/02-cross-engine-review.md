# Pass 2 — Cross-Engine Kernel Review Prompt

Paste everything below the line to a fresh model session that has access to
this repository (clone of `bundabrg-hermes/libr3d`) and can browse public
GitHub repositories.

---

You are performing a **read-only cross-engine kernel review**. Your job is to
find techniques that proven inference engines already use for the same kernel
shapes, and assess whether they transfer to our RDNA3 kernels. Do not write
tests, do not compile, do not run or benchmark anything. Research and code
review only.

**The project.** libr3d is an out-of-tree HIP kernel library for the Radiance
LLM inference engine, targeting AMD RDNA3 (gfx1100/gfx1101). Repo:
https://github.com/bundabrg-hermes/libr3d — kernels in `src/` (GEMM/GEMV:
`r3d_gemm_*.hip`, paged attention: `r3d_attn_paged_*.hip`, rmsnorm, rope,
actmul, kvstore, embed, sample). Read the relevant libr3d kernel first so you
know the shape you are comparing against.

**Mandatory first step.** Read `docs/COVERAGE.md` in full — the ledger of
techniques already implemented, tried, or deferred. Proposals duplicating an
entry there are rejected unless they are a concrete refinement; say so
explicitly if so.

**Reference engines (donors).**
- `Niko1221/Strata` — MIT-licensed RDNA3-targeted inference engine (pushed
  weekly). Closest donor: same architecture target, so its code is
  RDNA-proven by construction.
- `Ben1078/hipfire` — RDNA-native inference engine. Prior microbenchmark on a
  7900 XTX measured its quantized GEMV style at 2.5-3.5x our current GB/s on
  identical mxfp4/q8x shapes, so its memory-access and accumulation structure
  is worth reverse-engineering.
- `ggml-org/llama.cpp` — `ggml/src/ggml-cuda/` per-quantization `vec_dot`
  inline asm and the mmvq/mmq kernel family. CUDA, so idioms must be
  *translated*, not copied: justify each translation against the RDNA3 ISA
  corpus in `docs/isa/` (bundled in this repo).
- Other engines (vkFFT/vkMatmul-style Vulkan projects, MLC-LLM, TRL/
  tinygrad backends, whatever you know) are welcome if the donor code is
  public and you can cite it.

**Ground truth (verified on hardware, do not "correct" these):**
- gfx1100 has `v_dot4_i32_iu8` / `v_sudot4` (verified working). No
  AGPR/accvgpr accumulators; RDNA3 WMMA uses plain VGPRs. No CDNA matrix
  cores, no warp shuffles beyond what `docs/isa/` lists.
- Decode is memory-bandwidth-bound; deep-context KV attention runs
  ~215-217 GB/s effective on the shipped serving arm (v142-era rebaseline,
  queue row Q25; practical ceiling ~500 GB/s).
- Dtype universe is closed: {bf16, f32, fp8a8 (128-block scales), i8a8,
  mxfp4a8 (E8M0/32-block), w2a8}. Donor kernels for other quant formats
  (e.g. NVFP4, MXFP8) are accepted only as *structural* ideas, not formats.

**What to look for.** Kernel-level structural patterns, not tuning flags:
weight layout/ordering that beats our interleaving, dot-product accumulation
strategy (integer accumulate width, scale-application point), thread/block
mapping and vector width for memory-bound decode, split-K / stream-K for
tall-skinny decode GEMMs, persistent-kernel and cooperative-group patterns,
paged-KV block iteration order and vector width, epilogue fusion (rope/norm
fused into producers), and prefetch/software-pipelining depth choices.

**Citation rule (strict).** Every proposal must cite the donor implementation
at `file:line` granularity from a public repo (pinned commit or branch), and
state whether the donor runs on RDNA/AMD hardware (proven) or CUDA/other
(translated). A proposal without a donor citation belongs in pass 1 (ISA
review), not here.

**Output format.** Respond with exactly one fenced YAML block, no prose outside it:

```yaml
proposals:
  - id: X1
    title: short name
    target: "libr3d file.hip:function this would replace or change"
    donor_engine: repo name
    donor_ref: "org/repo path/to/file.ext:LINE-LINE @ branch-or-commit"  # REQUIRED
    donor_proven_on: RDNA | CUDA | other  # translated idioms need isa_basis too
    isa_basis: ["gfx11 instruction names backing the translation"]  # REQUIRED if donor_proven_on != RDNA
    transfer_plan: what changes in our kernel to adopt this (structure, not full code)
    expected_gain: why the donor is faster on our shapes; cite our COVERAGE.md baseline numbers if applicable
    coverage_check: why this is not already implemented/tried per docs/COVERAGE.md
    rejected_comparisons: donor techniques you examined and deliberately did NOT propose, with one-line reasons (this section is required — its absence fails the review)
    risks: correctness/precision/licensing/portability concerns
    effort: S | M | L
    confidence: high | medium | low
validation_notes: how the maintainer should A/B each proposal (which kernel geometry, which baseline in COVERAGE.md to beat)
no_op_findings: donor approaches you examined where our code already matches or the idea cannot transfer to gfx11
```

Order proposals by expected gain. Prefer one well-cited structural proposal
over five vague ones. Licensing: note each donor repo's license and flag any
copyleft that would prevent reimplementing the idea.
