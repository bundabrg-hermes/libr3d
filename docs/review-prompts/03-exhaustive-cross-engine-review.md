# Pass 3 — Exhaustive Cross-Engine Optimization Opportunity Audit

You are performing an **exhaustive, read-only optimization opportunity audit** of `libr3d`, an out-of-tree HIP GPU kernel library used by the Radiance inference engine.

Your job is to identify, investigate, rank, and document as many credible kernel-level optimization opportunities as possible by comparing the real implementation in `libr3d` against proven inference engines and other relevant public GPU kernels.

The objective is not to produce a short list of safe suggestions. It is to discover performance gaps that earlier reviews may have missed, including substantial structural differences in memory access, quantized arithmetic, work distribution, and data movement.

The final deliverable will be passed to a local coding worker that can investigate and implement the opportunities individually. Every recommendation must therefore be concrete enough to turn into a separate engineering task.

**Do not modify source code, write tests, compile, execute GPU code, run benchmarks, or claim performance improvements that have not been measured.** This is research and static code review only. You may inspect public repositories, source files, documentation, existing benchmarks, recorded results, and instruction-set documentation.

---

## 1. Project and hardware context

**Primary repository:** https://github.com/bundabrg-hermes/libr3d

The project provides custom HIP kernels for the Radiance inference engine, targeting AMD RDNA3, particularly `gfx1100` and `gfx1101`.

Relevant kernel families include:

- GEMM and GEMV: `src/r3d_gemm_*.hip`
- Paged attention: `src/r3d_attn_paged_*.hip`
- RMSNorm
- RoPE
- Activation multiplication
- KV-cache storage
- Embedding
- Sampling

Locate the actual files and functions in the current checkout; the filenames above are indicative rather than an exhaustive inventory.

### Mandatory first step: read the entire coverage ledger

Read `docs/COVERAGE.md` **and `docs/REVIEW_QUEUE.md`** **in full before forming recommendations**.

`docs/REVIEW_QUEUE.md` is the live adjudication queue for external review submissions; proposals that duplicate a queue row (any status, including `gated` or `parked`) are rejected unless they add a materially new mechanism to it.

This is the canonical record of work already implemented, attempted, deferred, measured, rejected, or promoted to default.

Extract its relevant contents into an internal coverage map, including:

- Implemented algorithms and layouts.
- Kernel dispatch conditions and shape-specific paths.
- Successful and unsuccessful experiments.
- Recorded benchmark numbers and measurement conditions.
- Correctness and precision limitations.
- Known regressions, crossover points, and deferred work.
- Existing limitations and suspected bottlenecks.

Read relevant libr3d kernel implementations after reading the ledger. Do not infer the code's structure from filenames, comments, documentation, or previous review summaries.

If the ledger references other documents that explain an optimization, consult those documents as needed.

**The coverage ledger is a duplicate-detection mechanism, not a restriction on the depth of the search.**

An idea that substantially refines an existing optimization is eligible. Explain precisely what changes relative to the existing implementation and why the refinement might help.

Do not reject an entire optimization family merely because one version of it has already been tried.

---

## 2. Ground truth: preserve these facts

The following information has been verified by the project maintainer. Accept it as ground truth; do not spend the review trying to disprove it.

### Hardware and ISA

- `gfx1100` has working `v_dot4_i32_iu8` and `v_sudot4` instructions.
- RDNA3 in this target configuration does not have usable AGPR/accvgpr accumulators.
- RDNA3 WMMA uses ordinary VGPRs for accumulation.
- RDNA3 is not CDNA and must not be treated as if it has CDNA matrix cores.
- There are no warp-shuffle capabilities beyond those documented in the project's `docs/isa/` corpus.
- The instruction documentation bundled under `docs/isa/` is the authoritative basis for evaluating translated instruction-level proposals.

### Performance and workload

### Correctness law: bit-exactness

libr3d kernels are gated on matching the engine's reference outputs under the project's established precision laws, and reduction-tree shapes are part of that contract. Any proposal that changes summation order, reduction-tree topology, rounding, or scale-application placement is a **precision-visible change** and must be explicitly flagged as such in its `risks` with the specific values/accumulation order that change. Proposals that preserve instruction-level reduction order are strongly preferred.

### Performance and workload

- Decode is memory-bandwidth-bound.
- Deep-context KV attention currently achieves approximately **171 GB/s effective bandwidth**.
- A previous microbenchmark on an RX 7900 XTX reported that hipfire's quantized GEMV approach achieved approximately **2.5–3.5× the effective GB/s of the earlier libr3d implementation** on comparable MXFP4/Q8x shapes. Status update: libr3d's Q8x path has since been promoted default-on in-engine; establish the gap against the *current* shipped kernel, not the pre-promotion implementation the microbenchmark measured.
- Treat this result as an important lead, not as proof that the current implementation still has the same gap. Establish the exact comparison, measurement conditions, and correspondence to the current source from available evidence. Do not silently dismiss the result because other optimizations have since been implemented.

### Supported dtype universe

Treat these as the supported activation/weight formats:

`{bf16, f32, fp8a8 (128-block scales), i8a8, mxfp4a8 (E8M0/32-block), w2a8}`

Other quantization formats may provide useful structural ideas, but do not propose introducing a new format unless explicitly justified as a separate scope change.

For example, NVFP4 or MXFP8 implementations may reveal a better way to pack data, map lanes, accumulate partial sums, or apply scales. Translate those ideas to the supported formats rather than treating the formats themselves as drop-in replacements.

---

## 3. Reference engines and source discovery

Investigate these repositories first, but do not limit yourself to them.

### A. Strata

Repository: https://github.com/Niko1221/Strata

Strata is an RDNA3-targeted inference engine and is therefore a particularly valuable source of architecture-relevant implementations.

Examine the actual kernel source for:

- Quantized GEMV and GEMM.
- Weight layouts and packing.
- Lane-to-element and wave-to-output mappings.
- Vector widths and memory transaction structure.
- Multi-row output processing.
- Scale handling and integer accumulation.
- WMMA use and fallback paths.
- Attention and KV-cache processing, where implemented.
- Prefetching, LDS use, and work scheduling.

Do not use an architecture label or a README claim as evidence that an individual technique is effective. Find the implementation and establish the applicable execution path.

### B. hipfire

Repository: https://github.com/Ben1078/hipfire

Treat hipfire's quantized GEMV implementation as a high-priority reverse-engineering target because of the historical bandwidth comparison above.

Reconstruct the complete relevant data path:

1. Input weight representation and address calculations.
2. Thread/lane ownership of packed bytes and values.
3. Global-memory loads, vector widths, alignment, and reuse.
4. Quantized unpacking and conversion.
5. Dot-product instruction selection.
6. Integer accumulator organization and width.
7. Scale application and partial-sum reduction.
8. Output layout and store pattern.
9. Dispatch conditions and shape-dependent alternatives.

Compare these structures directly against the corresponding current libr3d implementation.

The aim is to explain the performance difference, not merely to list assembly instructions.

If a reliable comparison cannot be attributed to a particular source version, report that limitation explicitly.

### C. llama.cpp

Repository: https://github.com/ggml-org/llama.cpp

Inspect the actual implementations under `ggml/src/ggml-cuda/`, particularly:

- Per-quantization `vec_dot` implementations and inline assembly.
- MMVQ/MMQ kernel families.
- Quantized GEMV/GEMM mappings.
- Weight packing and conversion.
- Scale application and reduction.
- Matrix tile organization.
- Split-K and other work-distribution approaches, where applicable.
- Fused epilogues and data reuse.

CUDA implementations are useful as structural references, but **CUDA idioms must never be copied directly into an RDNA3 proposal**.

For each translated idea, identify the relevant `gfx11` instruction semantics or ISA constraints in `libr3d/docs/isa/` and describe how the structure would map to RDNA3. Do not assume CUDA warp size, shuffle behavior, accumulator storage, or matrix instructions have AMD equivalents.

### D. Additional donors

Find other useful public implementations where they can contribute genuinely different ideas.

Potential sources include:

- MLC-LLM and its GPU backends.
- tinygrad GPU kernels.
- Vulkan compute and matrix-multiplication projects.
- Other inference engines with AMD-native kernels.
- Public implementations of paged attention and quantized matrix-vector products.
- Public GPU kernels for similar operations from other frameworks.

Choose donors according to implementation relevance rather than the number of repositories searched.

For every donor considered, identify its repository, exact source path, applicable revision, and license. A donor from an unfamiliar engine is welcome if its implementation is relevant and its source is accessible.

---

## 4. Search for optimization opportunities systematically

Review each family below against the actual libr3d implementation. These are investigation prompts, not a list of assumptions that every feature is missing.

### A. Quantized GEMV and decode

Prioritize small-M, large-K operations typical of inference decode.

Investigate:

- Alternative packed-weight layouts and byte ordering.
- Interleaving strategies and their effect on coalescing, cache-line utilization, and load instruction count.
- Lane ownership of quantized values and multiple output rows.
- Whether a thread or lane computes too few or too many outputs.
- Vector width, aligned versus unaligned loads, and redundant memory transactions.
- Number and arrangement of independent accumulators.
- Integer accumulation width, partial sums, and scale-application placement.
- Packing/unpacking overhead and opportunities for fused conversion.
- Reuse of activations, weights, scale values, or lookup data.
- How the implementation handles irregular K values and quantization-block boundaries.
- Small-M versus medium-M crossover points.
- Register use, occupancy, and instruction-level parallelism implied by the code.
- Persistent work assignment or multi-row GEMV variants.
- Distinct algorithms for M=1, small M, and larger M where useful.

Explicitly compare hipfire and Strata's actual implementation with the current libr3d functions.

A useful finding should state which loads, lanes, accumulators, or data layouts differ and how that difference could explain a performance gap.

### B. GEMM and tall-K, skinny-M workloads

Investigate:

- Tile dimensions and work assignment.
- Shape-dependent dispatch.
- WMMA and non-WMMA crossover behavior.
- Split-K, stream-K, and other ways of exposing more independent work.
- Reduction cost and temporary-buffer traffic.
- Partial-sum accumulation and epilogue handling.
- Persistent scheduling and work-queue design.
- Row and column tiling for skinny matrices.
- Weight reuse, activation reuse, and scale reuse.
- Register pressure and LDS/shared-memory tradeoffs.
- Whether output tiles expose enough concurrency to saturate the GPU.
- Whether a different mapping is appropriate for each supported dtype.

A proposal must compare against the current implementation and account for existing split or tiled paths in the coverage ledger. Do not simply recommend split-K because it is common in GEMM implementations.

### C. Paged attention and KV-cache reads

Investigate:

- KV page traversal order.
- Block iteration, vector widths, and alignment.
- Lane ownership of keys, values, and reduction dimensions.
- Address-generation and page-table overhead.
- Masking and boundary handling.
- Redundant loads or repeated index calculations.
- Intermediate reductions and synchronization.
- Online softmax structure and opportunities to reduce intermediate work.
- Split-KV scheduling and combination costs.
- Parallelism at different context lengths and head dimensions.
- Register pressure, LDS footprint, and occupancy.
- Memory coalescing and useful bytes transferred relative to total traffic.
- Attention kernel dispatch and alternative variants.
- Opportunities to fuse compatible KV operations without increasing total data movement.

Account for the already optimized attention variants and their recorded performance. In particular, compare any proposed split-KV or WMMA refinement against the existing path applicable to that geometry.

Do not infer that 171 GB/s proves a particular bottleneck. Explain which code structure could cause the inefficiency and what existing evidence supports that hypothesis.

### D. Fused kernels and inter-kernel data movement

Look beyond each individual kernel.

Investigate possible fusion of:

- RMSNorm and adjacent compatible operations.
- RoPE with KV writes or other compatible producers.
- Activation multiplication with preceding or following operations.
- Quantization or format conversion with data producers.
- Reduction epilogues with the next consumer.
- Embedding or sampling operations with compatible neighboring processing, where meaningful.

For each potential fusion, determine whether it could eliminate a global-memory read/write, an intermediate buffer, an extra kernel launch, or redundant conversion.

Account for register pressure, occupancy, synchronization, numerical differences, and whether the producer and consumer actually have compatible data ownership and execution geometry.

A fusion proposal without a plausible reduction in total work or data movement should not be promoted.

### E. Memory movement, prefetch, and pipelining

Investigate:

- Software-pipeline depth.
- Outstanding independent loads.
- Load scheduling relative to arithmetic.
- Prefetch distance and loop structure.
- Buffering and double-buffering.
- LDS staging and bank conflicts.
- Cache behavior implied by access patterns.
- Address-calculation overhead.
- Dependency chains that delay consumption of loaded data.
- Whether a proposed pipeline can overlap useful memory latency without excessive VGPR usage.

The coverage ledger includes prior experiments. Treat their measured outcomes seriously, but do not generalize a failed experiment to every kernel geometry or every different pipeline structure.

A change to a previously tested technique is eligible if the donor's actual implementation provides a specific reason that the new structure should behave differently.

### F. Dispatch and heterogeneous kernel families

Review the dispatch structure as well as the individual kernels.

Look for:

- Geometries with a poor fit to a generic kernel.
- Conditions where a specialized path is enabled too broadly or too narrowly.
- Missing specializations for particular supported dtype and shape combinations.
- Multiple kernels that could share common infrastructure without sacrificing performance.
- Existing specializations whose dispatch thresholds could be refined based on already recorded measurements.
- Opportunities to avoid expensive general-case handling on common shapes.
- Gaps between what the dispatch logic permits and what the kernel implementation supports.

Do not change thresholds based on invented estimates. Label any proposed threshold adjustment as requiring future measurement.

### G. Numerical, compiler, and portability constraints

For all structural candidates, consider:

- Exact accumulator-width requirements.
- Overflow bounds for integer dot products.
- Floating-point accumulation order and rounding.
- BF16 and FP8 behavior.
- MXFP4 scale-block boundaries.
- Quantization tails and non-multiple dimensions.
- Alignment assumptions.
- VGPR and LDS requirements.
- Wave-size assumptions.
- `gfx1100` versus `gfx1101` applicability.
- Correctness of partial reductions.
- Differences between theoretical instruction availability and demonstrated compiler support.

Do not propose an instruction unsupported by the target simply because another engine uses it.

Do not assume that code compiling for another AMD generation establishes correctness or performance on RDNA3.

---

## 5. Method: establish the gap before proposing the fix

For every promising technique, use this sequence.

**Step 1 — Understand libr3d.**

Identify the exact kernel function, dispatch conditions, relevant code, and current memory/arithmetic structure.

**Step 2 — Understand the donor.**

Find the actual implementation, not merely a README, high-level description, or issue comment. Follow helper functions or data-layout definitions when necessary to understand what the kernel really does.

**Step 3 — Identify the structural difference.**

Write down what the donor does differently from libr3d.

Examples include:

- One packed load instead of several narrower loads.
- Different lane ownership of packed quantized values.
- A different integer partial-sum arrangement.
- Different scale application.
- More outputs per thread or wave.
- Different KV block scheduling.
- Fewer intermediate global-memory transfers.
- A different number of partial reductions.
- More independent work without materially increasing resource use.

These are examples only. Identify the difference from source, rather than selecting a familiar technique and trying to justify it afterward.

**Step 4 — Check existing coverage.**

Search the entire coverage ledger and relevant source for the proposed structure. Classify the relationship as:

- `missing`: no equivalent implementation or experiment was found.
- `refinement`: an existing implementation exists, but this is a specific structural improvement.
- `previously_tried`: an equivalent experiment exists, with its recorded outcome.
- `already_present`: the donor and libr3d already use materially the same structure.
- `uncertain`: the available evidence is insufficient to decide.

Do not reject `refinement` automatically. For `previously_tried`, identify what is materially different from the failed or inconclusive attempt.

**Step 5 — Build a causal performance hypothesis.**

Explain why the difference could help the target geometry, considering memory traffic, instruction count, dependency structure, occupancy, and parallelism as applicable.

State the expected direction of the effect and the circumstances under which it could regress.

Do not invent a percentage speedup or claim an implementation is faster unless trustworthy measurements support that conclusion.

**Step 6 — Describe how it could be evaluated later.**

Specify the smallest useful A/B comparison, the relevant existing baseline, and the shapes or workloads that could distinguish the hypothesis from alternatives.

Do not run that comparison during this review.

---

## 6. Evidence and citation requirements

Traceability is mandatory. Avoid unsupported recommendations.

### libr3d evidence

For every candidate, cite the actual target implementation using `path:function` and line ranges whenever available.

Identify the relevant section or recorded result in `docs/COVERAGE.md`, preferably with a line range. Explain what makes the proposal different from that existing work.

### Donor evidence

For every donor-derived candidate, supply:

- Repository and owner.
- Exact source file path.
- Exact relevant line range.
- Pinned commit SHA, or an explicitly identified branch if a commit cannot be established.
- Public source URL where available.
- Repository license and any applicable file-specific license.
- Whether the donor's implementation is used on RDNA/AMD hardware, CUDA, or another architecture.
- Evidence supporting the architecture classification.

Prefer a pinned revision because line ranges on a moving branch can become invalid.

A directory listing, README, issue, benchmark headline, or general claim that an engine is RDNA-native is not a substitute for the implementation citation. Such material can provide context but should be cited separately from the source implementing the technique.

### Translated techniques

If the donor is CUDA or another architecture, identify the relevant `gfx11` instruction or architectural property that makes the translation plausible. Cite the project's ISA documentation and distinguish:

- The concept that transfers.
- The architecture-specific implementation that must change.
- Any behavior that is not equivalent between architectures.

If a useful candidate has no suitable donor, do not fabricate a donor citation. Put it in a clearly separate `first_principles_candidates` section, explain the source and reasoning behind the hypothesis, and mark it as not satisfying the cross-engine evidence requirement.

### Licensing

Record the actual repository license for every donor used. Check file-level licensing or notices where relevant.

Identify any copyleft or other licensing implications that could constrain source reuse. Distinguish implementing an independently derived algorithmic idea from copying or adapting donor code. Do not assume that all files in a repository have the same license without checking.

### Incomplete evidence

When a repository, file, commit, or source range cannot be verified, say so.

You may retain a plausible candidate with reduced confidence if the uncertainty is explicit and the next research step is concrete. Missing citations should not silently become fabricated facts, and uncertainty alone should not force every other candidate out of the review.

---

## 7. Required output

Return **one fenced YAML block only**, with no prose outside it.

The YAML is an engineering backlog, not just an essay. Keep entries self-contained so that a local worker can take one candidate without having to reconstruct the entire review.

Use the following schema. Additional fields are welcome when useful.

```yaml
review:
  title: "Pass 3 - Exhaustive Cross-Engine Optimization Opportunity Audit"
  repository: "https://github.com/bundabrg-hermes/libr3d"
  reviewed_commit: "..."
  coverage_ledger:
    read_in_full: true
    relevant_sections:
      - section: "..."
        evidence: "..."
  source_scope:
    libr3d_files_reviewed:
      - "path/to/file.hip:functions"
    donor_repositories_examined:
      - repo: "..."
        revision: "..."
        license: "..."
        architecture: "..."
        source_files_reviewed:
          - "..."
  limitations:
    - "Specific repositories, sources, or claims that could not be verified."

opportunities:
  - id: P3-001
    title: "Concise structural description"
    priority: P0
    category: "quantized_gemv"
    evidence_level: high
    confidence: medium
    effort: M

    target:
      file: "src/..."
      functions:
        - "..."
      geometry: "Exact or approximate shape class"
      dtype: "Supported dtype and quantization path"

    current_libr3d_structure:
      description: "What the code does now."
      evidence:
        - "src/path.hip:function:line-range"
      coverage_status: "missing | refinement | previously_tried | uncertain"
      coverage_evidence:
        - "docs/COVERAGE.md:section or line-range"
      previous_attempts:
        - "What was tried, what happened, and how this proposal differs."

    donor:
      repo: "owner/repository"
      revision: "commit SHA or explicitly named branch"
      file: "path/to/implementation"
      lines: "START-END"
      url: "Public source URL"
      license: "Verified license"
      runs_on: "RDNA/AMD | CUDA | other | unknown"
      architecture_evidence: "Evidence for the classification"

    structural_difference:
      description: "Exactly what differs between the donor and libr3d."
      donor_mechanism: "How the donor implementation works."
      relevant_isa:
        - "gfx11 instruction or architectural property"
      isa_citations:
        - "docs/isa/path:section or line reference"

    transfer_plan:
      - "Specific structural change to make in the named kernel."
      - "How lane ownership, memory access, accumulation, or scheduling would change."
      - "Required new specialization or dispatch conditions, if any."
      - "What must remain unchanged for correctness."

    performance_hypothesis:
      bottleneck: "Memory-bound | instruction-bound | latency-bound | occupancy-bound | launch-bound | uncertain"
      causal_argument: "Why the structural change could help."
      expected_effect: "Directional expectation, not an invented speedup."
      applicable_shapes: "Where the idea is expected to help."
      regression_shapes: "Where it may lose and why."
      measured_evidence: "Actual donor measurements or existing project results, with context; otherwise state none."

    validation_plan:
      baseline: "Existing coverage result and exact relevant configuration."
      comparison: "Current kernel versus proposed structure."
      geometries: "Shapes and dtype combinations that would discriminate the hypothesis."
      correctness_gates: "Relevant precision, boundary, and output requirements."
      performance_metric: "Relevant kernel and/or end-to-end metric."
      interpretation: "What result would support or reject the hypothesis."
      execution_status: "Not run; research-only review."

    risks:
      - "Correctness, precision, resource use, portability, licensing, or dispatch risks."
    prerequisites:
      - "Missing information or prerequisite investigation, if any."
    next_action: "One concrete task for the local coding worker."

first_principles_candidates:
  - id: P3-FP-001
    title: "Candidate without a verified donor implementation"
    target: "..."
    rationale: "..."
    evidence: ["Relevant libr3d source and existing project data."]
    uncertainty: "Why this is plausible but not established."
    next_research_step: "How to find or validate suitable evidence."
    next_action: "Concrete follow-up task."

rejected_comparisons:
  - donor: "..."
    source: "path:lines @ revision"
    technique: "..."
    disposition: "already_present | duplicate_experiment | incompatible | no_clear_advantage | insufficient_evidence"
    evidence: "Actual donor and libr3d evidence."
    explanation: "Why it should not currently become a separate implementation task."
    reconsider_if: "Condition under which a distinct refinement would become viable."

no_op_findings:
  - area: "..."
    donor: "..."
    evidence: "..."
    conclusion: "Where libr3d already matches the donor or why transfer is not useful."

coverage_gaps:
  - area: "..."
    description: "Relevant kernel or workload not sufficiently covered by the existing ledger."
    evidence: "Specific evidence for the gap."
    recommended_research: "What to inspect next."

research_followups:
  - id: P3-R-001
    title: "Remaining source-level question"
    priority: P1
    question: "What remains unresolved?"
    files_to_inspect: ["..."]
    expected_outcome: "What answering this could establish."

validation_notes:
  - "Prioritized future A/B instructions."
```

Replace all illustrative placeholders with real findings. Do not return the schema with empty lists as a substitute for completing the research.

**Output budget (hard cap):** at most **15** entries in `opportunities`, ranked by practical value — deeper investigation of the top entries beats breadth beyond the cap; anything past 15 goes to `research_followups` as one-liners instead. Keep `rejected_comparisons`, `no_op_findings`, and `coverage_gaps` to one-to-three lines each (citation plus disposition). If your response is truncated, the most valuable content is lost — write the YAML so that completeness of the document outweighs depth of any single entry.

---

## 8. Priority and confidence definitions

Prioritize candidates by **expected practical value**, not novelty or ease of implementation.

Use these priorities:

- `P0`: High-value opportunity or large unexplained performance gap deserving investigation first.
- `P1`: Credible, potentially material improvement with a reasonably clear transfer path.
- `P2`: Narrower improvement, workload-specific opportunity, or more speculative structural hypothesis.
- `P3`: Low-confidence research lead worth retaining but not yet ready for implementation.

Use these evidence levels:

- `high`: Exact source-level structural difference and clear applicability to the target kernel, with substantial architectural uncertainty already resolved.
- `medium`: A real structural difference has been identified, but transfer, applicability, or expected benefit remains uncertain.
- `low`: A plausible hypothesis with meaningful unresolved questions.

Confidence must be separate from evidence level. A well-documented donor technique can still have low confidence of improving libr3d if the bottleneck or shape correspondence is uncertain.

Effort is the estimated implementation and integration work, not the time needed to perform this review:

- `S`: Localized modification or specialization.
- `M`: Significant kernel restructuring or dispatch change.
- `L`: Major redesign, new kernel family, significant reduction/data-flow changes, or broad integration work.

Do not rank candidates by speculative speedup percentages. Explain the likely benefit and the assumptions on which the ranking depends.

Aim for broad coverage and multiple genuinely different opportunities where the evidence supports them. Do not set an arbitrary minimum number of proposals, and do not inflate the backlog with trivial changes merely to make it longer.

---

## 9. Rules against premature rejection

These rules are particularly important.

1. **Do not stop at the first matching technique in the coverage ledger.** If a donor has an already-implemented technique, inspect whether it has another meaningful difference in layout, mapping, vectorization, reduction, or scheduling.

2. **Do not treat an entire family as exhausted because one variant failed.** A failed prefetch distance, vector width, tile size, or scheduling strategy does not rule out all alternatives in that family.

3. **Do not mistake the same algorithm name for the same implementation.** Two GEMV kernels may both use integer dot products but differ substantially in byte packing, load coalescing, lane ownership, accumulator organization, or scale handling.

4. **Do not declare a candidate redundant without source evidence.** Identify the corresponding existing implementation or experiment and explain the equivalence. Similar terminology is not enough.

5. **Do not claim a performance gain merely because the donor is faster overall.** Explain the relevant structural difference, target bottleneck, and shape correspondence.

6. **Do not assume an undocumented bottleneck.** Label hypotheses as hypotheses and provide the specific code pattern that motivates them.

7. **Do not let missing donor evidence erase useful leads.** Move these to `first_principles_candidates` or `research_followups` rather than pretending the opportunity does not exist.

8. **Do not equate theoretical maximum bandwidth with expected achievable bandwidth.** Use the project's actual workload measurements and hardware limits only when they are supported by the available evidence.

9. **Do not recommend a new dtype, unsupported instruction, or architecture-specific feature outside the established target merely because a donor uses it.**

10. **Do not copy donor source code into libr3d or create patches.** Describe the structural change and record licensing implications for the downstream worker.

---

## 10. Research completion criteria

Before finalizing the YAML:

- Confirm that `docs/COVERAGE.md` was read in full.
- Inspect the actual libr3d implementation for every proposed target.
- Investigate hipfire's relevant GEMV source in enough detail to describe its memory-access and accumulation structure.
- Inspect real Strata implementations rather than relying on documentation summaries.
- Inspect relevant llama.cpp CUDA code and translate any proposed structures through the documented RDNA3 ISA.
- Review additional donors where they offer distinct, relevant implementations.
- Verify exact source locations, revisions, and licenses for each donor-derived proposal.
- Search for evidence of equivalent functionality in libr3d before classifying a candidate as missing.
- Preserve the existing measured baselines and failed experiments.
- Include promising uncertainties in the research backlog rather than suppressing them.
- Check that every entry has an actionable next step and that the output is valid YAML.

If a repository is inaccessible or a source claim cannot be verified, report that fact and continue with the remaining research. Do not fabricate line ranges, commits, measurements, or license information.

**Final objective:** produce the broadest useful and evidence-grounded optimization backlog possible for libr3d, with the clearest unexplained performance gaps investigated first. The audit succeeds by identifying real differences worth pursuing and explaining why they might matter—not by proving that no further optimization paths exist.

Remember: no source changes, tests, compilation, execution, or benchmarking during this pass.
