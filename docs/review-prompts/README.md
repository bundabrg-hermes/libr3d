# External Model Review Prompts

Three independent read-only review passes (plus a fourth verifier pass, run on
completed submissions) to be run by AI models with no access to target
hardware. Every pass gates on `docs/COVERAGE.md` (the ledger of
what has been implemented, tried, or deliberately deferred) and `docs/REVIEW_QUEUE.md`
(the live adjudicated backlog) and demands a strict
YAML output so results can be fed back to the maintainer agent for hardware
validation.

| Pass | File | Question asked |
|------|------|----------------|
| 1 | `01-isa-review.md` | Does the RDNA3 ISA offer cheaper instructions than what the hot loops use? |
| 2 | `02-cross-engine-review.md` | Do proven kernels in other inference engines do the same job with a better structure? |
| 3 | `03-exhaustive-cross-engine-review.md` | Broadest evidence-grounded backlog: is any family prematurely considered exhausted? (superset of pass 2; budget-capped output) |
| 4 | `04-citation-verifier.md` | (Verifier role) Mechanically fact-check a completed submission: do the cited instructions, anchors, and ledger quotes exist? Constrained enough for small/local models. |

## Model fitness (evidence-based, revised 2026-10-27)

A reviewer meta-review claimed small/local models are unfit for cross-engine
(pass 2/3) work because they fabricate citations and mis-anchor targets. Our
own submission record contradicts that as a blanket rule: the strongest pass-2
submission of its round came from a locally-quantized model (QFN), with donor
rationale comments quoted verbatim and zero fabricated citations (defects were
one stale anchor and one target-arithmetic slip). What is supported: passes
that demand long-horizon portfolio reasoning (pass 3 Step 0) and long schema
discipline across a superset output are where frontier models have earned
their keep, and every truncation/schema failure so far came from long outputs,
which are worse on small contexts.

Working policy:

- **Frontier models: pass 3** (including Step 0 gap attribution) as the
  primary review, pass 1 only for finishing selected kernels.
- **Local/small models: pass 1, pass 2, and pass 4.** The prompts convert
  research into retrieval-plus-citation, which quantized models handle well;
  run them free, not not.
- **Pass 4 is advisory triage only.** It pre-flags mechanical defects before
  the maintainer's desk-check; it never approves a proposal past the desk.
  Its own verdicts get spot-audited like any submission.

Scope policy (adopted 2026-10-27 on reviewer meta-review, endorsed by the
maintainer after checking the submission record): pass 1 is a kernel-FINISHING
tool, not a discovery engine. History supports this - the largest wins of the
last week (Q30 env vars, the spill census, the Q1 barrier-idiom family) came
from pass 2 donors, third-party triage, and measurement attribution, while
recent pure-ISA pass-1 rounds returned flat-to-modest results (with Q32 the
standing counterexample). Do not run further pass-1 rounds to choose the next
major optimization; use the Step 0 gap-attribution stage in pass 3 for that,
and spend pass 1 only polishing kernels already selected for work.

Usage:

1. Give the model this repo (clone or file access to the GitHub fork
   `bundabrg-hermes/libr3d`), not the radiance engine repo. Kernel contracts
   live in `src/r3d_gemm.h` / `src/r3d_attn.h`.
2. Paste one prompt (pass 1 and pass 2 should be separate model sessions;
   combining them dilutes both).
3. The model must return exactly one fenced YAML block. Feed it back to the
   maintainer session unmodified.
4. Optional before sending anything to the maintainer: run pass 4 on a
   submission with any cheap local model to pre-flag fabricated mnemonics,
   dead anchors, and contradicted ledger quotes. Fast, free triage — but the
   maintainer desk-check runs regardless of the verifier's verdicts.

Proposals that survive desk-checking are tracked in `docs/REVIEW_QUEUE.md`
(the ordered backlog). If an idea is already queued there, only propose it
as a concrete refinement.

Maintainer-side validation (not the reviewer's job):

- Pass 1: every `isa_basis` mnemonic is cross-checked against `docs/isa/`
  (and `docs/isa.db` if present); low-confidence items get a hardware probe
  before any benchmark slot is spent.
- Pass 2: every `donor_ref` file:line is opened and read; RDNA3-proven donors
  are ported first, CUDA-only idioms are re-validated as gfx11 ideas only.
