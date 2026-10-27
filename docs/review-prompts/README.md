# External Model Review Prompts

Three independent read-only review passes to be run by frontier AI models with no
access to target hardware. Every pass gates on `docs/COVERAGE.md` (the ledger of
what has been implemented, tried, or deliberately deferred) and `docs/REVIEW_QUEUE.md`
(the live adjudicated backlog) and demands a strict
YAML output so results can be fed back to the maintainer agent for hardware
validation.

| Pass | File | Question asked |
|------|------|----------------|
| 1 | `01-isa-review.md` | Does the RDNA3 ISA offer cheaper instructions than what the hot loops use? |
| 2 | `02-cross-engine-review.md` | Do proven kernels in other inference engines do the same job with a better structure? |
| 3 | `03-exhaustive-cross-engine-review.md` | Broadest evidence-grounded backlog: is any family prematurely considered exhausted? (superset of pass 2; budget-capped output) |

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

Proposals that survive desk-checking are tracked in `docs/REVIEW_QUEUE.md`
(the ordered backlog). If an idea is already queued there, only propose it
as a concrete refinement.

Maintainer-side validation (not the reviewer's job):

- Pass 1: every `isa_basis` mnemonic is cross-checked against `docs/isa/`
  (and `docs/isa.db` if present); low-confidence items get a hardware probe
  before any benchmark slot is spent.
- Pass 2: every `donor_ref` file:line is opened and read; RDNA3-proven donors
  are ported first, CUDA-only idioms are re-validated as gfx11 ideas only.
