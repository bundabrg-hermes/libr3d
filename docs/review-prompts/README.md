# External Model Review Prompts

Two independent read-only review passes to be run by frontier AI models with no
access to target hardware. Every pass gates on `docs/COVERAGE.md` (the ledger of
what has been implemented, tried, or deliberately deferred) and demands a strict
YAML output so results can be fed back to the maintainer agent for hardware
validation.

| Pass | File | Question asked |
|------|------|----------------|
| 1 | `01-isa-review.md` | Does the RDNA3 ISA offer cheaper instructions than what the hot loops use? |
| 2 | `02-cross-engine-review.md` | Do proven kernels in other inference engines do the same job with a better structure? |

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
