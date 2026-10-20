---
mnemonic: S_ASHR_I32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "12"
pages: [200]
source: RDNA3 ISA PDF p.200-200
---

# S_ASHR_I32 (opcode 12)

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.
