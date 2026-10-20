---
mnemonic: S_LSHL_B32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "8"
pages: [199]
source: RDNA3 ISA PDF p.199-199
---

# S_LSHL_B32 (opcode 8)

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.
