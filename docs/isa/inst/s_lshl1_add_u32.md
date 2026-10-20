---
mnemonic: S_LSHL1_ADD_U32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "14"
pages: [200]
source: RDNA3 ISA PDF p.200-200
---

# S_LSHL1_ADD_U32 (opcode 14)

Calculate the logical shift left of the first input by 1, then add the second input, store the result into a scalar register and set SCC iff the summation results in an unsigned overflow.
