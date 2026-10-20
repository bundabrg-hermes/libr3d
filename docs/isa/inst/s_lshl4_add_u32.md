---
mnemonic: S_LSHL4_ADD_U32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "17"
pages: [201]
source: RDNA3 ISA PDF p.201-201
---

# S_LSHL4_ADD_U32 (opcode 17)

Calculate the logical shift left of the first input by 4, then add the second input, store the result into a scalar register and set SCC iff the summation results in an unsigned overflow.
