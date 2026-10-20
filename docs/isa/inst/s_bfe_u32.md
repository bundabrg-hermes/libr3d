---
mnemonic: S_BFE_U32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "38"
pages: [205]
source: RDNA3 ISA PDF p.205-205
---

# S_BFE_U32 (opcode 38)

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store the result into a scalar register and set SCC iff the result is nonzero.
