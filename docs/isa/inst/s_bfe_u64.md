---
mnemonic: S_BFE_U64
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "40"
pages: [206]
source: RDNA3 ISA PDF p.206-206
---

# S_BFE_U64 (opcode 40)

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store the result into a scalar register and set SCC iff the result is nonzero.
