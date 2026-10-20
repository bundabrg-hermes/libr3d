---
mnemonic: S_BFE_I64
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "41"
pages: [206]
source: RDNA3 ISA PDF p.206-206
---

# S_BFE_I64 (opcode 41)

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the result into a scalar register and set SCC iff the result is nonzero.
