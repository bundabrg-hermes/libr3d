---
mnemonic: S_XOR_B64
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "27"
pages: [203]
source: RDNA3 ISA PDF p.203-203
---

# S_XOR_B64 (opcode 27)

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is nonzero.
