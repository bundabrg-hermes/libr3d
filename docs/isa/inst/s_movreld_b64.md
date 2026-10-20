---
mnemonic: S_MOVRELD_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "67"
pages: [235]
source: RDNA3 ISA PDF p.235-235
---

# S_MOVRELD_B64 (opcode 67)

Move data from a scalar input into a relatively-indexed scalar register. The index in M0.u and the operand address in DST.u must be even for this operation.
