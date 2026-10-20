---
mnemonic: S_MOVRELS_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "65"
pages: [234]
source: RDNA3 ISA PDF p.234-234
---

# S_MOVRELS_B64 (opcode 65)

Move data from a relatively-indexed scalar register into another scalar register. The index in M0.u and the operand address in SRC0.u must be even for this operation.
