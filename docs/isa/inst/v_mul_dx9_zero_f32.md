---
mnemonic: V_MUL_DX9_ZERO_F32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "7"
pages: [260]
source: RDNA3 ISA PDF p.260-260
---

# V_MUL_DX9_ZERO_F32 (opcode 7)

Multiply two floating point inputs and store the result in a vector register. Follows DX9 rules where 0.0 times anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).
