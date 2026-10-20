---
mnemonic: V_MUL_DX9_ZERO_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "263"
pages: [403]
source: RDNA3 ISA PDF p.403-403
---

# V_MUL_DX9_ZERO_F32 (opcode 263)

Multiply two floating point inputs and store the result in a vector register. Follows DX9 rules where 0.0 times anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).
