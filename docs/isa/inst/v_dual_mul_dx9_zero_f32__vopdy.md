---
mnemonic: V_DUAL_MUL_DX9_ZERO_F32
section: "16.11.2. VOPD Y-Instructions"
format: VOPDY
opcode: "7"
pages: [371]
source: RDNA3 ISA PDF p.371-371
---

# V_DUAL_MUL_DX9_ZERO_F32 (opcode 7)

Multiply two floating point inputs and store the result in a vector register. Follows DX9 rules where 0.0 times anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).
