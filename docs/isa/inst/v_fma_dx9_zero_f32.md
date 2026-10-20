---
mnemonic: V_FMA_DX9_ZERO_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "521"
pages: [414]
source: RDNA3 ISA PDF p.414-414
---

# V_FMA_DX9_ZERO_F32 (opcode 521)

Multiply and add single-precision values. Follows DX9 rules where 0.0 times anything produces 0.0 (this is not IEEE compliant).
