---
mnemonic: V_FMAC_DX9_ZERO_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "262"
pages: [402]
source: RDNA3 ISA PDF p.402-402
---

# V_FMAC_DX9_ZERO_F32 (opcode 262)

Multiply two single-precision values and accumulate the result with the destination. Follows DX9 rules where 0.0 times anything produces 0.0 (this is not IEEE compliant).
