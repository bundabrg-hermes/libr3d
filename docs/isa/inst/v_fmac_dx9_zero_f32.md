---
mnemonic: V_FMAC_DX9_ZERO_F32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "6"
pages: [260]
source: RDNA3 ISA PDF p.260-260
---

# V_FMAC_DX9_ZERO_F32 (opcode 6)

Multiply two single-precision values and accumulate the result with the destination. Follows DX9 rules where 0.0 times anything produces 0.0 (this is not IEEE compliant).
