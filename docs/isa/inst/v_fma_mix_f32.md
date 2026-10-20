---
mnemonic: V_FMA_MIX_F32
section: "16.10. VOP3P Instructions"
format: VOP3P
opcode: "32"
pages: [366]
source: RDNA3 ISA PDF p.366-366
---

# V_FMA_MIX_F32 (opcode 32)

Fused-multiply-add of single-precision values with MIX encoding. Size and location of S0, S1 and S2 controlled by OPSEL: 0=src[31:0], 1=src[31:0], 2=src[15:0], 3=src[31:16]. Also, for FMA_MIX, the NEG_HI field acts instead as an absolute-value modifier.
