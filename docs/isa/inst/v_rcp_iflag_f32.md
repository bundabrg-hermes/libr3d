---
mnemonic: V_RCP_IFLAG_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "43"
pages: [284, 285]
source: RDNA3 ISA PDF p.284-285
---

# V_RCP_IFLAG_F32 (opcode 43)

Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result into a vector register. This opcode is intended for use as part of an integer division macro.

### Notes

Can raise integer DIV_BY_ZERO exception but cannot raise floating-point exceptions. To be used in an integer reciprocal macro by the compiler with one of the sequences listed below (depending on signed or unsigned operation). Unsigned usage: CVT_F32_U32 RCP_IFLAG_F32 MUL_F32 (2**32 - 1) CVT_U32_F32 Signed usage: CVT_F32_I32 RCP_IFLAG_F32 MUL_F32 (2**31 - 1) CVT_I32_F32
