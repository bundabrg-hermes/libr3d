---
mnemonic: V_MOV_B16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "28"
pages: [281]
source: RDNA3 ISA PDF p.281-281
---

# V_MOV_B16 (opcode 28)

Move data to a VGPR.

### Notes

Floating-point modifiers are valid for this instruction if S0.u16 is a 16-bit floating point value. This instruction is suitable for negating or taking the absolute value of a floating-point value.
