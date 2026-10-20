---
mnemonic: V_MOV_B16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "412"
pages: [382]
source: RDNA3 ISA PDF p.382-382
---

# V_MOV_B16 (opcode 412)

Move data to a VGPR.

### Notes

Floating-point modifiers are valid for this instruction if S0.u16 is a 16-bit floating point value. This instruction is suitable for negating or taking the absolute value of a floating-point value.
