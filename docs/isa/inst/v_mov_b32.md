---
mnemonic: V_MOV_B32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "1"
pages: [274]
source: RDNA3 ISA PDF p.274-274
---

# V_MOV_B32 (opcode 1)

Move data from a vector input into a vector register.

### Notes

Floating-point modifiers are valid for this instruction if S0.u is a 32-bit floating point value. This instruction is suitable for negating or taking the absolute value of a floating-point value. Functional examples:
