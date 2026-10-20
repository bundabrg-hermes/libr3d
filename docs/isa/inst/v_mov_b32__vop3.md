---
mnemonic: V_MOV_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "385"
pages: [374]
source: RDNA3 ISA PDF p.374-374
---

# V_MOV_B32 (opcode 385)

Move data from a vector input into a vector register.

### Notes

Floating-point modifiers are valid for this instruction if S0.u is a 32-bit floating point value. This instruction is suitable for negating or taking the absolute value of a floating-point value. Functional examples:
