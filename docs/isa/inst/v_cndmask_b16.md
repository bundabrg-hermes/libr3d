---
mnemonic: V_CNDMASK_B16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "605"
pages: [434]
source: RDNA3 ISA PDF p.434-434
---

# V_CNDMASK_B16 (opcode 605)

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

### Notes

In VOP3 the VCC source may be a scalar GPR specified in S2. Floating-point modifiers are valid for this instruction if S0 and S1 are 16-bit floating point values. This instruction is suitable for negating or taking the absolute value of a floating-point value.
