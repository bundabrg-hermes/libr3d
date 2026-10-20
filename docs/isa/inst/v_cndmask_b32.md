---
mnemonic: V_CNDMASK_B32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "1"
pages: [259]
source: RDNA3 ISA PDF p.259-259
---

# V_CNDMASK_B32 (opcode 1)

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

### Notes

In VOP3 the VCC source may be a scalar GPR specified in S2. Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This instruction is suitable for negating or taking the absolute value of a floating-point value.
