---
mnemonic: V_CNDMASK_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "257"
pages: [401, 402]
source: RDNA3 ISA PDF p.401-402
---

# V_CNDMASK_B32 (opcode 257)

Copy data from one of two inputs based on the vector condition code and store the result into a vector register.

### Notes

In VOP3 the VCC source may be a scalar GPR specified in S2. Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This instruction is suitable for negating or taking the absolute value of a floating-point value.
