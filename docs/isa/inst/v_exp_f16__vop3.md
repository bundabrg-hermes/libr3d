---
mnemonic: V_EXP_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "472"
pages: [396, 397]
source: RDNA3 ISA PDF p.396-397
---

# V_EXP_F16 (opcode 472)

Calculate 2 raised to the power of the half-precision float input and store the result into a vector register.

### Notes

0.51ULP accuracy, denormals are supported. Functional examples:
