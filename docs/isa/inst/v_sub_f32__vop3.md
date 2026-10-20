---
mnemonic: V_SUB_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "260"
pages: [402]
source: RDNA3 ISA PDF p.402-402
---

# V_SUB_F32 (opcode 260)

Subtract the second floating point input from the first input and store the result into a vector register.

### Notes

0.5ULP precision, denormals are supported.
