---
mnemonic: V_SUB_F32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "4"
pages: [260]
source: RDNA3 ISA PDF p.260-260
---

# V_SUB_F32 (opcode 4)

Subtract the second floating point input from the first input and store the result into a vector register.

### Notes

0.5ULP precision, denormals are supported.
