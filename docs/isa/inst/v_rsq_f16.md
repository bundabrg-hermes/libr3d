---
mnemonic: V_RSQ_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "86"
pages: [295]
source: RDNA3 ISA PDF p.295-295
---

# V_RSQ_F16 (opcode 86)

Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result into a vector register.

### Notes

0.51ULP accuracy, denormals are supported. Functional examples:
