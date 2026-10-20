---
mnemonic: V_LOG_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "87"
pages: [296]
source: RDNA3 ISA PDF p.296-296
---

# V_LOG_F16 (opcode 87)

Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register.

### Notes

0.51ULP accuracy, denormals are supported. Functional examples:
