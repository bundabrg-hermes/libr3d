---
mnemonic: V_CVT_F16_I16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "465"
pages: [394]
source: RDNA3 ISA PDF p.394-394
---

# V_CVT_F16_I16 (opcode 465)

Convert from a signed 16-bit integer input to an FP16 float and store the result into a vector register.

### Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.
