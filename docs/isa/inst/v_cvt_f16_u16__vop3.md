---
mnemonic: V_CVT_F16_U16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "464"
pages: [393, 394]
source: RDNA3 ISA PDF p.393-394
---

# V_CVT_F16_U16 (opcode 464)

Convert from an unsigned 16-bit integer input to an FP16 float and store the result into a vector register.

### Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.
