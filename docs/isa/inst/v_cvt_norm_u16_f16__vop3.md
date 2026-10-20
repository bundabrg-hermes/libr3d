---
mnemonic: V_CVT_NORM_U16_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "484"
pages: [400]
source: RDNA3 ISA PDF p.400-400
---

# V_CVT_NORM_U16_F16 (opcode 484)

Convert from an FP16 float input to an unsigned normalized short and store the result into a vector register.

### Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.
