---
mnemonic: V_CVT_F16_U16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "80"
pages: [293]
source: RDNA3 ISA PDF p.293-293
---

# V_CVT_F16_U16 (opcode 80)

Convert from an unsigned 16-bit integer input to an FP16 float and store the result into a vector register.

### Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.
