---
mnemonic: V_CVT_F16_I16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "81"
pages: [293, 294]
source: RDNA3 ISA PDF p.293-294
---

# V_CVT_F16_I16 (opcode 81)

Convert from a signed 16-bit integer input to an FP16 float and store the result into a vector register.

### Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.
