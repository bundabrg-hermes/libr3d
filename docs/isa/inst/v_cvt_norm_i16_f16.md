---
mnemonic: V_CVT_NORM_I16_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "99"
pages: [300]
source: RDNA3 ISA PDF p.300-300
---

# V_CVT_NORM_I16_F16 (opcode 99)

Convert from an FP16 float input to a signed normalized short and store the result into a vector register.

### Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.
