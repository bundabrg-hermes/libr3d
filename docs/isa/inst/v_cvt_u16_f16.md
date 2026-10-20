---
mnemonic: V_CVT_U16_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "82"
pages: [294]
source: RDNA3 ISA PDF p.294-294
---

# V_CVT_U16_F16 (opcode 82)

Convert from an FP16 float input to an unsigned 16-bit integer and store the result into a vector register.

### Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion is done with truncation. Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this conversion iff CLAMP == 1.
