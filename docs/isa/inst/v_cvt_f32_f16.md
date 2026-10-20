---
mnemonic: V_CVT_F32_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "11"
pages: [277]
source: RDNA3 ISA PDF p.277-277
---

# V_CVT_F32_F16 (opcode 11)

Convert from an FP16 float input to a single-precision float and store the result into a vector register.

### Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm mode.
