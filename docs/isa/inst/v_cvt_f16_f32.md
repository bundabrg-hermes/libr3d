---
mnemonic: V_CVT_F16_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "10"
pages: [277]
source: RDNA3 ISA PDF p.277-277
---

# V_CVT_F16_F32 (opcode 10)

Convert from a single-precision float input to an FP16 float and store the result into a vector register.

### Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on output if specified based on DP denorm mode. Output rounding based on DP rounding mode.
