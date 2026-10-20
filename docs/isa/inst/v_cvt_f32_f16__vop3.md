---
mnemonic: V_CVT_F32_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "395"
pages: [377]
source: RDNA3 ISA PDF p.377-377
---

# V_CVT_F32_F16 (opcode 395)

Convert from an FP16 float input to a single-precision float and store the result into a vector register.

### Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm mode.
