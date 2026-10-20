---
mnemonic: V_CVT_F32_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "15"
pages: [278]
source: RDNA3 ISA PDF p.278-278
---

# V_CVT_F32_F64 (opcode 15)

Convert from a double-precision float input to a single-precision float and store the result into a vector register.

### Notes

0.5ULP accuracy, denormals are supported.
