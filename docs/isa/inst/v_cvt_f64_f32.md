---
mnemonic: V_CVT_F64_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "16"
pages: [279]
source: RDNA3 ISA PDF p.279-279
---

# V_CVT_F64_F32 (opcode 16)

Convert from a single-precision float input to a double-precision float and store the result into a vector register.

### Notes

0ULP accuracy, denormals are supported.
