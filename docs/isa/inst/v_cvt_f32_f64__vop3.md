---
mnemonic: V_CVT_F32_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "399"
pages: [379]
source: RDNA3 ISA PDF p.379-379
---

# V_CVT_F32_F64 (opcode 399)

Convert from a double-precision float input to a single-precision float and store the result into a vector register.

### Notes

0.5ULP accuracy, denormals are supported.
