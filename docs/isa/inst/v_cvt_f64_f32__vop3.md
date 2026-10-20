---
mnemonic: V_CVT_F64_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "400"
pages: [379]
source: RDNA3 ISA PDF p.379-379
---

# V_CVT_F64_F32 (opcode 400)

Convert from a single-precision float input to a double-precision float and store the result into a vector register.

### Notes

0ULP accuracy, denormals are supported.
