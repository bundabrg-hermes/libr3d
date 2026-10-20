---
mnemonic: V_RCP_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "47"
pages: [285]
source: RDNA3 ISA PDF p.285-285
---

# V_RCP_F64 (opcode 47)

Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector register.

### Notes

This opcode has (2**29)ULP accuracy and supports denormals.
