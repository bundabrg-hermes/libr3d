---
mnemonic: V_RSQ_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "49"
pages: [285, 286]
source: RDNA3 ISA PDF p.285-286
---

# V_RSQ_F64 (opcode 49)

Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the result into a vector register.

### Notes

This opcode has (2**29)ULP accuracy and supports denormals.
