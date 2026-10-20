---
mnemonic: V_SQRT_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "436"
pages: [386, 387]
source: RDNA3 ISA PDF p.386-387
---

# V_SQRT_F64 (opcode 436)

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector register.

### Notes

This opcode has (2**29)ULP accuracy and supports denormals.
