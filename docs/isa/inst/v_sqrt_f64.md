---
mnemonic: V_SQRT_F64
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "52"
pages: [286]
source: RDNA3 ISA PDF p.286-286
---

# V_SQRT_F64 (opcode 52)

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector register.

### Notes

This opcode has (2**29)ULP accuracy and supports denormals.
