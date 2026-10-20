---
mnemonic: V_SQRT_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "51"
pages: [286]
source: RDNA3 ISA PDF p.286-286
---

# V_SQRT_F32 (opcode 51)

Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
