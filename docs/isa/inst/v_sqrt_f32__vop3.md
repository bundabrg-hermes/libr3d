---
mnemonic: V_SQRT_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "435"
pages: [386]
source: RDNA3 ISA PDF p.386-386
---

# V_SQRT_F32 (opcode 435)

Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
