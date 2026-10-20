---
mnemonic: V_LOG_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "39"
pages: [283]
source: RDNA3 ISA PDF p.283-283
---

# V_LOG_F32 (opcode 39)

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
