---
mnemonic: V_EXP_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "421"
pages: [383]
source: RDNA3 ISA PDF p.383-383
---

# V_EXP_F32 (opcode 421)

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
