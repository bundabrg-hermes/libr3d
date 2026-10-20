---
mnemonic: V_RSQ_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "46"
pages: [285]
source: RDNA3 ISA PDF p.285-285
---

# V_RSQ_F32 (opcode 46)

Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
