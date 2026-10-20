---
mnemonic: V_EXP_F32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "37"
pages: [283]
source: RDNA3 ISA PDF p.283-283
---

# V_EXP_F32 (opcode 37)

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
