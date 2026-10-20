---
mnemonic: V_RCP_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "426"
pages: [384]
source: RDNA3 ISA PDF p.384-384
---

# V_RCP_F32 (opcode 426)

Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector register.

### Notes

1ULP accuracy. Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA operations. Denormals are flushed. Functional examples:
