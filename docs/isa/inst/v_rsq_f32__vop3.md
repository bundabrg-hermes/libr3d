---
mnemonic: V_RSQ_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "430"
pages: [385]
source: RDNA3 ISA PDF p.385-385
---

# V_RSQ_F32 (opcode 430)

Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
