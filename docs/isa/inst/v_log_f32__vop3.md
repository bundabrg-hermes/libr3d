---
mnemonic: V_LOG_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "423"
pages: [384]
source: RDNA3 ISA PDF p.384-384
---

# V_LOG_F32 (opcode 423)

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

### Notes

1ULP accuracy, denormals are flushed. Functional examples:
