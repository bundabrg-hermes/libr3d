---
mnemonic: V_LOG_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "471"
pages: [396]
source: RDNA3 ISA PDF p.396-396
---

# V_LOG_F16 (opcode 471)

Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register.

### Notes

0.51ULP accuracy, denormals are supported. Functional examples:
