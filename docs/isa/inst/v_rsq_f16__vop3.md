---
mnemonic: V_RSQ_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "470"
pages: [395, 396]
source: RDNA3 ISA PDF p.395-396
---

# V_RSQ_F16 (opcode 470)

Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result into a vector register.

### Notes

0.51ULP accuracy, denormals are supported. Functional examples:
