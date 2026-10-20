---
mnemonic: V_CLS_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "443"
pages: [389, 390]
source: RDNA3 ISA PDF p.389-390
---

# V_CLS_I32 (opcode 443)

Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a vector register. Store -1 if all input bits are the same.

### Notes

Compare with S_CLS_I32, which performs the equivalent operation in the scalar ALU. Functional examples:
