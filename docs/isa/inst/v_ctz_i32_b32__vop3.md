---
mnemonic: V_CTZ_I32_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "442"
pages: [389]
source: RDNA3 ISA PDF p.389-389
---

# V_CTZ_I32_B32 (opcode 442)

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector register. Store -1 if there are no "1" bits in the input.

### Notes

Compare with S_CTZ_I32_B32, which performs the equivalent operation in the scalar ALU. Functional examples:
