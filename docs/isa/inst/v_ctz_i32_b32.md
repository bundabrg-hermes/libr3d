---
mnemonic: V_CTZ_I32_B32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "58"
pages: [288, 289]
source: RDNA3 ISA PDF p.288-289
---

# V_CTZ_I32_B32 (opcode 58)

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector register. Store -1 if there are no "1" bits in the input.

### Notes

Compare with S_CTZ_I32_B32, which performs the equivalent operation in the scalar ALU. Functional examples:
