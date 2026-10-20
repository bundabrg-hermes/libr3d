---
mnemonic: V_CLZ_I32_U32
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "57"
pages: [288]
source: RDNA3 ISA PDF p.288-288
---

# V_CLZ_I32_U32 (opcode 57)

Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector register. Store -1 if there are no "1" bits.

### Notes

Compare with S_CLZ_I32_U32, which performs the equivalent operation in the scalar ALU. Functional examples:
