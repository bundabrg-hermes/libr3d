---
mnemonic: V_CLZ_I32_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "441"
pages: [388]
source: RDNA3 ISA PDF p.388-388
---

# V_CLZ_I32_U32 (opcode 441)

Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector register. Store -1 if there are no "1" bits.

### Notes

Compare with S_CLZ_I32_U32, which performs the equivalent operation in the scalar ALU. Functional examples:
