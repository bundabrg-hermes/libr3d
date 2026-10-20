---
mnemonic: S_CTZ_I32_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "8"
pages: [218]
source: RDNA3 ISA PDF p.218-218
---

# S_CTZ_I32_B32 (opcode 8)

Count the number of trailing "0" bits before the first "1" in a scalar input and store the result into a scalar register. Store -1 if there are no "1" bits in the input.

### Notes

Functional examples: Compare with V_CTZ_I32_B32, which performs the equivalent operation in the vector ALU.
