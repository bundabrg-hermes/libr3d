---
mnemonic: S_CLZ_I32_U64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "11"
pages: [219]
source: RDNA3 ISA PDF p.219-219
---

# S_CLZ_I32_U64 (opcode 11)

Count the number of leading "0" bits before the first "1" in a scalar input and store the result into a scalar register. Store -1 if there are no "1" bits.
