---
mnemonic: S_BCNT0_I32_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "22"
pages: [223]
source: RDNA3 ISA PDF p.223-223
---

# S_BCNT0_I32_B32 (opcode 22)

Count the number of "0" bits in a scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

### Notes

Functional examples:
