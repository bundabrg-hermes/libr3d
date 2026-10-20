---
mnemonic: S_ABSDIFF_I32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "6"
pages: [198, 199]
source: RDNA3 ISA PDF p.198-199
---

# S_ABSDIFF_I32 (opcode 6)

Calculate the absolute value of difference between two scalar inputs, store the result into a scalar register and set SCC iff the result is nonzero.

### Notes

Functional examples:
