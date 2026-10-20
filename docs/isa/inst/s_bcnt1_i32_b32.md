---
mnemonic: S_BCNT1_I32_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "24"
pages: [223, 224]
source: RDNA3 ISA PDF p.223-224
---

# S_BCNT1_I32_B32 (opcode 24)

Count the number of "1" bits in a scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

### Notes

Functional examples:
