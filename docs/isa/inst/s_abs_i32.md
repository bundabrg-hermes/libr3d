---
mnemonic: S_ABS_I32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "21"
pages: [222]
source: RDNA3 ISA PDF p.222-222
---

# S_ABS_I32 (opcode 21)

Compute the absolute value of a scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

### Notes

Functional examples:
