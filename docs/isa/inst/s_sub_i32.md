---
mnemonic: S_SUB_I32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "3"
pages: [198]
source: RDNA3 ISA PDF p.198-198
---

# S_SUB_I32 (opcode 3)

Subtract the second signed input from the first input, store the result into a scalar register and store the carry- out bit into SCC.

### Notes

This opcode is not suitable for use with S_SUBB_U32 for implementing 64-bit operations.
