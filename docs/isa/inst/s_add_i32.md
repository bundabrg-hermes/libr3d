---
mnemonic: S_ADD_I32
section: "16.1. SOP2 Instructions"
format: SOP2
opcode: "2"
pages: [197]
source: RDNA3 ISA PDF p.197-197
---

# S_ADD_I32 (opcode 2)

Add two signed inputs, store the result into a scalar register and store the carry-out bit into SCC.

### Notes

This opcode is not suitable for use with S_ADDC_U32 for implementing 64-bit operations.
