---
mnemonic: S_GETPC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "71"
pages: [236]
source: RDNA3 ISA PDF p.236-236
---

# S_GETPC_B64 (opcode 71)

Store the address of the next instruction to a scalar register. The byte address of the instruction immediately following this instruction is saved to the destination.

### Notes

This instruction must be 4 bytes.
