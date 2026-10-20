---
mnemonic: S_SWAPPC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "73"
pages: [236]
source: RDNA3 ISA PDF p.236-236
---

# S_SWAPPC_B64 (opcode 73)

Store the address of the next instruction to a scalar register and then jump to an address specified in the scalar input. The argument is a byte address of the instruction to jump to. The byte address of the instruction immediately following this instruction is saved to the destination.

### Notes

This instruction must be 4 bytes.
