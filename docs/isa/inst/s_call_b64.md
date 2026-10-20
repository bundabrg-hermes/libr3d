---
mnemonic: S_CALL_B64
section: "16.2. SOPK Instructions"
format: SOPK
opcode: "20"
pages: [214]
source: RDNA3 ISA PDF p.214-214
---

# S_CALL_B64 (opcode 20)

Store the address of the next instruction to a scalar register and then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction. The byte address of the instruction immediately following this instruction is saved to the destination.

### Notes

This implements a short subroutine call where the return address (the next instruction after the S_CALL_B64) is saved to D. Long calls should consider S_SWAPPC_B64 instead. This instruction must be 4 bytes.
