---
mnemonic: S_BRANCH
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "32"
pages: [249]
source: RDNA3 ISA PDF p.249-249
---

# S_BRANCH (opcode 32)

Jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.

### Notes

For a long jump or an indirect jump use S_SETPC_B64. Examples:
