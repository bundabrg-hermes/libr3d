---
mnemonic: S_CBRANCH_CDBGSYS_OR_USER
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "41"
pages: [252]
source: RDNA3 ISA PDF p.252-252
---

# S_CBRANCH_CDBGSYS_OR_USER (opcode 41)

If either the system debug flag or the user debug flag is set then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
