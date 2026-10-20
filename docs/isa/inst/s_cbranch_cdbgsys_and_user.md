---
mnemonic: S_CBRANCH_CDBGSYS_AND_USER
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "42"
pages: [252]
source: RDNA3 ISA PDF p.252-252
---

# S_CBRANCH_CDBGSYS_AND_USER (opcode 42)

If both the system debug flag and the user debug flag are set then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
