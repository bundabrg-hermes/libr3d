---
mnemonic: S_CBRANCH_CDBGSYS
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "39"
pages: [251]
source: RDNA3 ISA PDF p.251-251
---

# S_CBRANCH_CDBGSYS (opcode 39)

If the system debug flag is set then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
