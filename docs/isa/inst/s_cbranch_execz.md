---
mnemonic: S_CBRANCH_EXECZ
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "37"
pages: [251]
source: RDNA3 ISA PDF p.251-251
---

# S_CBRANCH_EXECZ (opcode 37)

If EXECZ is 1 then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
