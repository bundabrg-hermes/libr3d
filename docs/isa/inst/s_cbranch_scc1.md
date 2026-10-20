---
mnemonic: S_CBRANCH_SCC1
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "34"
pages: [250]
source: RDNA3 ISA PDF p.250-250
---

# S_CBRANCH_SCC1 (opcode 34)

If SCC is 1 then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
