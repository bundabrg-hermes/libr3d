---
mnemonic: S_CBRANCH_VCCNZ
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "36"
pages: [250]
source: RDNA3 ISA PDF p.250-250
---

# S_CBRANCH_VCCNZ (opcode 36)

If VCCZ is 0 then jump to a constant offset relative to the current PC. The literal argument is a signed DWORD offset relative to the PC of the next instruction.
