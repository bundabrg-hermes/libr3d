---
mnemonic: S_ROUND_MODE
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "17"
pages: [248]
source: RDNA3 ISA PDF p.248-248
---

# S_ROUND_MODE (opcode 17)

Set floating point round mode using an immediate constant. Avoids wait state penalty that would be imposed by S_SETREG.
