---
mnemonic: S_DENORM_MODE
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "18"
pages: [248]
source: RDNA3 ISA PDF p.248-248
---

# S_DENORM_MODE (opcode 18)

Set floating point denormal mode using an immediate constant. Avoids wait state penalty that would be imposed by S_SETREG.
