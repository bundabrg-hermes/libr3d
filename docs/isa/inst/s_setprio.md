---
mnemonic: S_SETPRIO
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "53"
pages: [253]
source: RDNA3 ISA PDF p.253-253
---

# S_SETPRIO (opcode 53)

Change wave user priority. User settable wave priority is set to SIMM16[1:0]. 0 is the lowest priority and 3 is the highest. The overall wave priority is: The system priority cannot be modified from within the wave.
