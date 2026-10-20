---
mnemonic: S_BITCMP0_B32
section: "16.4. SOPC Instructions"
format: SOPC
opcode: "12"
pages: [240]
source: RDNA3 ISA PDF p.240-240
---

# S_BITCMP0_B32 (opcode 12)

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the extracted bit is equal to 0.
