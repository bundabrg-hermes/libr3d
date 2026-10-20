---
mnemonic: S_NOR_SAVEEXEC_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "41"
pages: [229]
source: RDNA3 ISA PDF p.229-229
---

# S_NOR_SAVEEXEC_B64 (opcode 41)

Bitwise NOR with EXEC mask. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
