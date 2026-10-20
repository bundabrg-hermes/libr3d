---
mnemonic: S_XNOR_SAVEEXEC_B32
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "42"
pages: [229]
source: RDNA3 ISA PDF p.229-229
---

# S_XNOR_SAVEEXEC_B32 (opcode 42)

Bitwise XNOR with EXEC mask. The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.
