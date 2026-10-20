---
mnemonic: S_RFE_B64
section: "16.3. SOP1 Instructions"
format: SOP1
opcode: "74"
pages: [236, 237]
source: RDNA3 ISA PDF p.236-237
---

# S_RFE_B64 (opcode 74)

Return from the exception handler. Clear the wave's PRIV bit and then jump to an address specified by the scalar input. The argument is a byte address of the instruction to jump to; this address is likely derived from the state passed into the trap handler. This instruction may only be used within a trap handler.
