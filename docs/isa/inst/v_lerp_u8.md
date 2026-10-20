---
mnemonic: V_LERP_U8
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "533"
pages: [418]
source: RDNA3 ISA PDF p.418-418
---

# V_LERP_U8 (opcode 533)

Unsigned 8-bit pixel average on packed unsigned bytes (linear interpolation). Each byte in S2 acts as a round mode; if the LSB is set then 0.5 rounds up, otherwise 0.5 truncates.
