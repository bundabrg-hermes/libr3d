---
mnemonic: IMAGE_GATHER4H
section: "16.18. MIMG Instructions"
format: MIMG
opcode: "144"
pages: [580]
source: RDNA3 ISA PDF p.580-580
---

# IMAGE_GATHER4H (opcode 144)

Fetch 1 component per texel from 4x1 texels. DMASK selects which component to read (R,G,B,A) and must have only one bit set to 1.
