---
mnemonic: BUFFER_ATOMIC_ADD_F32
section: "16.16. MUBUF Instructions"
format: MUBUF
opcode: "86"
pages: [562]
source: RDNA3 ISA PDF p.562-562
---

# BUFFER_ATOMIC_ADD_F32 (opcode 86)

Add data register to floating-point memory value.

### Notes

Floating-point addition handles NAN/INF/denorm.
