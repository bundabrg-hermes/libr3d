---
mnemonic: BUFFER_ATOMIC_MIN_F32
section: "16.16. MUBUF Instructions"
format: MUBUF
opcode: "81"
pages: [562]
source: RDNA3 ISA PDF p.562-562
---

# BUFFER_ATOMIC_MIN_F32 (opcode 81)

Minimum of two floating-point values.

### Notes

Floating-point compare handles NAN/INF/denorm.
