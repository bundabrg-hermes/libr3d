---
mnemonic: BUFFER_ATOMIC_CMPSWAP_F32
section: "16.16. MUBUF Instructions"
format: MUBUF
opcode: "80"
pages: [561, 562]
source: RDNA3 ISA PDF p.561-562
---

# BUFFER_ATOMIC_CMPSWAP_F32 (opcode 80)

Compare and swap with floating-point memory value.

### Notes

Floating-point compare handles NAN/INF/denorm.
