---
mnemonic: GLOBAL_ATOMIC_CMPSWAP_F32
section: "16.20.3. Global Instructions"
format: GLOBAL
opcode: "80"
pages: [608, 609]
source: RDNA3 ISA PDF p.608-609
---

# GLOBAL_ATOMIC_CMPSWAP_F32 (opcode 80)

Compare and swap with floating-point memory value.

### Notes

Floating-point compare handles NAN/INF/denorm.
