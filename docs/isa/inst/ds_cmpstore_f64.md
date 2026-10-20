---
mnemonic: DS_CMPSTORE_F64
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "81"
pages: [531]
source: RDNA3 ISA PDF p.531-531
---

# DS_CMPSTORE_F64 (opcode 81)

Floating point compare and store that handles NAN/INF/denormal values.

### Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.
