---
mnemonic: DS_CMPSTORE_F32
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "17"
pages: [515]
source: RDNA3 ISA PDF p.515-515
---

# DS_CMPSTORE_F32 (opcode 17)

Floating point compare and store that handles NAN/INF/denormal values.

### Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.
