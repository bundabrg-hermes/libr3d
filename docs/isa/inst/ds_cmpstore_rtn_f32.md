---
mnemonic: DS_CMPSTORE_RTN_F32
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "49"
pages: [521]
source: RDNA3 ISA PDF p.521-521
---

# DS_CMPSTORE_RTN_F32 (opcode 49)

Floating point compare and store that handles NAN/INF/denormal values.

### Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.
