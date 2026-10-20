---
mnemonic: DS_CMPSTORE_RTN_F64
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "113"
pages: [536]
source: RDNA3 ISA PDF p.536-536
---

# DS_CMPSTORE_RTN_F64 (opcode 113)

Floating point compare and store that handles NAN/INF/denormal values.

### Notes

In this architecture the order of src and cmp agree with the BUFFER_ATOMIC_CMPSWAP opcode.
