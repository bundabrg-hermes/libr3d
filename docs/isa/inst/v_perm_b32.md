---
mnemonic: V_PERM_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "580"
pages: [426, 427]
source: RDNA3 ISA PDF p.426-427
---

# V_PERM_B32 (opcode 580)

Permute a 64-bit value constructed from two vector inputs using a per-lane selector and store the result into a vector register.

### Notes

Selects 8 through 11 are useful in modeling sign extension of a smaller-precision signed integer to a larger- precision result. Note the MSBs of the 64-bit value being selected are stored in S0. This is counterintuitive for a little-endian architecture.
