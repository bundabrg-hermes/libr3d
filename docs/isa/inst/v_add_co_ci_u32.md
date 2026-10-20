---
mnemonic: V_ADD_CO_CI_U32
section: "16.7. VOP2 Instructions"
format: VOP2
opcode: "32"
pages: [266]
source: RDNA3 ISA PDF p.266-266
---

# V_ADD_CO_CI_U32 (opcode 32)

Add two unsigned inputs and a bit from a carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at S2.u. Supports saturation (unsigned 32-bit integer domain).
