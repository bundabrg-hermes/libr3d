---
mnemonic: V_ADD_CO_CI_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "288"
pages: [408, 409]
source: RDNA3 ISA PDF p.408-409
---

# V_ADD_CO_CI_U32 (opcode 288)

Add two unsigned inputs and a bit from a carry-in mask, store the result into a vector register and store the carry-out mask into a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at S2.u. Supports saturation (unsigned 32-bit integer domain).
