---
mnemonic: V_SUBREV_CO_CI_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "290"
pages: [409]
source: RDNA3 ISA PDF p.409-409
---

# V_SUBREV_CO_CI_U32 (opcode 290)

Subtract the first unsigned input from the second input, subtract a bit from the carry-in mask, store the result into a vector register and store the carry-out mask to a scalar register.

### Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at S2.u. Supports saturation (unsigned 32-bit integer domain).
