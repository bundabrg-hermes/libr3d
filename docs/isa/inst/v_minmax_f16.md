---
mnemonic: V_MINMAX_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "609"
pages: [435]
source: RDNA3 ISA PDF p.435-435
---

# V_MINMAX_F16 (opcode 609)

Compute minimum of first two operands, followed by maximum of that result and the third operand. This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the clamp's maxBound < minBound.

### Notes

Support input denorm control, allow output denorm value. Exceptions are supported. Note: +0.0 > -0.0 is true.
