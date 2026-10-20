---
mnemonic: V_MAXMIN_U32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "610"
pages: [435, 436]
source: RDNA3 ISA PDF p.435-436
---

# V_MAXMIN_U32 (opcode 610)

Compute maximum of first two operands, followed by minimum of that result and the third operand. This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the clamp's maxBound < minBound.
