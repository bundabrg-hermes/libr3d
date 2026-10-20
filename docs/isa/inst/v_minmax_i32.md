---
mnemonic: V_MINMAX_I32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "613"
pages: [436]
source: RDNA3 ISA PDF p.436-436
---

# V_MINMAX_I32 (opcode 613)

Compute minimum of first two operands, followed by maximum of that result and the third operand. This instruction can emulate an API-level "clamp"; unlike MED3 this correctly handles the case where the clamp's maxBound < minBound.
