---
mnemonic: V_DIV_SCALE_F64
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "765"
pages: [438]
source: RDNA3 ISA PDF p.438-438
---

# V_DIV_SCALE_F64 (opcode 765)

Double precision division pre-scale. S0 = Input to scale (either denominator or numerator), S1 = Denominator, S2 = Numerator. Given a numerator and denominator, this opcode appropriately scales inputs for division to avoid subnormal terms during Newton-Raphson correction method. S0 must be the same value as either S1 or S2. This opcode produces a VCC flag for post-scaling of the quotient (using V_DIV_FMAS_F64).
