---
mnemonic: V_COS_F32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "438"
pages: [387]
source: RDNA3 ISA PDF p.387-387
---

# V_COS_F32 (opcode 438)

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
