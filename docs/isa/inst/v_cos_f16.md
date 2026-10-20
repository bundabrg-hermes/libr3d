---
mnemonic: V_COS_F16
section: "16.8. VOP1 Instructions"
format: VOP1
opcode: "97"
pages: [299]
source: RDNA3 ISA PDF p.299-299
---

# V_COS_F16 (opcode 97)

Calculate the trigonometric cosine of an FP16 float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
