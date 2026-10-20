---
mnemonic: V_COS_F16
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "481"
pages: [399]
source: RDNA3 ISA PDF p.399-399
---

# V_COS_F16 (opcode 481)

Calculate the trigonometric cosine of an FP16 float value using IEEE rules and store the result into a vector register. The operand is calculated by scaling the vector input by 2 PI.

### Notes

Denormals are supported. Full range input is supported. Functional examples:
