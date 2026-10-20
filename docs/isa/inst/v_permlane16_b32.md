---
mnemonic: V_PERMLANE16_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "603"
pages: [432]
source: RDNA3 ISA PDF p.432-432
---

# V_PERMLANE16_B32 (opcode 603)

Perform arbitrary gather-style operation within a row (16 contiguous lanes). The first source must be a VGPR and the second and third sources must be scalar values; the second and third source are combined into a single 64-bit value representing lane selects used to swizzle within each row. OPSEL is not used in its typical manner for this instruction. For this instruction OPSEL[0] is overloaded to represent the DPP 'FI' (Fetch Inactive) bit and OPSEL[1] is overloaded to represent the DPP 'BOUND_CTRL' bit. The remaining OPSEL bits are reserved for this instruction. Compare with V_PERMLANEX16_B32.

### Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction. Example implementing a rotation within each row:
