---
mnemonic: V_WRITELANE_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "865"
pages: [452]
source: RDNA3 ISA PDF p.452-452
---

# V_WRITELANE_B32 (opcode 865)

Write scalar value into one VGPR in one lane.

### Notes

Ignores EXEC mask for the VGPR write. Input and output modifiers not supported; this is an untyped operation.
