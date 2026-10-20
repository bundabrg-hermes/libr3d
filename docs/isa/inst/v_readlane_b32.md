---
mnemonic: V_READLANE_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "864"
pages: [451, 452]
source: RDNA3 ISA PDF p.451-452
---

# V_READLANE_B32 (opcode 864)

Copy one VGPR value from a single lane to one SGPR.

### Notes

Ignores EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped operation.
