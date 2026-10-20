---
mnemonic: S_LOAD_B32
section: "16.6. SMEM Instructions"
format: SMEM
opcode: "0"
pages: [255]
source: RDNA3 ISA PDF p.255-255
---

# S_LOAD_B32 (opcode 0)

Load 32 bits of data from the scalar data cache into a scalar register.

### Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored). If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.
