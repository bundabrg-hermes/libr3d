---
mnemonic: LDS_DIRECT_LOAD
section: "16.14. Parameter and Direct Load from LDS Instructions"
format: PARAM
opcode: "1"
pages: [510]
source: RDNA3 ISA PDF p.510-510
---

# LDS_DIRECT_LOAD (opcode 1)

Read a single 32-bit value from LDS to all lanes. A single DWORD is read from LDS memory at ADDR[M0[15:0]], where M0[15:0] is a byte address and is dword-aligned. M0[18:16] specify the data type for the read and may be 0=UBYTE, 1=USHORT, 2=DWORD, 4=SBYTE, 5=SSHORT.  This instruction runs in whole quad mode: if any pixel of a quad is active then all 4 pixels of that quad are written.
