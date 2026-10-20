---
mnemonic: GLOBAL_STORE_ADDTID_B32
section: "16.20.3. Global Instructions"
format: GLOBAL
opcode: "41"
pages: [602]
source: RDNA3 ISA PDF p.602-602
---

# GLOBAL_STORE_ADDTID_B32 (opcode 41)

Untyped buffer store dword. No VGPR address is supplied in this instruction. TID is added to the address as shown below: memory_Addr = sgpr_addr(64) + inst_offset(12) + tid*4
