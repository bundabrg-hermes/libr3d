# 3.3.2. VGPRs

_(pages 26-27)_

### 3.3.2.1. VGPR Allocation and Alignment

VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a shader may have up to 256 VGPRs. In other words, VGPRs are allocated in units of (16*32 or 8*64 = 512 DWORDs). A wave may not be created with zero VGPRs. Devices which have 1536 VGPRs per SIMD allocate in blocks of 24 for wave32 and 12 for wave64. A wave may voluntarily deallocate all of its VGPRs via S_SENDMSG. Once this is done, the wave may not reallocate them and the only valid action is to terminate the wave. This can be useful if a wave has issued stores to memory and is waiting for the write-confirms before terminating. Releasing the VGPRs while waiting may allow a new wave to allocate them and start earlier.

### 3.3.2.2. VGPR Out of Range Behavior

Given an instruction operand that uses one or more DWORDs of VGPR data: "V" Vs = the first VGPR DWORD (start) Ve = the last VGPR DWORD (end) For a 32-bit operand, Vs==Ve; for a 64-bit operand Ve=Vs+1, etc. Operand is out of range if: • Vs < 0 || Vs >= VGPR_SIZE • Ve < 0 || Ve >= VGPR_SIZE V_MOVREL indexed operand out of range if either: • Index > 255 • (Vs + M0) >= VGPR_SIZE • (Ve + M0) >= VGPR_SIZE Out of range consequences: • If a dest VGPR is out of range, the instruction is ignored (treat as NOP). • V_SWAP & V_SWAPREL : since both arguments are destinations, if either is out of range, discard the instruction. ◦VALU instructions with multiple destination (e.g. VGPR and SGPR): nothing is written to any GPR • If a source VGPR is out of range in a VMEM or Export instruction: VGPR0 is used ◦Memory instructions that use a group of consecutive VGPRs that are out of range use VGPR0 for the individual out of range VGPRs. • If a source VGPR in a VALU instruction is out of range in a VALU instruction: VGPR0 ◦VOPD has different rules: the source address forced to (VGPRaddr % 4). Instructions with multiple destinations (e.g. V_ADD_CO): if any destination is out of range, no results are written.
