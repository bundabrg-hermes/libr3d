---
mnemonic: V_MBCNT_HI_U32_B32
section: "16.12. VOP3 & VOP3SD Instructions"
format: VOP3
opcode: "800"
pages: [444, 445]
source: RDNA3 ISA PDF p.444-445
---

# V_MBCNT_HI_U32_B32 (opcode 800)

Masked bit count, high pass. laneId is the position of this thread in the wavefront (in 0..63). See also V_MBCNT_LO_U32_B32.

### Notes

Note that in Wave32 mode ThreadMask[63:32] == 0 and this instruction simply performs a move from S1 to D. Example to compute each thread's position in 0..63:
