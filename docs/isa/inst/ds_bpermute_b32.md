---
mnemonic: DS_BPERMUTE_B32
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "179"
pages: [544]
source: RDNA3 ISA PDF p.544-544
---

# DS_BPERMUTE_B32 (opcode 179)

Backward permute. This does not access LDS memory and may be called even if no LDS memory is allocated to the wave. It uses LDS hardware to implement an arbitrary swizzle across threads in a wavefront. Note the address passed in is the thread ID multiplied by 4. Note that EXEC mask is applied to both VGPR read and write. If src_lane selects a disabled thread then zero is returned. See also DS_PERMUTE_B32.

### Notes

Examples (simplified 4-thread wavefronts):
