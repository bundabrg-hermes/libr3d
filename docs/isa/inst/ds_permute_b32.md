---
mnemonic: DS_PERMUTE_B32
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "178"
pages: [543]
source: RDNA3 ISA PDF p.543-543
---

# DS_PERMUTE_B32 (opcode 178)

Forward permute. This does not access LDS memory and may be called even if no LDS memory is allocated to the wave. It uses LDS to implement an arbitrary swizzle across threads in a wavefront. Note the address passed in is the thread ID multiplied by 4. If multiple sources map to the same destination lane, it is not deterministic which source lane writes to the destination lane. See also DS_BPERMUTE_B32.

### Notes

Examples (simplified 4-thread wavefronts):
