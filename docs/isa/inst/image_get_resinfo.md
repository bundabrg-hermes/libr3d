---
mnemonic: IMAGE_GET_RESINFO
section: "16.18. MIMG Instructions"
format: MIMG
opcode: "23"
pages: [571]
source: RDNA3 ISA PDF p.571-571
---

# IMAGE_GET_RESINFO (opcode 23)

Return resource info for a given mip level specified in the address vgpr. No sampler. Returns 4 integer values into VGPRs 3-0: {num_mip_levels, depth, height, width}.
