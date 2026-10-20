---
mnemonic: IMAGE_BVH64_INTERSECT_RAY
section: "16.18. MIMG Instructions"
format: MIMG
opcode: "26"
pages: [573]
source: RDNA3 ISA PDF p.573-573
---

# IMAGE_BVH64_INTERSECT_RAY (opcode 26)

Intersection test on bound volume hierarchy nodes for ray tracing acceleration. 64-bit node pointer. No sampler. This instruction allows support for very large BVHs (larger than 32 GBs) that may occur in workstation workloads. See IMAGE_BVH_INTERSECT_RAY for basic information including restrictions. Only differences are described here. ADDR (A16 = 0): 12 address VGPRs contain the ray data and BVH node pointer for the intersection test. The data is laid out as follows (dependent on NSA mode): • NSA=0  NSA=1  Value VADDR[0] VADDR[0] = node_pointer[31:0] (uint32) VADDR[1] VADDR[1] = node_pointer[63:32] (uint32) VADDR[2] VADDRA[0] = ray_extent (float32) VADDR[3] VADDRB[0] = ray_origin.x (float32) VADDR[4] VADDRB[1] = ray_origin.y (float32) VADDR[5] VADDRB[2] = ray_origin.z (float32) VADDR[6] VADDRC[0] = ray_dir.x (float32) VADDR[7] VADDRC[1] = ray_dir.y (float32) VADDR[8] VADDRC[2] = ray_dir.z (float32) VADDR[9] VADDRD[0] = ray_inv_dir.x (float32) VADDR[10] VADDRD[1] = ray_inv_dir.y (float32) VADDR[11] VADDRD[2] = ray_inv_dir.z (float32) ADDR (A16 = 1): When the instruction is encoded with 16 bit addresses only 9 address VGPRs are used as follows (dependent on NSA mode): • NSA=0  NSA=1  Value VADDR[0] VADDR[0] = node_pointer[31:0] (uint32) VADDR[1] VADDR[1] = node_pointer[63:32] (uint32) VADDR[2] VADDRA[0] = ray_extent (float32) VADDR[3] VADDRB[0] = ray_origin.x (float32) VADDR[4] VADDRB[1] = ray_origin.y (float32) VADDR[5] VADDRB[2] = ray_origin.z (float32) VADDR[6] VADDRC[0] = {ray_inv_dir.x, ray_dir.x} (2x float16) VADDR[7] VADDRC[1] = {ray_inv_dir.y, ray_dir.y} (2x float16) VADDR[8] VADDRC[2] = {ray_inv_dir.z, ray_dir.z} (2x float16)
