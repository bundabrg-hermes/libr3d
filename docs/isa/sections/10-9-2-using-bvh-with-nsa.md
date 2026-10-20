# 10.9.2. Using BVH with NSA

_(pages 117-118)_

When using the BVH instruction with Non-Sequential Address, the BVH components fall into 5 groups each of which is specified by a NSA address VGPR. • node pointer : 1 vgpr • ray extent : 1 vgpr • ray origin : 3 consecutive vgprs • ray dir : 3 consecutive vgprs • ray inv dir : 3 consecutive vgprs (paired with ray-dir for 16-bit addresses) NSA and A16: • A16=0, MIMG-NSA specifies 5 groups of consecutive VGPRs: node_pointer, ray_extent, ray_origin, ray_dir and ray_inv_dir. • A16=1, MIMG-NSA specifies 4 groups. In the above set, ray_dir and ray_inv_dir are packed into 3 VGPRs. When using A16=1 mode, ray-dir and ray-inv-dir share the same vgprs and ADDR4 is unused.
