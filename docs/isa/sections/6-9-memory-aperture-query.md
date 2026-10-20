# 6.9. Memory Aperture Query

_(pages 61-62)_

Shaders can query the memory aperture base and size for shared and private space through scalar operands: • PRIVATE_BASE • PRIVATE_LIMIT • SHARED_BASE • SHARED_LIMIT These values originate from the SH_MEM_BASES register ("SMB"), and are used primarily with FLAT memory instructions. Setting Shared Base or Private Base to zero disables that aperture. "PTR32" is short for "Address mode is 32bit", and "SMB" is short for "SH_MEM_BASES". These constants can be used by SALU and VALU ops, and are 64-bit unsigned integers: "Hole" = (addr[63:47] != all zeros or all ones) and is the illegal address section of memory
