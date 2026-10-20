# 9.4.2. Swizzled Buffer Addressing

_(pages 99-100)_

Swizzled addressing rearranges the data in the buffer that may improve cache locality for arrays of structures. Swizzled addressing also requires DWORD-aligned accesses. A single fetch instruction must not fetch a unit larger than const_element_size. The buffer’s STRIDE must be a multiple of const_element_size. const_element_size is either 4 or 16 bytes, depending on the setting of V#.swizzle_enable Example of Buffer Swizzling
