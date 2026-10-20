# 8.1.2. Loads using Buffer Constant

_(pages 86-86)_

S_BUFFER_LOAD instructions use a similar formula, but the base address comes from the buffer constant’s base_address field. Buffer constant fields used: base_address, stride, num_records. Other fields are ignored. Scalar memory load does not support "swizzled" buffers. Stride is used only for memory address bounds checking, not for computing the address to access. The SMEM supplies only a SBASE address (byte) and an offset (byte or DWORD). Any "index * stride" must be calculated manually in shader code and added to the offset prior to the SMEM. Inst_offset must be non- negative - a negative value of inst_offset results in a MEMVIOL. The two LSBs of V#.base and of the final address are ignored to force DWORD alignment.
