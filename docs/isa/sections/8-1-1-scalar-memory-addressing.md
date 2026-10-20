# 8.1.1. Scalar Memory Addressing

_(pages 86-86)_

Non-buffer S_LOAD instructions use the following formula to calculate the memory address: ADDR = SGPR[base] + inst_offset + { M0 or SGPR[offset] or zero } All components of the address (base, offset, inst_offset, M0) are in bytes, but the two LSBs are ignored and treated as if they were zero. It is illegal and undefined for the inst_offset to be negative if the resulting (inst_offset + (M0 or SGPR[offset])) is negative.
