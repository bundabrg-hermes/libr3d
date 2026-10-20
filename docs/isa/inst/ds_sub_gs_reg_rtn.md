---
mnemonic: DS_SUB_GS_REG_RTN
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "123"
pages: [538, 539]
source: RDNA3 ISA PDF p.538-539
---

# DS_SUB_GS_REG_RTN (opcode 123)

Perform an atomic subtraction from data in specific registers embedded in GDS rather than operating on GDS memory directly. This instruction returns the pre-op value. This instruction is only used by the GS stage and is used to facilitate streamout. The return value may be 32 bits or 64 bits depending on the GS register accessed. The data value is 32 bits. 32-bit GS registers: offset[5:2] Register 0 GDS_STRMOUT_BUFFER_FILLED_SIZE_0 1 GDS_STRMOUT_BUFFER_FILLED_SIZE_1 2 GDS_STRMOUT_BUFFER_FILLED_SIZE_2 3 GDS_STRMOUT_BUFFER_FILLED_SIZE_3 4 GDS_GS_0 5 GDS_GS_1 6 GDS_GS_2 7 GDS_GS_3 64-bit GS registers: offset[5:2] Register 8 GDS_STRMOUT_PRIMS_NEEDED_0 9 GDS_STRMOUT_PRIMS_WRITTEN_0 10 GDS_STRMOUT_PRIMS_NEEDED_1 11 GDS_STRMOUT_PRIMS_WRITTEN_1 12 GDS_STRMOUT_PRIMS_NEEDED_2 13 GDS_STRMOUT_PRIMS_WRITTEN_2 14 GDS_STRMOUT_PRIMS_NEEDED_3 15 GDS_STRMOUT_PRIMS_WRITTEN_3
