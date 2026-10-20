# 12.5. Data Share Indexed and Atomic Access

_(pages 134-136)_

Both LDS and GDS can perform indexed and atomic data share operations. For brevity, "LDS" is used in the text below and, except where noted, also applies to GDS. Indexed and atomic operations supply a unique address per work-item from the VGPRs to the LDS, and supply or return unique data per work-item back to VGPRs. Due to the internal banked structure of LDS, operations can complete in as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending upon the number of bank conflicts (addresses that map to the same memory bank). Indexed operations are simple LDS load and store operations that read data from, and return data to, VGPRs. Atomic operations are arithmetic operations that combine data from VGPRs and data in LDS, and write the result back to LDS. Atomic operations have the option of returning the LDS "pre-op" value to VGPRs. LDS Indexed and atomic instructions use LGKMcnt to track when they have completed. LGKMcnt is incremented as each instruction is issued, and decremented when they have completed execution. LDS instructions stay in-order with other LDS instructions from the same wave. The table below lists and briefly describes the LDS instruction fields. Table 59. LDS Instruction Fields The M0 register is not used for most LDS-indexed operations: only the "ADDTID" instructions read M0 and for these it represents a byte address. Table 60. LDS Indexed Load/Store

### Single Address Instructions

### Double Address Instructions

The double address instructions are: LOAD_2ADDR*, STORE_2ADDR*, and STOREXCHG_2ADDR_*. The address comes from VGPR, and both VGPR[ADDR] and InstOffset are byte addresses. At the time of wave creation, LDS_BASE is assigned to the physical LDS region owned by this wave or work-group.

### DS_{LOAD,STORE}_ADDTID Addressing

The "ADDTID" (add thread-id) is a separate form where the base address for the instruction is common to all threads, but then each thread has a fixed offset added in based on its thread-ID within the wave. This can allow a convenient way to quickly transfer data between VGPRs and LDS without having to use a VGPR to supply an address.
