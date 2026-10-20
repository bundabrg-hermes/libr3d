# 11.1.2. Global

_(pages 123-123)_

Global operations transfer data between VGPR and global memory. Global instructions are similar to Flat, but the programmer is responsible to make sure that no threads access LDS or private space. Because of this, no LDS bandwidth is used by global instructions. Since these instructions do not access LDS, only VMcnt (or VScnt) is used, not LGKMcnt. If a global instruction does attempt to access LDS, the instruction returns MEMVIOL. Global includes two instructions which do not use any VGPRs for addressing, just SGPRs and INST_OFFSET: • GLOBAL_LOAD_ADDTID_B32 • GLOBAL_STORE_ADDTID_B32
