# 10.8. Vector Memory Instruction Data Dependencies

_(pages 116-116)_

When a VM instruction is issued, it schedules the reads of address and store-data from VGPRs to be sent to the texture unit. Any ALU instruction that attempts to write this data before it has been sent to the texture unit is stalled. The shader developer’s responsibility to avoid data hazards associated with VMEM instructions include waiting for VMEM load instruction completion before reading data fetched from the cache (VMCNT and VSCNT). This is explained in the section: Data Dependency Resolution
