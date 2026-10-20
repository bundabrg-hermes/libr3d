# 12.7. Alignment and Errors

_(pages 140-140)_

GDS and LDS operations (both direct & indexed) report Memory Violation (memviol) for misaligned atomics. LDS handles misaligned indexed reads & writes, but only when SH_MEM_CONFIG. alignment_mode == UNALIGNED. Atomics must be aligned. LDS Alignment modes (config-reg controlled, in SH_MEM_CONFIG): • ALIGNMENT_MODE_DWORD: Automatic alignment to multiple of element size • ALIGNMENT_MODE_UNALIGNED: No alignment requirements.
