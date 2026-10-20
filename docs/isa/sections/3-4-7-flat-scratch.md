# 3.4.7. FLAT_SCRATCH

_(pages 33-34)_

FLAT_SCRATCH is a 64-bit register that holds a pointer to the base of scratch memory for this wave. For waves that have scratch space allocated, wave-launch hardware initializes the FLAT_SCRATCH register with the scratch base address unique to this wave. This register is read-only, except while in the trap handler where it is writable. The value is a byte address and must be 256byte aligned. If the wave has no scratch space allocated, then reading FLAT_SCRATCH returns zero. The value for FLAT_SCRATCH is computed in hardware and initialized for any wave that has scratch space allocated: scratch_base = scratch_base[63:0] + spi_scratch_offset[31:0] FLAT_SCRATCH_LO = scratch_base [31:0] FLAT_SCRATCH_HI = scratch_base [63:32]
