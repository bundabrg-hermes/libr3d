# 15.9. Flat Formats

_(pages 192-192)_

Flat memory instructions come in three versions:

### FLAT

memory address (per work-item) may be in global memory, scratch (private) memory or shared memory (LDS)

### GLOBAL

same as FLAT, but assumes all memory addresses are global memory.

### SCRATCH

same as FLAT, but assumes all memory addresses are scratch (private) memory. The microcode format is identical for each, and only the value of the SEG (segment) field differs.
