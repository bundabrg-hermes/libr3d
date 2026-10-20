# 9.6. Buffer Resource

_(pages 101-102)_

The buffer resource (V#) describes the location of a buffer in memory and the format of the data in the buffer. It is specified in four consecutive SGPRs (4-SGPR aligned) and sent to the texture cache with each buffer instruction. The table below details the fields that make up the buffer resource descriptor. Table 47. Buffer Resource Descriptor

### Unbound Resources

Setting the resource constant to all zeros has the effect of forcing any loads to return zero, and stores to be ignored. This is keyed off the "data-format" being set to zero (INVALID), and for MUBUF the "add_tid_en = false".

### Resource - Instruction mismatch

If the resource type and instruction mismatch (e.g. a buffer constant with an image instruction, or an image resource with a buffer instruction), the instruction is ignored (loads return nothing and stores do not alter memory).
