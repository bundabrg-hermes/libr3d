# 12.6. Global Data Share

_(pages 139-139)_

Global data Share is similar to LDS, but is a single memory accessible by all waves on the GPU. Global Data Share uses the same instruction format as local data share (indexed operations only - no interpolation or direct loads). Instructions increment the LGKMcnt for all loads, stores and atomics, and decrement LGKMcnt when the instruction completes. GDS instructions support only one active lane per instruction. The first active lane (based on EXEC) is used and others are ignored. M0 is used for: • [15:0] holds SIZE, in bytes • [31:16] holds BASE address in bytes
