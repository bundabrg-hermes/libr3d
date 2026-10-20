---
mnemonic: DS_BVH_STACK_RTN_B32
section: "16.15. LDS & GDS Instructions"
format: LDS
opcode: "173"
pages: [541]
source: RDNA3 ISA PDF p.541-541
---

# DS_BVH_STACK_RTN_B32 (opcode 173)

Ray tracing involves traversing a BVH which is a kind of tree where nodes have up to 4 children. Each shader thread processes one child at a time, and overflow nodes are stored temporarily in LDS using a stack. This instruction supports pushing/popping the stack to reduce the number of VALU instructions required per traversal and reduce VMEM bandwidth requirements. The LDS stack address is computed using values packed into ADDR and part of OFFSET1. ADDR carries the stack address for the lane. OFFSET1[5:4] contains stack_size[1:0] -- this value is constant for all lanes and is patched into the shader by software. Valid stack sizes are {8, 16, 32, 64}. A new stack address is returned to ADDR --- note that this VGPR is an in-out operand. DATA0 contains the last node pointer for BVH. DATA1 contains up to 4 valid data DWORDs for each thread. At a high level the first 3 DWORDs (DATA1[0:2]) is pushed to the stack if they are valid, and the last DWORD (DATA1[3]) is returned. If the last DWORD is invalid then pop the stack and return the value from memory. In general this instruction performs the following :
