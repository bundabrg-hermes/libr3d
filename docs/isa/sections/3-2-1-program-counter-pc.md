# 3.2.1. Program Counter (PC)

_(pages 23-23)_

The Program Counter is a DWORD-aligned byte address that points to the next instruction to execute. When a wave is created the PC is initialized to the first instruction in the program. There are a few instructions to interact directly with the PC: S_GETPC_B64, S_SETPC_B64, S_CALL_B64, S_RFE_B64 and S_SWAPPC_B64. These transfer the PC to and from an even-aligned SGPR pair (sign-extended). Branches jump to (PC_of_the_instruction_after_the_branch + offset*4). Branches, GET_PC and SWAP_PC are PC- relative to the next instruction, not the current one. S_TRAP, on the other hand, saves the PC of the S_TRAP instruction itself. During wave debugging, the program counter may be read. The PC points to the next instruction to issue. All prior instructions have been issued but may or may not have completed execution.
