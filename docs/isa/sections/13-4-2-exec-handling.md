# 13.4.2. EXEC Handling

_(pages 143-143)_

GDS / GWS is now only a single lane wide. If the EXEC mask has more than one bit set to 1, hardware behaves as if only EXEC had only one "1" in it: the least significant one. GDS / GWS opcodes are not skipped when EXEC==0. For these opcodes, if EXEC==0, the hardware acts as if EXEC==0…001 for the instruction: ORDERED_COUNT / GWS_INIT / SEMA_BR/GWS_BARRIER For other GDS / GWS opcodes, the instruction is sent with EXE==0, nothing is sent to or returned from GDS/GWS. In hardware, data is sent but it is ignored and data is returned and ignored in order to keep LGKMcnt working.
