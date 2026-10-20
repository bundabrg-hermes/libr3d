# 3.4.8. Hardware Internal Registers

_(pages 34-34)_

These registers are read-only and can be accessed by the S_GETREG instruction. They return information about hardware allocation and status. HW_ID and the various *_BASE values are not predictable and may change over the lifetime of a wave if context-switching can occur.

### HW_ID1

### HW_ID2

Other S_GETREG, S_SETREG targets: Note: TMA and TBA are read using S_SENDMSG_RTN.
