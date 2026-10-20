# 7.2.1. Non-Standard Uses of Operand Fields

_(pages 66-67)_

A few instructions use the operand fields in non-standard ways: The readlane lane-select is limited to the valid range of lanes (0-31 for wave32, 0-63 for wave64) by ignoring upper bits of the lane number.

### Inline constants with DOT2_F16_F16 and DOT2_BF16_BF16

For these 2 instructions, the inline constant for sources 0 and 1 replicate the inline constant value into bits[31:16]. For source2, the OPSEL bit is used to control replication or not (gets zero if not replicating low bits).
