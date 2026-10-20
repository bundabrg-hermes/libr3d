# 3.4.5. SCC: Scalar Condition Code

_(pages 32-33)_

Many scalar ALU instructions set the Scalar Condition Code (SCC) bit, indicating the result of the operation. Compare operations: 1 = true Arithmetic operations: 1 = carry out Bit/logical operations: 1 = result was not zero Move: does not alter SCC The SCC can be used as the carry-in for extended-precision integer arithmetic, as well as the selector for conditional moves and branches.
