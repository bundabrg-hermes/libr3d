# 6.3. Scalar Condition Code (SCC)

_(pages 58-58)_

The scalar condition code (SCC) is written as a result of executing most SALU instructions. For integer arithmetic it is used as carry/borrow in for extended integer arithmetic. The SCC is set by many instructions: • Compare operations: 1 = true. • Arithmetic operations: 1 = carry out. ◦SCC = overflow for signed add and subtract operations. For add ops, overflow = both operands are of the same sign, and the MSB (sign bit) of the result is different than the sign of the operands. For subtract (A - B), overflow = A and B have opposite signs and the resulting sign is not the same as the sign of A. • Bit/logical operations: 1 = result was not zero.
