# 11.3. Memory Error Checking

_(pages 125-126)_

Both Texture and LDS can report that an error occurred due to a bad address. This can occur due to: • Invalid address (outside any aperture) • Write to read-only global memory address • Misaligned data (scratch accesses may be misaligned) • Out-of-range address: ◦LDS access with an address outside the range: [ 0, LDS_SIZE-1 ] The policy for threads with bad addresses is: stores outside this range do not write a value, and reads return zero. The aperture check for invalid address occurs before adding any address offsets - it is based only on the base address; the other checks are performed after adding the offsets. Addressing errors from either LDS or TA are returned on their respective "instruction done" busses as MEMVIOL. This sets the wave’s MEMVIOL TrapStatus bit, and also causes an exception (trap).
