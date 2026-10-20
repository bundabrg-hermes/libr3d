# 7.4. 16-bit Math and VGPRs

_(pages 74-74)_

VALU instructions that operate on 16-bit data (non-packed) can separately address the two halves of a 32-bit VGPR. 16-bit VGPR-pairs are packed into a 32-bit VGPRs: the 32-bit VGPR "V0" contains two 16-bit VGPRs: "V0.L" representing V0[15:0] and "V0.H" representing V0[31:16]. How this addressing is encoded in the ISA varies by the instruction encoding: The 16-bit instructions can be encoded using VOP1/2/C as well as VOP3/VOP3P/VINTERP.

### 16bit VGPR Naming

The 32-bit VGPR is "V0". The two halves are called "V0.L" and "V0.H".

### VOP1, VOP2, VOPC Encoding

16-bit VGPRs are encoded as: SRC/DST[6:0] = 32-bit VGPR address; SRC/DST[7] = (1=hi, 0=lo half) In this encoding, only 256 16-bit VGPRs can be addressed.

### VOP3, VOP3P, VINTERP

16-bit VGPRs are encoded as: SRC/DST[7:0] = 32-bit VGPR address, OPSEL = high/low. In this encoding, a wave can address 512 16-bit VGPRs. The packing shown below allows reading or writing in one cycle: • 32 lanes of one 32-bit VGPR: V0 • 64 lanes of one 16-bit VGPR: V0.L • 32 lanes of two 16-bit VGPRs (a pair, as used by packed math): V0.L and V0.H
