---
mnemonic: S_DELAY_ALU
section: "16.5. SOPP Instructions"
format: SOPP
opcode: "7"
pages: [245, 246]
source: RDNA3 ISA PDF p.245-246
---

# S_DELAY_ALU (opcode 7)

Insert delay between dependent SALU/VALU instructions. The SIMM16 argument is encoded as:

### INSTID0 = SIMM16[3:0]

Hazard to delay for with the next VALU instruction.

### INSTSKIP = SIMM16[6:4]

Identify the VALU instruction that the second delay condition applies to.

### INSTID1 = SIMM16[10:7]

Hazard to delay for with the VALU instruction identified by INSTSKIP. Legal values for the InstID0 and InstID1 fields are:

### INSTID_NO_DEP (0x0)

No dependency on any prior instruction.

### INSTID_VALU_DEP_1 (0x1)

Dependent on previous VALU instruction, 1 instruction back.

### INSTID_VALU_DEP_2 (0x2)

Dependent on previous VALU instruction, 2 instructions back.

### INSTID_VALU_DEP_3 (0x3)

Dependent on previous VALU instruction, 3 instructions back.

### INSTID_VALU_DEP_4 (0x4)

Dependent on previous VALU instruction, 4 instructions back.

### INSTID_TRANS32_DEP_1 (0x5)

Dependent on previous TRANS32 instruction, 1 instruction back.

### INSTID_TRANS32_DEP_2 (0x6)

Dependent on previous TRANS32 instruction, 2 instructions back.

### INSTID_TRANS32_DEP_3 (0x7)

Dependent on previous TRANS32 instruction, 3 instructions back.

### INSTID_FMA_ACCUM_CYCLE_1 (0x8)

Single cycle penalty for FMA accumulation (reserved).

### INSTID_SALU_CYCLE_1 (0x9)

1 cycle penalty for a prior SALU instruction.

### INSTID_SALU_CYCLE_2 (0xa)

2 cycle penalty for a prior SALU instruction (reserved).

### INSTID_SALU_CYCLE_3 (0xb)

3 cycle penalty for a prior SALU instruction (reserved). Legal values for the InstSkip field are:

### INSTSKIP_SAME (0x0)

Apply second dependency to same instruction (2 dependencies on one instruction).

### INSTSKIP_NEXT (0x1)

Apply second dependency to next instruction (no skip).

### INSTSKIP_SKIP_1 (0x2)

Skip 1 instruction then apply dependency.

### INSTSKIP_SKIP_2 (0x3)

Skip 2 instructions then apply dependency.

### INSTSKIP_SKIP_3 (0x4)

Skip 3 instructions then apply dependency.

### INSTSKIP_SKIP_4 (0x5)

Skip 4 instructions then apply dependency. This instruction describes dependencies for two instructions, directing the hardware to insert delay if the dependent instruction was issued too recently to forward data to the second. S_DELAY_ALU instructions record the required delay with respect to a previous VALU instruction and indicate data dependencies that benefit from having extra idle cycles inserted between them. These instructions are optional: without them the program still functions correctly but performance may suffer when multiple waves are in flight; IB may issue dependent instructions that stall in the ALU, preventing those cycles from being utilized by other wavefronts. If enough independent instructions are between dependent ones then no delay is necessary and this instruction may be omitted. For wave64 the compiler may not know the status of the EXEC mask and hence does not know if instructions require 1 or 2 passes to issue. S_DELAY_ALU encodes the type of dependency so that hardware may apply the correct delay depending on the number of active passes. S_DELAY_ALU may execute in zero cycles. To reduce instruction stream overhead the S_DELAY_ALU instructions packs two delay values into one instruction, with a "skip" indicator so the two delayed instructions don't need to be back-to-back. S_DELAY_ALU is illegal inside of a clause created by S_CLAUSE. Example:
