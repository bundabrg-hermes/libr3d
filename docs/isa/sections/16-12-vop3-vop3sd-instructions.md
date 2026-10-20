# 16.12. VOP3 & VOP3SD Instructions

_(family intro + opcode table, pages 374-506)_

VOP3 instructions use one of two encodings:

### VOP3SD

this encoding allows specifying a unique scalar destination, and is used only for: V_ADD_CO_U32 V_SUB_CO_U32 V_SUBREV_CO_U32 V_ADDC_CO_U32 V_SUBB_CO_U32 V_SUBBREV_CO_U32 V_DIV_SCALE_F32 V_DIV_SCALE_F64 V_MAD_U64_U32 V_MAD_I64_I32

### VOP3

all other VALU instructions use this encoding
