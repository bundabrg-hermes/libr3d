# q54_isa.awk -- per-kernel resource census from LLVM -S asm.
# Tracks the current function from "; -- Begin function <name>", prints AMDGPU
# resource directives and counts scratch (spill) instructions per function.
/-- Begin function/ { fn = $NF }
/\.amdgpu_vgpr |\.amdgpu_sgpr |\.amdgpu_lds_size|vgpr_spill_count|sgpr_spill_count|\.vgpr_count|\.vgpr_spill_count|\.sgpr_count|\.private_segment_fixed_size|\.group_segment_fixed_size/ {
    printf "META %s | %s\n", fn, $0
}
/scratch_load|scratch_store/ { c[fn]++ }
END { for (k in c) printf "SCRATCH %s sites=%d\n", k, c[k] }
