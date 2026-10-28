#!/usr/bin/env bash
# q54_isa.sh -- per-kernel VGPR/SGPR/LDS/spill census from the BUILT binary.
# The llvm tools are OFF PATH in radiance-dev:v1.11.0 (/opt/rocm/lib/llvm/bin).
set -euo pipefail
SCR=/home/brendan/scratch/r3d/q54
docker run --rm -v "$SCR":/q54 radiance-dev:v1.11.0 bash -lc '
set -e
cd /q54
rm -rf isa && mkdir isa
/opt/rocm/lib/llvm/bin/roc-obj /q54/bench_fp8_skinny > /q54/isa/rocobj.log 2>&1 || true
mv *.hipfb* isa/ 2>/dev/null || true
for img in $(ls isa/*amdgcn*.img 2>/dev/null || ls isa/*.img 2>/dev/null); do
  echo "===== $img"
  /opt/rocm/lib/llvm/bin/llvm-readelf --notes "$img" > /q54/isa/notes.txt 2>/dev/null
  grep -E "\.name:|\.vgpr_count:|\.sgpr_count:|spill_count|group_segment_fixed_size|private_segment_fixed_size" /q54/isa/notes.txt
  echo "scratch_instrs=$(/opt/rocm/lib/llvm/bin/llvm-objdump --disassemble "$img" 2>/dev/null | grep -c "scratch_\(load\|store\)" || true)"
done' 2>&1 | tee "$SCR/isa_census.txt"
