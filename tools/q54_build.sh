#!/usr/bin/env bash
# q54_build.sh -- build the Q54 bench (card-free compile inside radiance-dev)
# and dump a per-kernel VGPR/SGPR/LDS/spill census from the saved temps.
# Law: never rebuild a build dir a running bench dlopens; this only writes $SCR.
set -euo pipefail
SCR=/home/brendan/scratch/r3d/q54
REPO=/home/brendan/projects/libr3d/libr3d
IMG=radiance-dev:v1.11.0
mkdir -p "$SCR"
install -m 0644 "$REPO/tools/bench_fp8_skinny.cpp" "$SCR/bench_fp8_skinny.cpp"
md5sum "$SCR/bench_fp8_skinny.cpp" "$REPO/tools/bench_fp8_skinny.cpp" | tee "$SCR/build.md5"
docker run --rm -v "$SCR":/q54 "$IMG" bash -lc '
set -e
cd /q54
rm -rf obj bench_fp8_skinny
mkdir obj
hipcc -O2 --offload-arch=gfx1100 -mno-wavefrontsize64 -std=c++17 -ffp-contract=off \
      -I/opt/radiance/include/radiance -w -save-temps=obj \
      /q54/bench_fp8_skinny.cpp -o /q54/bench_fp8_skinny -ldl
echo BUILD-OK
f=$(find obj -name "*gfx1100*.s" | head -1)
if [ -z "$f" ]; then f=$(find obj -name "*.s" ! -name "*preopt*" | head -1); fi
echo "ASM=$f"
awk -f /q54/q54_isa.awk "$f" | grep -i skinny
' 2>&1 | tee "$SCR/build.log"
