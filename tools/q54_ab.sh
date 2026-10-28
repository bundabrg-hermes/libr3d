#!/usr/bin/env bash
# q54_ab.sh <round> [rev] -- Q54 sk window/grid A/B legs on the XTX (ROCR 1).
# One bench PROCESS per shape: in-process arms = library refs (TILE/base SKINNY/GEMV,
# shipped build_v161 md5 be50cd43*, mounted ro -- never rebuilt) + the skinny2 arms
# (SK8V,SK4,SK2,SK1,SKW8,SKW4,SKW2,SKW1). rev=1 crosses the arm order (round crossing).
# Data via env vars only (never `bash script $R`-style $0 binding).
set -u
R=${Q54_R:?round via env Q54_R}
REV=${Q54_REV:-0}
SCR=/home/brendan/scratch/r3d/q54
LIBDIR=/home/brendan/projects/libr3d/libr3d/build_v161/radiance_home/kernels
IMG=radiance-dev:v1.11.0
DEV=(--device /dev/kfd --device /dev/dri --group-add video)
out=$SCR/ab_r${R}.out
: > "$out"
md5sum "$SCR/bench_fp8_skinny" >> "$out" 2>&1
leg() { # leg <label> <M> <K> <N> <reps> <rot> <expected-AB-lines>
  local lbl=$1 M=$2 K=$3 N=$4 reps=$5 rot=$6 exp=$7
  docker run --rm "${DEV[@]}" -e ROCR_VISIBLE_DEVICES=1 \
      -e LIBR3D_SO=/blib/libr3d.so -e F8SK_ROUND="$R" -e F8SK_REV="$REV" \
      -v "$SCR":/q54:ro -v "$LIBDIR":/blib:ro "$IMG" \
      /q54/bench_fp8_skinny "$M" "$K" "$N" "$reps" "$rot" \
      > "$SCR/leg_${lbl}_r${R}.out" 2>&1
  local rc=$?
  local n
  n=$(grep -c '^AB r=' "$SCR/leg_${lbl}_r${R}.out")
  echo "LEG $lbl r=$R rc=$rc ab=$n exp=$exp rev=$REV" >> "$out"
  cat "$SCR/leg_${lbl}_r${R}.out" >> "$out"
  [ "$n" = "$exp" ] || echo "LINECOUNT-FAIL $lbl: got $n expected $exp" >> "$out"
}
# ---- cold regime: one B plane >> 96 MB Infinity Cache ----
leg s1_m14_r4  14 16384 16384 30 4 10
leg s1_m14_r1  14 16384 16384 30 1 10   # rot1 at plane>>IC: L2-assist control (Q50: rot1==rot4)
leg s1_m4_r4    4 16384 16384 30 4 11   # includes GEMV AB line (M<=8)
leg s1_m32_r4  32 16384 16384 30 4 10
leg s2_m14_r4  14 8192  28672 30 4 10
leg s3_m14_r4  14 14336 4096  30 4 10   # Q50 small-N tail class (70.5 GB/s baseline)
# ---- cache-fit regime: plane <= 96 MB IC, same plane re-read ----
leg s3c_m14_r1 14 14336 4096  30 1 10
leg c8k_m14_r1 14 8192  8192  30 1 10   # 64-block shape: sub-96 grid tail
echo "ROUND $R DONE" >> "$out"
