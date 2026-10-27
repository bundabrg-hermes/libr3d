#!/usr/bin/env bash
# Q46 decode identity + natural-split hash sweep. Card legs run INSIDE
# radiance-dev with ROCR_VISIBLE_DEVICES=1 (the gfx1100 XTX after the slot move;
# the gate.sh phantom law: any leg on the wrong device is not a result). HARD
# assertion below replicates gate.sh's: exactly 1 visible device, gfx1100.
# Hash/parity legs only -- no timing (safe to share the card).
#  (1) rsk (repro_attn_splitk) forced-split matrix x {v153 .so, v154 .so} x policy:
#      v153 vs v154 with the SAME forced split count must be IDENTICAL (the
#      merge-patch bit-identity gate; REPRO_CTX=300 lands in the interleaved-inert
#      cut class where the floor cut puts live splits past inert ones).
#  (2) bench_attn_decode parity (natural split resolution) x ctx grid x arms:
#      spl (DBG line), rel_l2, yhash. Rows whose spl did not move MUST hash-equal.
set -uo pipefail
REPO=$HOME/projects/libr3d/libr3d
if [ -z "${Q46_IN_CONTAINER:-}" ]; then
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    -e ROCR_VISIBLE_DEVICES=1 -e Q46_IN_CONTAINER=1 \
    -v $REPO:/work -w /work --shm-size=4g --entrypoint /bin/bash \
    radiance-dev:v1.11.0 /work/tools/q46_id.sh
  exit $?
fi
cd /work
mkdir -p q46logs q46bin; LOG=q46logs/q46_id.log; : > $LOG
say() { echo "$@" | tee -a $LOG; }
SO153=/work/build_v153/radiance_home/kernels/libr3d.so
SO154=/work/build_v154/radiance_home/kernels/libr3d.so
cat > q46bin/dev0.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n=0;hipGetDeviceCount(&n);printf("%d device(s)",n);
for(int i=0;i<n;i++){hipDeviceProp_t p;hipGetDeviceProperties(&p,i);
printf(" [%d:%s]",i,p.gcnArchName);}printf("\n");return 0;}
EOF
hipcc -O1 --offload-arch=gfx1100 q46bin/dev0.hip -o q46bin/dev0 || exit 1
dv=$(./q46bin/dev0); echo "$dv" | tee -a $LOG
case "$dv" in "1 device(s) [0:gfx1100]") ;; *) echo "FATAL: not gfx1100 alone -- phantom leg, abort" | tee -a $LOG; exit 3;; esac
say "== build harnesses (project correctness flags, -Wall) $(date -u +%H:%M:%S) =="
for t in repro_attn_splitk bench_attn_decode; do
  hipcc -O1 -std=c++20 -Wall -ffp-contract=off -I/opt/radiance/include/radiance \
        tools/$t.hip -o q46bin/$t -ldl || { say "BUILD FAIL $t"; exit 1; }
done
say "harness builds OK"
say "-- leg 1: rsk merge-identity matrix --"
for HD in 128 256; do
 for SPL in 1 2 4 8 16 32 48 64; do
  for CTX in 300 4096; do
   for V in 153 154; do
    for POL in 0 1; do
      f=q46logs/rsk_${HD}_${SPL}_${CTX}_v${V}_p${POL}
      SO=$SO154; [ $V = 153 ] && SO=$SO153
      LIBR3D_SO=$SO R3D_SPLIT_POLICY=$POL REPRO_HD=$HD REPRO_SPL=$SPL REPRO_CTX=$CTX \
        ./q46bin/rsk > $f.txt 2>&1 || say "rsk RUNFAIL $f"
    done
   done
  done
 done
done
say "rsk-diff-format-sample:"; head -6 q46logs/rsk_128_8_300_v154_p0.txt | tee -a $LOG
for HD in 128 256; do for SPL in 1 2 4 8 16 32 48 64; do for CTX in 300 4096; do
  a=q46logs/rsk_${HD}_${SPL}_${CTX}_v153_p0.txt; b=q46logs/rsk_${HD}_${SPL}_${CTX}_v154_p0.txt
  c=q46logs/rsk_${HD}_${SPL}_${CTX}_v154_p1.txt
  d1=$(diff $a $b | wc -l); d2=$(diff $b $c | wc -l)
  say "rsk hd=$HD spl=$SPL ctx=$CTX: v153-vs-v154 difflines=$d1 policy0-vs-1 difflines=$d2"
done; done; done
say "-- leg 2: natural-split parity sweep (policy arms race) --"
for CTX in 512 1024 2048 4096 8192; do
 for POL in 0 1; do
  for V in 153 154; do
    f=q46logs/par_${CTX}_v${V}_p${POL}
    SO=$SO154; [ $V = 153 ] && SO=$SO153
    LIBR3D_SO=$SO R3D_SPLIT_POLICY=$POL R3D_DEBUG_DECODE=1 \
      ./q46bin/bench_attn_decode parity 32 8 1 $CTX > $f.txt 2>&1 || say "par RUNFAIL $f"
    spl=$(grep -oE "spl=[0-9]+" $f.txt | head -1 | grep -oE "[0-9]+")
    arm=$(grep -oE "split-policy arm=[a-z]+" $f.txt | head -1)
    say "par ctx=$CTX v$V p$POL: ${arm:-ARM-LINE-MISSING} spl=${spl:-none} $(grep PARITY $f.txt)"
  done
 done
done
say "-- h256 natural parity at 8192 (p48 anomaly shape) --"
for POL in 0 1; do
  for EXTRA in "none" "force48"; do
    f=q46logs/par256_8192_p${POL}_${EXTRA}
    if [ $EXTRA = force48 ]; then F=48; else F=0; fi
    LIBR3D_SO=$SO154 R3D_SPLIT_POLICY=$POL R3D_FORCE_SPLITS=$F R3D_DEBUG_DECODE=1 BENCH_HD=256 \
      ./q46bin/bench_attn_decode parity 16 2 1 8192 > $f.txt 2>&1 || say "par256 RUNFAIL $f"
    spl=$(grep -oE "spl=[0-9]+" $f.txt | head -1 | grep -oE "[0-9]+")
    say "par256 ctx=8192 p$POL $EXTRA: spl=${spl:-none} $(grep PARITY $f.txt)"
  done
done
say "== ALL DONE $(date -u +%H:%M:%S) =="
