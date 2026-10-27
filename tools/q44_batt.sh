#!/usr/bin/env bash
# q44_batt.sh -- v159 battery: lever-0 (withdrawal) + lever-1 legs vs standing v153 packs.
# Phantom-pass law: every leg runs R3D_DEBUG_GEMM=1 and liveness is COUNTED per leg.
# Serial legs, one at a time (solo-runner law). Export the build under test (gate.sh law).
set -uo pipefail
cd /home/brendan/projects/libr3d/libr3d
export R3D_BUILD=build_v159
G=tools/gate.sh
S=~/scratch/r3d
LOG=$S/q44_batt.log
: > $LOG

run_pack() { # run_pack <packname> <gate args...>  (arm/LMAP/FIXTURE from caller env)
  local p=$1; shift; local out=$S/q44_${arm}_${p}.out
  export R3D_Q8X_LMAP=$LMAP R3D_DEBUG_GEMM=1
  if bash $G "$@" > "$out" 2>&1; then echo "-- $arm $p OK" | tee -a $LOG; else echo "-- $arm $p FAIL-RC" | tee -a $LOG; fi
  grep -c "arm launched" "$out" | xargs echo "   liveness $arm $p =" | tee -a $LOG
  grep -m1 "arm launched" "$out" | sed 's/^/   /' | tee -a $LOG
}

# the 7 standing packs (q18_batt pattern); reports land as $S/<name>.md
for arm in l0 l1; do
  case $arm in l0) LMAP=0;; l1) LMAP=1;; esac
  run_pack kernels   stock  v159_${arm}_kernels
  run_pack full      full   v159_${arm}_full
  run_pack attr      op gemm_nt_q v159_${arm}_attr
  R3D_FIXTURE=/scratch/27b_dflash2.rkb run_pack df2 op gemm_nt_q v159_${arm}_df2
  R3D_FIXTURE=/scratch/27b_df3.rkb     run_pack df3 op gemm_nt_q v159_${arm}_df3
  run_pack fp8       fp8    v159_${arm}_fp8
  R3D_FIXTURE=/scratch/mxfp4_nodraft.rkb run_pack attn_bf op attn_paged v159_${arm}_attn_bf
done

echo "== fingerprint diffs vs standing v153 packs ==" | tee -a $LOG
for p in attr df2 df3 kernels full fp8 attn_bf; do
  for arm in l0 l1; do
    d=$(diff <(grep -v "generated\|elapsed\|host\|date\|budget" $S/v153_$p.md 2>/dev/null) \
             <(grep -v "generated\|elapsed\|host\|date\|budget" $S/v159_${arm}_$p.md 2>/dev/null) | wc -l)
    echo "$p $arm vs v153: difflines=$d" | tee -a $LOG
  done
done
echo "== BATTERY DONE $(date -u +%H:%M:%S) ==" | tee -a $LOG
