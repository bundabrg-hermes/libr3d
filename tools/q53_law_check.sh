#!/usr/bin/env bash
# q53_law_check.sh -- debug-gated Q53 class-gate self-check (zero release cost:
# tool-only; the .so carries no check code). Runs the fold and plain decode arms
# at PINNED splits on the standard fixture (tools/q53_probe.cpp, seed-fixed) and
# asserts the booked Q53 equivalence class on the final bf16 outputs:
#   1. every drifting (tok, head) satisfies parity(tok) != parity(head % gqa);
#   2. every drift element is <= 1 bf16 ulp apart, or both values are within
#      2e-6 absolute (the near-zero softmax-tail class);
#   3. drift rate stays under 0.5% of elements (both arms are individually
#      oracle-class; the rate is a canary, not the law).
# Exit 0 = inside the class. Non-zero = the booked law has been violated by a
# code or toolchain change: re-open Q53.
# Usage: bash q53_law_check.sh   (host needs docker + radiance-dev image; card may co-tenant)
set -uo pipefail
REPO=${REPO:-$HOME/projects/libr3d/libr3d}
OUT=${OUT:-$HOME/scratch/r3d/q53}
IMG=${IMG:-radiance-dev:v1.11.0}
SO=${SO:-/rh158/kernels/libr3d.so}
mkdir -p "$OUT"
INNER=$(cat <<'INNER'
set -u; cd /s2
rm -f /s2/q53_probe
hipcc -O1 -std=c++20 -ffp-contract=off -I/opt/radiance/include/radiance -w /work/tools/q53_probe.cpp -o /s2/q53_probe -ldl 2>&1 | tail -3
for arm in fold plain; do
  env R3D_FORCE_SPLITS=32 $([ $arm = plain ] && echo R3D_NO_GQA=1) BENCH_HD=256 \
    LIBR3D_SO=/rh158/kernels/libr3d.so /s2/q53_probe dump q53lc_$arm 16 4 4 4096 2>&1 | grep -E "DUMP|FATAL|ABORT"
done
python3 - <<'PY'
import numpy as np, sys
def out(tag): return np.fromfile(f"/s2/q53lc_{tag}.out", dtype=np.uint16).reshape(4, 16, 256)
x, y = out("fold"), out("plain")
gqa = 4
d = (x != y)
tok = np.arange(4)[:, None, None]; head = np.arange(16)[None, :, None]
parity_ok = (tok % 2) != ((head % gqa) % 2)
viol_class = int((d & ~parity_ok).sum())
def t(u):
    u = u.astype(np.int64)
    return np.where((u >> 15) & 1, -(u & 0x7fff), u & 0x7fff)
ulp = np.abs(t(x).astype(np.int64) - t(y).astype(np.int64))
def val(u):
    u = u.astype(np.int64); e = (u >> 7) & 0xff; m = u & 0x7f
    v = np.where(e > 0, np.ldexp(1.0 + m / 128.0, e.astype(float) - 127), m * 2.0 ** -133)
    return np.where((u >> 15) & 1, -v, v)
big = d & (ulp > 1) & (np.maximum(np.abs(val(x)), np.abs(val(y))) > 2e-6)
n = int(d.sum()); total = d.size
print(f"law-check: drift={n}/{total} ({100.0*n/total:.2f}%) parity-violations={viol_class} "
      f"out-of-ulp-violations={int(big.sum())}")
ok = viol_class == 0 and int(big.sum()) == 0 and n <= 0.005 * total
print("Q53-LAW:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
INNER
)
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -e ROCR_VISIBLE_DEVICES=1 \
  -v "$REPO":/work:ro \
  -v "$HOME/projects/libr3d/libr3d/build_v158/radiance_home":/rh158:ro \
  -v "$OUT":/s2 "$IMG" bash -c "$INNER"
rc=$?
echo "q53_law_check exit=$rc"
exit $rc
