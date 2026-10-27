#!/usr/bin/env bash
# q53_law_check.sh -- debug-gated Q53 class-gate self-check (zero release cost:
# tool-only; the .so carries no check code). Runs the fold and plain decode arms
# at PINNED splits on the standard fixture (tools/q53_probe.cpp, seed-fixed) and
# asserts the booked Q53 equivalence class on the final bf16 outputs:
#   1. every drifting (tok, head) satisfies parity(tok) != parity(head % gqa);
#   2. every drift element is <= 1 bf16 ulp apart, or their difference is
#      within 2e-6 absolute (the near-zero softmax-tail class);
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
import struct, sys
def out(tag):
    return struct.unpack(f"<{4*16*256}H", open(f"/s2/q53lc_{tag}.out","rb").read())
x, y = out("fold"), out("plain")
gqa = 4
def tf32(u):                      # bf16 -> order-preserving int (same sign: bit distance)
    return -(u & 0x7fff) if (u >> 15) & 1 else (u & 0x7fff)
def val(u):
    e, m = (u >> 7) & 0xff, u & 0x7f
    v = ((1.0 + m / 128.0) * 2.0 ** (e - 127)) if e else m * 2.0 ** -133
    return -v if (u >> 15) & 1 else v
n = viol_class = big = 0
for i, (a, b) in enumerate(zip(x, y)):
    if a == b: continue
    n += 1
    tok = i // (16 * 256); head = (i // 256) % 16
    if (tok % 2) == ((head % gqa) % 2): viol_class += 1
    if abs(tf32(a) - tf32(b)) > 1 and abs(val(a) - val(b)) > 2e-6: big += 1
total = len(x)
print(f"law-check: drift={n}/{total} ({100.0*n/total:.2f}%) parity-violations={viol_class} "
      f"out-of-ulp-violations={big}")
ok = viol_class == 0 and big == 0 and n <= 0.005 * total
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
