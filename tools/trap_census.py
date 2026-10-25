#!/usr/bin/env python3
"""Q19 compiler-trap counts census over the shipped gfx1100 disassembly.

Count-only pass (no fixes): per kernel, over the gfx1100 slice of each
device object, census the donor trap families from docs/LIBR35D_TRIAGE.md
delta4:

  T1 true16 no-op extraction pairs: v_lshrrev_b32 imm16, *_d16_* half-moves,
     v_alignbyte_b32, and adjacent pairs (lshr within PAIR_WINDOW of an
     extraction mov) — the donor's volatile-asm LICM signature's residue.
  T2 guards inside unrolled K steps: s_cbranch_* within KSTEP_WINDOW
     instructions of a v_wmma_* (per-WMMA exec-mask branch class), reported
     as (branches-near-wmma, wmma-count) pairs.
  T3 natural-order LDS row + dword rename: register-to-register
     v_mov_b32 vN, vM with N==M+1 (dword rename) appearing within
     RENAME_WINDOW after a ds_read_*.
  T4 remaining ds_bpermute count per kernel (post-Q6 attribution check).
  T5 adjacent ds_read_b64 pair with offsets 8 bytes apart (unmerged b128),
     inside a window, only outside inline asm by construction (objdump).

Usage: trap_census.py <objdump-binary> <object...>
Env OBJDUMP_ARCH selects the offloading arch (default gfx1100).
Prints one table; exits non-zero if an object yields NO instructions
(empty-parse trap guard, per COVERAGE law).
"""
import re
import subprocess
import sys
import os

# OBJDUMP_ARCH: extra objdump flag for fat/universal inputs; plain gfx1100
# slices (post clang-offload-bundler) need none.
ARCH = os.environ.get("OBJDUMP_ARCH", "")
PAIR_WINDOW = 6
KSTEP_WINDOW = 8
RENAME_WINDOW = 10

KERNEL_RE = re.compile(r"^[0-9a-fA-F]+ <(.+)>:")
# ROCm llvm-objdump prints "  mnemonic operands // addr: hex" — and the
# addr/encoding hex is UPPERCASE on some builds ([0-9a-f] silently drops
# every line whose addr contains a-f: measured 8x undercount). Accept both.
INSN_RE = re.compile(r"^\t+([a-z][a-z0-9_.]*)\s(.*?)\s*//\s*[0-9a-fA-F]+:")
# Loose companion for the coverage guard: any printed instruction line.
LOOSE_RE = re.compile(r"^\t+[a-z][a-z0-9_.]*\s.*//\s*[0-9a-fA-F]+:")


def disassemble(objdump, obj):
    cmd = [objdump, "-d"]
    if ARCH:
        cmd.append(f"--offloading-arch={ARCH}")
    cmd.append(obj)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def kernels(text):
    """Split objdump text into (name, [ (mnemonic, operand_str), ... ]), plus
    the loose-line count for the coverage guard."""
    out, cur, body = [], None, []
    loose = 0
    for line in text.splitlines():
        m = KERNEL_RE.match(line)
        if m:
            if cur:
                out.append((cur, body))
            cur, body = m.group(1), []
            continue
        if LOOSE_RE.match(line):
            loose += 1
        m = INSN_RE.match(line)
        if m and cur is not None:
            body.append((m.group(1), m.group(2)))
    if cur:
        out.append((cur, body))
    return out, loose


def reg_num(tok):
    m = re.match(r"v(\d+)$", tok)
    return int(m.group(1)) if m else None


def extraction_just_mn(mn, ops):
    """True for v_lshrrev_b32 with the shift imm = 16 (imm-first RDNA form
    `v, 16, v` or the tail form)."""
    return (mn == "v_lshrrev_b32"
            and (re.match(r"[^,]+,\s*16\s*,", ops)
                 or re.search(r",\s*16\s*$", ops)))


def census(name, body):
    n = len(body)

    def extraction(idx):
        mn, ops = body[idx]
        if mn == "v_alignbyte_b32" or "_d16" in mn:
            return True
        return (mn == "v_lshrrev_b32"
                and (re.match(r"[^,]+,\s*16\s*,", ops)
                     or re.search(r",\s*16\s*$", ops)))

    is_ext = [extraction(i) for i in range(n)]
    t1 = {"lshr16": sum(1 for mn, ops in body if extraction_just_mn(mn, ops)),
          "d16": sum(1 for mn, _ in body if "_d16" in mn),
          "alignbyte": sum(1 for mn, _ in body if mn == "v_alignbyte_b32"),
          "pairs": 0}
    for i in range(n):
        if not is_ext[i]:
            continue
        lo, hi = max(0, i - PAIR_WINDOW), min(n, i + PAIR_WINDOW + 1)
        t1["pairs"] += sum(1 for j in range(lo, hi) if j != i and is_ext[j])
    t1["pairs"] //= 2  # each pair counted from both ends

    t2 = {"wmma": sum(1 for mn, _ in body if mn.startswith("v_wmma")),
          "branch_near_wmma": 0, "execz": 0}
    for i, (mn, ops) in enumerate(body):
        if mn.startswith("s_cbranch"):
            lo, hi = max(0, i - KSTEP_WINDOW), min(n, i + KSTEP_WINDOW + 1)
            if any(body[j][0].startswith("v_wmma") for j in range(lo, hi)):
                t2["branch_near_wmma"] += 1
                if "exec" in ops or "execz" in mn:
                    t2["execz"] += 1

    # T3: dword rename (pure reg mov, |dst-src|==1) shortly after an LDS
    # load. RDNA mnemonics are ds_load_*/ds_store_* (NOT ds_read_*) and the
    # movs carry _e32/_e64 suffixes — exact matches on GCN names silently
    # report zero on the whole tree.
    t3 = 0
    for i, (mn, ops) in enumerate(body):
        if not mn.startswith("v_mov_b32"):
            continue
        m = re.match(r"\s*v(\d+)\s*,\s*v(\d+)\s*$", ops)
        if not m:
            continue
        d, s = int(m.group(1)), int(m.group(2))
        if abs(d - s) != 1:
            continue
        lo = max(0, i - RENAME_WINDOW)
        if any(body[j][0].startswith(("ds_load", "ds_read"))
               for j in range(lo, i)):
            t3 += 1

    t4 = sum(1 for mn, _ in body if mn.startswith("ds_bpermute"))

    # T5: adjacent ds_load_b64 pair, same base vreg, offsets 8 B apart —
    # an unmerged b128 (mergeable only outside inline asm, donor law).
    t5 = 0
    b64_re = re.compile(r",\s*v(\d+)\s+offset:(-?\d+)")
    for i in range(n - 1):
        for w in (1, 2):
            if i + w >= n:
                break
            mn1, o1 = body[i]
            mn2, o2 = body[i + w]
            if not (mn1 in ("ds_load_b64", "ds_read_b64")
                    and mn2 in ("ds_load_b64", "ds_read_b64")):
                continue
            m1, m2 = b64_re.search(o1), b64_re.search(o2)
            if m1 and m2 and m1.group(1) == m2.group(1) \
                    and int(m1.group(2)) + 8 == int(m2.group(2)):
                t5 += 1
                break

    # T6 (added after first run): ds_swizzle is the dominant WMMA fragment
    # feed on gfx11 (rocwmma column-major B via lane shuffle). Counted to
    # attribute DS-pipe traffic per kernel.
    t6 = sum(1 for mn, _ in body if mn.startswith("ds_swizzle"))

    return t1, t2, t3, t4, t5, t6


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    objdump, objs = sys.argv[1], sys.argv[2:]
    rows, total_insns, total_kernels = [], 0, 0
    loose_total = 0
    for obj in objs:
        text = disassemble(objdump, obj)
        ks, loose = kernels(text)
        loose_total += loose
        total_kernels += len(ks)
        obj_insns = sum(len(b) for _, b in ks)
        total_insns += obj_insns
        for name, body in ks:
            t1, t2, t3, t4, t5, t6 = census(name, body)
            if not (t1["lshr16"] or t1["d16"] or t1["alignbyte"]
                    or t2["branch_near_wmma"] or t3 or t4 or t5 or t6
                    or t2["wmma"]):
                continue
            rows.append((os.path.basename(obj).split("-hip")[0], name,
                         len(body), t1["lshr16"], t1["d16"], t1["alignbyte"],
                         t1["pairs"], t2["wmma"], t2["branch_near_wmma"],
                         t3, t4, t5, t6))
    if total_insns == 0:
        print("FATAL: objdump produced ZERO instructions — arch filter "
              "matched nothing, parse vacuous. Not reporting.")
        return 1
    # Coverage guard: any printed instruction line NOT captured by the strict
    # parse means a format drift (uppercase hex, new operand forms) silently
    # undercounts. Refuse to report rather than report quietly-wrong numbers.
    if loose_total and total_insns / loose_total < 0.95:
        print(f"FATAL: parse coverage {total_insns}/{loose_total} "
              f"({100.0*total_insns/loose_total:.1f}%) — strict regex misses "
              f"printed instructions. Fix the parse, don't report it.")
        return 1
    hdr = ("file", "kernel", "insns", "lshr16", "d16", "alignb", "pairs",
           "wmma", "br@wmma", "rename", "bperm", "b64adj", "swzl")
    # demangle kernel names (truncating mangled strings collapses template
    # instantiations into indistinguishable rows)
    names = sorted({r[1] for r in rows})
    pretty = {}
    try:
        p = subprocess.run(["c++filt"], input="\n".join(names),
                           capture_output=True, text=True)
        for n, d in zip(names, p.stdout.splitlines()):
            short = re.sub(r"^void\s+", "", d)
            pretty[n] = short[:78]
    except FileNotFoundError:
        pretty = {n: n for n in names}
    W = (14, 78, 7, 6, 5, 6, 5, 6, 8, 6, 5, 6, 6)
    print(" ".join(h.rjust(w) for h, w in zip(hdr, W)))
    for r in rows:
        cols = (r[0][:14], pretty.get(r[1], r[1])[:78]) + r[2:]
        print(" ".join(str(c).rjust(w) for c, w in zip(cols, W)))
    print(f"\nTOTALS scanned={total_kernels} kernels-listed={len(rows)} "
          f"instructions-parsed={total_insns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
