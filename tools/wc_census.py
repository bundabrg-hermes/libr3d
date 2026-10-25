#!/usr/bin/env python3
# Q18b card-free gate: per-arm allocator + scratch census (old vs new) AND instruction
# -stream identity for the arms that must not change.
# Label lines in a .lto.s carry a trailing '; @mangled' comment -- the empty-slice trap
# the ledger records (COVERAGE window 11): anchor on the name + colon, allow the comment.
import re, sys
from collections import Counter

def key(name):
    ints = [int(("-" if n else "") + d) for n, d in re.findall(r"Li(n?)(\d+)E", name)]
    base = re.match(r"(_ZN\S+?kernel)", name)
    while len(ints) < 4: ints.append(0)
    return (base.group(1) if base else name, tuple(ints))

def bodies(path):
    lines = open(path).read().split("\n")
    lab = re.compile(r"^(_Z[^:\s]+):\s*(?:;.*)?$")
    idx = [(i, lab.match(l)) for i, l in enumerate(lines)]
    labs = [(i, m.group(1)) for i, m in idx if m]
    out = {}
    for j, (s, name) in enumerate(labs):
        e = labs[j+1][0] if j+1 < len(labs) else len(lines)
        body = [l for l in lines[s+1:e] if re.match(r"\t[a-z_]", l)]  # instructions only: every directive line starts with a tab then "."
        # Basic-block / func-end labels renumber module-wide when a TU gains functions
        # (.LBB18_7 -> .LBB24_7, .Lfunc_end18 -> .Lfunc_end24) with every operand
        # identical: normalize the counter so the identity gate measures codegen, not
        # the label allocator.
        norm = [re.sub(r"\.L([A-Za-z_]+)\d+", r".L\1", l) for l in body]
        # per-kernel metadata block (amdhsa directives live right after the body)
        meta_blk = "\n".join(lines[s:e])
        def g(d):
            m = re.search(r"\.amdhsa_" + d + r"\s+(\d+)", meta_blk)
            return int(m.group(1)) if m else None
        out.setdefault(key(name), []).append(
            dict(insn=norm, priv=g("private_segment_fixed_size"),
                 scr=sum(1 for l in body if "scratch" in l),
                 n=len(body)))
    return out

def meta(path):
    txt = open(path).read(); blk = txt[txt.find(".amdgpu_metadata"):]
    out, cur = {}, None
    for line in blk.splitlines():
        m = re.match(r"\s*\.name:\s+(\S+)", line)
        if m: cur = m.group(1); out.setdefault(cur, {}); continue
        if cur is None: continue
        m = re.match(r"\s*\.(vgpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count):\s*(\S+)", line)
        if m:
            try: out[cur][m.group(1)] = int(m.group(2))
            except ValueError: pass
    return out

def merge(x):
    """one dict per normalized key; duplicate labels (multiple code objects) take the max"""
    o = {}
    for k, lst in x.items():
        acc = o.setdefault(k, dict(priv=0, scr=0, n=0))
        for d in lst:
            acc["priv"] = max(acc["priv"], d["priv"] or 0)
            acc["scr"]  = max(acc["scr"], d["scr"])
            acc["n"]    = max(acc["n"], d["n"])
    return acc_meta(x, o)

def acc_meta(x, o):
    return o

a_p, b_p = sys.argv[1], sys.argv[2]
A, B = bodies(a_p), bodies(b_p)
MA, MB = meta(a_p), meta(b_p)
def mval(M, k, f):
    for n, d in M.items():
        if key(n) == k and f in d: return d[f]
    return None

keys = sorted({k for k in A if "gemv_q8_kernel" in k[0]} | {k for k in B if "gemv_q8_kernel" in k[0]})
assert keys, "no q8 kernels parsed -- parse collapse, trust nothing"
print("%-20s %-24s %-24s %-9s %-7s %s" % ("arm MT,W,C4,WC", "vgpr old->new", "spill(v/s) old->new", "privB o->n", "scratch", "insns"))
for k in keys:
    mt, w, c4, wc = k[1]
    va, vb = mval(MA, k, "vgpr_count"), mval(MB, k, "vgpr_count")
    sa = (mval(MA, k, "vgpr_spill_count"), mval(MA, k, "sgpr_spill_count"))
    sb = (mval(MB, k, "vgpr_spill_count"), mval(MB, k, "sgpr_spill_count"))
    ba = A.get(k, [{}])[0] if k in A else {}
    bb = B.get(k, [{}])[0] if k in B else {}
    pa, pb = ba.get("priv"), bb.get("priv")
    print("MT=%2d W=%2d C4=%d WC=%2d %-12s %-24s %-24s %-9s %-7s %s%s" % (
        mt, w, c4, wc, "",
        "%s -> %s" % (va if k in A else "-", vb),
        "%s/%s -> %s/%s" % (sa[0], sa[1], sb[0], sb[1]) if k in A else "- -> %s/%s" % sb,
        "%s -> %s" % (pa if k in A else "-", pb),
        "%s -> %s" % (ba.get("scr", "-"), bb.get("scr", "-")),
        "%s -> %s" % (ba.get("n", "-"), bb.get("n", "-")),
        "   [NEW ARM]" if k not in A else ""))

print()
common = [k for k in sorted(set(A) & set(B)) if k in keys]
ident = [k for k in common if [d["insn"] for d in A[k]] == [d["insn"] for d in B[k]]]
diff  = [k for k in common if k not in ident]
onlyb = sorted(set(B) - set(A))
print("ASM IDENTITY (q8 family): common=%d identical=%d DIFFERING=%d new-only=%d"
      % (len(common), len(ident), len(diff), len(onlyb)))
for k in diff:
    ca = Counter(l.strip().split()[0] for l in A[k][0]["insn"])
    cb = Counter(l.strip().split()[0] for l in B[k][0]["insn"])
    delta = {m: cb[m]-ca[m] for m in set(ca)|set(cb) if ca[m] != cb[m]}
    print("  DIFF MT=%d W=%d C4=%d WC=%d insns %d->%d mnem-delta %s" %
          (k[1][0], k[1][1], k[1][2], k[1][3], len(A[k][0]["insn"]), len(B[k][0]["insn"]), delta))
print("  new-only:", ["MT=%d W=%d C4=%d WC=%d" % k[1] for k in onlyb])
