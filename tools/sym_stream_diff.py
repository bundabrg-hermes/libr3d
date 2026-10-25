#!/usr/bin/env python3
# Q23 gate: per-symbol instruction-stream identity between two host objects.
# usage: sym_stream_diff.py <objA.o> <objB.o>
# Compares every gfx1100 kernel symbol present in both; reports additions,
# removals, identical streams, and differing streams (with mnemonic deltas).
import os, re, struct, subprocess, sys, tempfile
from collections import Counter

def carve_objs(path):
    fb = tempfile.mktemp(suffix=".fb")
    r = subprocess.run(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + fb, path, "/dev/null"],
                       capture_output=True, text=True)
    if not os.path.exists(fb) or os.path.getsize(fb) == 0:
        sys.exit(f"no .hip_fatbin in {path}")
    data = open(fb, "rb").read(); os.unlink(fb)
    out = []
    i = 0
    while True:
        i = data.find(b"\x7fELF", i + 1)
        if i < 0 or len(data) < i + 64: break
        if data[i+4] != 2 or struct.unpack_from("<H", data, i+18)[0] != 224:
            continue
        p = tempfile.mktemp(suffix=".elf")
        open(p, "wb").write(data[i:])
        txt = subprocess.run(["llvm-readelf", "--notes", p], capture_output=True, text=True).stdout
        if "gfx1100" in txt: out.append(p)
        else: os.unlink(p)
        i += 8
    return out

label_re = re.compile(r"^([0-9a-fA-F]{4,16}) <(.+)>:")
ins_re   = re.compile(r"^\s+(\S+)(?:\s+(.*?))?\s*//\s*([0-9a-fA-F]{4,16}):")

def streams(objs):
    syms = {}
    n_instrs = 0
    for o in objs:
        r = subprocess.run(["llvm-objdump", "-d", o], capture_output=True, text=True)
        cur = None
        for line in r.stdout.splitlines():
            m = label_re.match(line)
            if m: cur = m.group(2); syms.setdefault(cur, []); continue
            if cur is None: continue
            m = ins_re.search(line)
            if m: syms[cur].append((m.group(1), m.group(2) or "")); n_instrs += 1
    return syms, n_instrs

def norm_key(name):
    """Mangling gains a trailing Li0E per defaulted template arg: v136's 7-arg
    kernel1 names must compare equal to v137's 8-arg names. Key = name text
    minus template ints + int tuple zero-padded to 8 for kernel1."""
    head = re.match(r"(_ZN12_GLOBAL__N_1\d+attn_paged_wmma_kernel\d*)", name)
    ints = [int(x) for x in re.findall(r"Li(\d+)E", name)]
    if head and not head.group(1).endswith(("2", "3")):
        while len(ints) < 8: ints.append(0)
    return (head.group(1) if head else name, tuple(ints))

a, b = sys.argv[1], sys.argv[2]
sa_raw, na = streams(carve_objs(a))
sb_raw, nb = streams(carve_objs(b))
sa = {norm_key(k): v for k, v in sa_raw.items()}
sb = {norm_key(k): v for k, v in sb_raw.items()}
assert na > 1000 and nb > 1000, f"parse collapse: {na} vs {nb} instrs"
common = set(sa) & set(sb)
onlya, onlyb = set(sa) - set(sb), set(sb) - set(sa)
ident = [k for k in common if sa[k] == sb[k]]
diff  = [k for k in common if sa[k] != sb[k]]
print(f"COVERAGE: A={na} instrs/{len(sa)} syms, B={nb}/{len(sb)}, common={len(common)}")
print(f"only-in-A: {sorted(onlya)}" if onlya else "only-in-A: none")
print(f"only-in-B: {sorted(onlyb)}" if onlyb else "only-in-B: none")
print(f"identical (full stream incl. operands): {len(ident)}  differing: {len(diff)}")
for k in sorted(diff):
    ca = Counter(m for m, _ in sa[k]); cb = Counter(m for m, _ in sb[k])
    delta = {m: ca[m]-cb[m] for m in set(ca)|set(cb) if ca[m]!=cb[m]}
    # operand-level renames: same mnem sequence, differing operand text?
    seqsame = [m for m,_ in sa[k]] == [m for m,_ in sb[k]]
    ndiff = sum(1 for x, y in zip(sa[k], sb[k]) if x != y)
    print(f"  DIFF {k[:80]} lens {len(sa[k])}/{len(sb[k])} seq-same-mnems={seqsame} "
          f"operand-diff-lines={ndiff} mnem-delta {delta}")
