#!/usr/bin/env python3
"""q44_schedule_diff.py -- Q44 desk probe schedule-proxy analyzer (v2).
Loop = widest back edge (s_branch OR s_cbranch, symbol-relative comment targets);
reports per-loop cross-lane counts, XOP-outside-loops (tail placement),
XOP->consumer distance, dependent fold-run lengths (bounded window), fusion
class counts, resources from code-object metadata, saveexec audit.
usage: q44_schedule_diff.py <file.o>"""
import os, re, struct, subprocess, sys, tempfile

SO = sys.argv[1]
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout

p = tempfile.mktemp(suffix=".elf")
subprocess.run(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + p, SO, "/dev/null"], check=True)
data = open(p, "rb").read(); os.unlink(p)
objs, i = [], 0
while True:
    i = data.find(b"\x7fELF", i + 1)
    if i < 0 or len(data) < i + 64: break
    if data[i+4] != 2 or struct.unpack_from("<H", data, i+18)[0] != 224: continue
    q = tempfile.mktemp(suffix=".elf")
    open(q, "wb").write(data[i:])
    notes = sh(["llvm-readelf", "--notes", q])
    if re.search(r"(?:amdhsa\.target|\.target):\s*\S*gfx1100", notes): objs.append((q, notes))
    else: os.unlink(q)
    i += 8

XOP = re.compile(r"\b(ds_bpermute_b32|v_permlanex16_b32|v_mov_b32_dpp|v_readfirstlane_b32)\b")
CLASSES = {
    "bperm": r"\bds_bpermute_b32\b", "plx": r"\bv_permlanex16_b32\b",
    "dppmv": r"\bv_mov_b32_dpp\b", "rfl": r"\bv_readfirstlane_b32\b",
    "ds": r"\bds_(read|load|write|store)",
    "dot4": r"\bv_dot4_i32_i[uo]8\b",
    "add3": r"\bv_add3_u32\b", "cndmask": r"\bv_cndmask_b32\b",
    "vadd_f32": r"\bv_add_f32", "vmul_f32": r"\bv_mul_f32",
    "gload": r"\bglobal_load", "cvt": r"\bv_cvt_f32_i32\b",
}
CL = {k: re.compile(v) for k, v in CLASSES.items()}
LABEL = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")

for obj, notes in objs:
    asm = sh(["llvm-objdump", "-d", obj]).splitlines()
    res = {}
    for m in re.finditer(r"- \.args:.*?\.name:\s+(\S+)\n(.*?)(?=\n  - \.args:|\namdhsa\.target|\Z)", notes, re.S):
        nm, body = m.group(1), m.group(2)
        g = lambda k: (re.search(r"\.%s:\s+(\d+)" % k, body) or [None, "?"])[1]
        res[nm] = (g("vgpr_count"), g("sgpr_count"), g("vgpr_spill_count"), g("sgpr_spill_count"))
    syms, bases, cur = {}, {}, None
    for ln in asm:
        m = LABEL.match(ln)
        if m:
            cur = m.group(2); bases[cur] = int(m.group(1), 16); syms[cur] = []
        elif cur and ln.startswith("\t"):
            syms[cur].append(ln)

    for nm, body in sorted(syms.items()):
        if "_kernel" not in nm or ("formA" not in nm and "formB" not in nm):
            continue
        addrs, texts, srcs = [], [], []
        for ln in body:
            am = re.match(r"\t([a-z.]\w*)\s*(.*)", ln)
            if not am: continue
            cm = re.search(r"//\s*([0-9a-fA-F]+):", ln)
            addrs.append(int(cm.group(1), 16) if cm else (addrs[-1] + 4 if addrs else 0))
            texts.append(am.group(1) + " " + am.group(2)); srcs.append(ln)
        base = bases.get(nm, 0)
        back = []
        for idx, t in enumerate(texts):
            if t.startswith("s_cbranch") or t.startswith("s_branch"):
                tm = re.search(r"<([^>+]+)\+0x([0-9a-f]+)>", srcs[idx])
                if tm and tm.group(1) == nm:
                    tgt = base + int(tm.group(2), 16)
                    if tgt < addrs[idx] - 4:  # strictly backward
                        back.append((tgt, addrs[idx]))
        # keep maximal spans; drop any loop contained in another; also drop the
        # tail-reduce "loop" (s_cbranch_execz style predication) if contained.
        back.sort(key=lambda x: -(x[1] - x[0]))
        loops = []
        for l in back:
            if not any(l[0] >= w[0] and l[1] <= w[1] for w in loops): loops.append(l)
        # the it-loops are the maximal BACK edges; cap at 3
        loops = loops[:3]
        tot = {k: sum(1 for t in texts if rx.search(t)) for k, rx in CL.items()}
        inloop = [False] * len(texts)
        for lo, hi in loops:
            for j in range(len(texts)):
                if lo <= addrs[j] <= hi: inloop[j] = True
        def cnt(region):
            return {k: sum(1 for j, t in enumerate(texts) if region[j] and rx.search(t))
                    for k, rx in CL.items()}
        xl = [i for i, t in enumerate(texts) if XOP.search(t)]
        xl_in = sum(1 for i in xl if inloop[i])
        for li, (lo, hi) in enumerate(loops):
            reg = [lo <= a <= hi for a in addrs]
            c = cnt(reg)
            print(f"  L{li} [{hex(lo)},{hex(hi)}] n={sum(reg)} " +
                  " ".join(f"{k}={c[k]}" for k in ("bperm","plx","dppmv","ds","dot4","add3","cndmask","gload")) +
                  f" XOP={c['bperm']+c['plx']+c['dppmv']+c['rfl']}")
        # chain metrics on ALL XOP then loop-only
        def dst_reg(t):
            m = re.match(r"\w+\s+(v\d+)\b", t); return m.group(1) if m else None
        def reads(t, r): return re.search(r"\b%s\b" % r, t) is not None
        dists, runs = [], []
        used = set()
        for i0 in xl:
            r0 = dst_reg(texts[i0])
            if r0:
                for j in range(i0 + 1, min(i0 + 200, len(texts))):
                    if reads(texts[j], r0): dists.append(j - i0); break
            if i0 in used: continue
            chain, cur_i, xf = [i0], i0, 1
            while True:
                d = dst_reg(texts[cur_i])
                if not d: break
                c = next((j for j in range(cur_i + 1, min(cur_i + 40, len(texts))) if reads(texts[j], d)), None)
                if c is None: break
                cd = dst_reg(texts[c])
                nxt = next((j for j in xl if c < j <= c + 40 and cd and reads(texts[j], cd)), None)
                if nxt is None: break
                chain.append(nxt); cur_i = nxt; xf += 1
            used.update(chain); runs.append(xf)
        med = sorted(dists)[len(dists)//2] if dists else -1
        sev = sum(1 for t in texts if re.search(r"\bs_(and|or)_saveexec\b", t))
        rsrc = res.get(nm, ("?",) * 4)
        print(f"SYM {nm[:50]} n={len(texts)} vgpr={rsrc[0]} sgpr={rsrc[1]} vsp={rsrc[2]} ssp={rsrc[3]} "
              f"TOT-XOP={len(xl)} XOP-in-loops={xl_in} XOP-tail={len(xl)-xl_in}")
        print(f"  TOT " + " ".join(f"{k}={tot[k]}" for k in CLASSES) +
              f" | medXOP->cons={med} maxfoldrun={max(runs, default=0)} runs={len(runs)} saveexec={sev}")
for q, _ in objs: os.unlink(q)
