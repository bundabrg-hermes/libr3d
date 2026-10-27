#!/usr/bin/env python3
# q48_census.py -- per-symbol instruction-class census across EVERY gfx1100 code
# object in a fatbin-bearing file. Modeled on carve_dis.py + q11_slice.py, with the
# coverage-guard laws baked in:
#   - the two-token wait mnemonic (s_waitcnt lgkmcnt(N)) is parsed as a class,
#   - LDS ops counted as ds_load/ds_store/ds_read*/ds_write* (RDNA3 mnemonics),
#   - every class column proven by an assertion helper: a kernel whose body parses
#     < 200 instructions is flagged, never silently zeroed,
#   - a loop kernel (has a backward branch) showing zero backward branches is a
#     parse bug: backward branch = s_cbranch with a target < current address
#     (offsets print unsigned 16-bit -- parse the TARGET address, not the offset).
# usage: q48_census.py <file.so> <symbol-substring-filter> [--full <outdir.asm>]
import os, re, struct, subprocess, sys, tempfile

path, needle = sys.argv[1], sys.argv[2]
# Window-52 drift repair: the pinned /opt/rocm/core-10.1 LLVM path is gone.
# The LLVM bin dir is now overridable via R3D_LLVM_BIN=<dir> or --llvm <dir>;
# the DEFAULT is the system PATH (current /usr/bin llvm-readelf/objdump/objcopy).
LLVM = os.environ.get("R3D_LLVM_BIN", "").strip()
if "--llvm" in sys.argv: LLVM = sys.argv[sys.argv.index("--llvm") + 1]
ENV = dict(os.environ, PATH=(LLVM + ":" if LLVM else "") + os.environ.get("PATH", ""))

def rd(tag):
    p = tempfile.mktemp(suffix=".elf")
    subprocess.run(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + p, path, "/dev/null"],
                   check=True, env=ENV)
    data = open(p, "rb").read(); os.unlink(p)
    objs = []
    i = 0
    while True:
        i = data.find(b"\x7fELF", i + 1)
        if i < 0 or len(data) < i + 64: break
        if data[i+4] != 2 or struct.unpack_from("<H", data, i+18)[0] != 224: continue
        q = tempfile.mktemp(suffix=".elf")
        open(q, "wb").write(data[i:])
        txt = subprocess.run(["llvm-readelf", "--notes", q], capture_output=True,
                             text=True, env=ENV).stdout
        # Window-52 drift repair: current llvm-readelf prints "amdhsa.target:";
        # ROCm 10.1-era printed ".target:". Match BOTH forms explicitly.
        tg = re.search(r"(?:amdhsa\.target|\.target):\s*(\S+)", txt)
        if tg and "gfx1100" in tg.group(1):
            objs.append(q)
        else:
            os.unlink(q)
        i += 8
    return objs

CLASSES = {
    "ds_bpermute":   re.compile(r"\bds_bpermute_b32\b"),
    "ds_swizzle":    re.compile(r"\bds_swizzle_b32\b"),
    "permlanex16":   re.compile(r"\bv_permlanex16_b32\b"),
    "dpp_mov":       re.compile(r"\bv_mov_b32_dpp\b"),
    "dpp_other":     re.compile(r"\bv_(?!mov_b32_dpp)\w+_dpp\b"),
    "shfl_source":   re.compile(r"\bv_readlane_b32\b|\bwritems_b32\b"),
    "ds_read":       re.compile(r"\bds_read\b|\bds_load_b(32|64|128)\b|\bds_read_(b|u|s|i)(8|16|32|64|128)\b|\bds_read2\b|\bds_load_2addr\b"),
    "ds_write":      re.compile(r"\bds_write\b|\bds_store_b(32|64|128)\b|\bds_write_(b|u|s|i)(8|16|32|64|128)\b|\bds_store_2addr\b|\bds_write2\b|\bds_write_b16\b"),
    "wmma":          re.compile(r"\bv_wmma_"),
    "wait_lgkm":     re.compile(r"\bs_waitcnt\s+lgkmcnt"),
    "branch":        re.compile(r"\bs_cbranch"),
}

objs = rd(needle)
# Window-52 guard law: zero selected objects is a REGEX/NOTES DRIFT signal,
# never a silent pass (previously the needle-in-notes check matched zero and
# the tool printed an empty table).
if not objs:
    sys.exit("GUARD: 0 gfx1100 objects selected -- target-note form or fatbin "
             "layout drift; check the llvm-readelf --notes output form.")
rows = []
lab = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")  # window-52: current llvm prints UPPER-CASE hex
full_out = None
if "--full" in sys.argv:
    full_out = open(sys.argv[sys.argv.index("--full") + 1], "w")

for obj in objs:
    r = subprocess.run(["llvm-objdump", "-d", obj], capture_output=True, text=True, env=ENV)
    if full_out: full_out.write(r.stdout)
    cur = None; acc = None
    for line in r.stdout.splitlines():
        m = lab.match(line)
        if m:
            if cur: rows.append(acc)
            cur = m.group(2)
            acc = {"name": cur, "n": 0, "lastaddr": 0, "backbranch": 0, "base": int(m.group(1), 16)}
            acc.update({k: 0 for k in CLASSES})
            continue
        if acc is None or line[:1] != "\t": continue
        # body line: <TAB><mnemonic> <operands>  // <addr>: <enc> [<sym>+0xoff>]
        # (the window-18 law: llvm-objdump prints addresses in the TRAILING comment)
        am = re.match(r"\t([a-z]\w*)\s*(.*)", line)
        if not am: continue
        acc["n"] += 1
        body = am.group(1) + " " + am.group(2)
        for k, rx in CLASSES.items():
            if rx.search(body): acc[k] += 1
        cm = re.search(r"//\s*([0-9a-fA-F]+):", line)
        addr = int(cm.group(1), 16) if cm else acc["lastaddr"] + 4
        acc["lastaddr"] = addr
        if body.startswith("s_cbranch"):
            tm = re.search(r"<([^>+]+)\+0x([0-9a-fA-F]+)>", line)
            if tm:
                if tm.group(1) == acc["name"]:
                    # window-52: comment offsets are SYMBOL-RELATIVE
                    if int(tm.group(2), 16) <= addr - acc["base"]: acc["backbranch"] += 1
            else:
                tm = re.search(r"//\s*[0-9a-f]+:\s*[0-9a-f]+\s+(0x[0-9a-f]+)", line)
                if tm and int(tm.group(1), 16) <= addr: acc["backbranch"] += 1
    if cur: rows.append(acc)
if full_out: full_out.close()
for obj in objs: os.unlink(obj)

print(f"{'kernel':80.80s} {'instrs':>7s} {'bperm':>6s} {'swz':>6s} {'plx16':>6s} {'dppmv':>6s} {'dppo':>5s} {'ldsR':>6s} {'ldsW':>6s} {'wmma':>5s} {'br':>4s} {'brck':>5s}")
for acc in rows:
    if needle not in acc["name"]: continue
    flag = "  <SMALL-BODY>" if acc["n"] < 200 and "merge" not in acc["name"] else ""
    bug = "  <PARSE-BUG?>\n" if (acc["branch"] > 0 and acc["backbranch"] == 0) else ""
    print(f"{acc['name']:80.80s} {acc['n']:7d} {acc['ds_bpermute']:6d} {acc['ds_swizzle']:6d} "
          f"{acc['permlanex16']:6d} {acc['dpp_mov']:6d} {acc['dpp_other']:5d} "
          f"{acc['ds_read']:6d} {acc['ds_write']:6d} {acc['wmma']:5d} {acc['branch']:4d} {acc['backbranch']:5d}{flag}{bug}")
filtered = sum(1 for a in rows if needle in a["name"])
print("objects=%d total_symbols=%d filtered=%d" % (len(objs), len(rows), filtered))
if filtered == 0:
    sys.exit("GUARD: filter %s matched 0 symbols -- symbol-form drift." % needle)
