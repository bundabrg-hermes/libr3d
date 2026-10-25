#!/usr/bin/env python3
"""Spill census: per-kernel register-pressure census for the gfx1100 code
objects in a libr3d build. Run inside radiance-dev (needs llvm-objcopy + llvm-readelf on
PATH, e.g. /opt/rocm/llvm/bin):

    docker run --rm -v $PWD:/work -w /work radiance-dev \
      bash -c 'export PATH=/opt/rocm/llvm/bin:$PATH && python3 tools/spill_census.py [build]'

v2 (2026-10-25, LIBR35D_TRIAGE verification): the v1 tool reported only
vgpr_spill_count and private_segment_fixed_size, which misses three live signal
classes found on our own tree: (a) SGPR spills (gdn carries 49-95 per kernel),
(b) kernels parked just under the spill cliff at the 256-VGPR ceiling (zero
spill today, one edit away from spilling -- the attn h256 MT=3 non-DBUF arms
sit at 249/242), and (c) LDS footprint, which couples to the same occupancy
budget. v1 also called a private segment without VGPR spill "stack/LDS-path
traffic, verify before chasing": the w2a8 q8 GEMV answer was 16x
scratch_store_b128 in the staging preamble -- real traffic, not metadata noise.
v2 prints the ceiling list and flags private-without-spill rows for the
objdump follow-up (grep scratch_store in llvm-objdump -d of the carved ELF).

Prints kernels with spills or a private segment worst first, then the
near-ceiling and big-LDS lists. A spill cliff is invisible to a timing table
that only times the shapes its fixture holds; census the whole tree.
"""
import glob
import os
import re
import struct
import subprocess
import sys
import tempfile

def kernels_of(o):
    """Per-kernel dicts from every gfx ELF embedded in one host object."""
    fb = tempfile.mktemp(suffix=".fb")
    nul = tempfile.mktemp(suffix=".nul.o")
    open(nul, "wb").write(b"\x7fELF\x02\x01\x01\x00" + b"\0" * 56)
    r = subprocess.run(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + fb, o, nul],
                       capture_output=True, text=True)
    os.unlink(nul)
    if r.returncode or not os.path.exists(fb):
        return []
    data = open(fb, "rb").read()
    out = []
    i = 0
    while True:
        i = data.find(b"\x7fELF", i + 1)
        if i < 0 or len(data) < i + 64:
            break
        if data[i + 4] != 2 or struct.unpack_from("<H", data, i + 18)[0] != 224:
            continue
        p = tempfile.mktemp(suffix=".elf")
        open(p, "wb").write(data[i:])
        txt = subprocess.run(["llvm-readelf", "--notes", p],
                             capture_output=True, text=True).stdout
        os.unlink(p)
        if "amdhsa.kernels" not in txt:
            i += 8
            continue
        t = re.search(r"amdhsa\.target:\s+(\S+)", txt)
        arch = t.group(1).rsplit("-", 1)[-1] if t else "?"
        body = txt.split("amdhsa.kernels:")[1]
        for block in re.split(r"\n(?=\s+- \.args:)", body)[1:]:
            kn = re.search(r"\.name:\s+(\S+)", block)
            if not kn:
                continue
            def num(f, block=block):
                m = re.search(r"\.%s:\s+(\d+)" % f, block)
                return int(m.group(1)) if m else 0
            out.append(dict(name=kn.group(1), arch=arch, unit=os.path.basename(o),
                            vgpr=num("vgpr_count"), vsp=num("vgpr_spill_count"),
                            ssp=num("sgpr_spill_count"),
                            priv=num("private_segment_fixed_size"),
                            lds=num("group_segment_fixed_size")))
        i += 8
    os.unlink(fb)
    return out

def main():
    build = sys.argv[1] if len(sys.argv) > 1 else "build"
    src = os.path.join(build, "CMakeFiles", "libr3d.dir", "src")
    if not os.path.isdir(src):
        sys.exit(f"no objects under {src} (build the library first)")
    want = os.environ.get("CENSUS_ARCH", "gfx1100")
    rows = {}
    for o in sorted(glob.glob(os.path.join(src, "*.o"))):
        for k in kernels_of(o):
            if k["arch"] == want:
                rows[(k["unit"], k["name"])] = k
    g = list(rows.values())
    bad = [r for r in g if r["vsp"] > 0 or r["ssp"] > 0 or r["priv"] > 0]
    print(f"{want} kernels: {len(g)}  spilling/private: {len(bad)}")
    for r in sorted(bad, key=lambda r: -(r["vsp"] + r["ssp"] + r["priv"])):
        tag = "   <- private w/o spill: objdump for scratch_store" if (
            r["priv"] > 0 and r["vsp"] == 0 and r["ssp"] == 0) else ""
        print(f"{r['vsp']:>6} vsp {r['ssp']:>4} ssp {r['priv']:>6} priv "
              f"{r['vgpr']:>4} vgpr {r['lds']:>6} lds  {r['unit']:<30} {r['name']}{tag}")
    print()
    print("near-ceiling (no spill, VGPR >= 200): one edit away from the cliff")
    for r in sorted((r for r in g if r["vsp"] == 0 and r["vgpr"] >= 200),
                    key=lambda r: -r["vgpr"]):
        print(f"{r['vgpr']:>4} vgpr {r['lds']:>6} lds  {r['unit']:<30} {r['name']}")
    print()
    print("biggest LDS (occupancy budget, couples to the above)")
    for r in sorted(g, key=lambda r: -r["lds"])[:6]:
        print(f"{r['lds']:>6} lds  {r['unit']:<30} {r['name']}")

if __name__ == "__main__":
    main()
