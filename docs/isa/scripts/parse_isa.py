#!/usr/bin/env python3
"""Parse the AMD RDNA3 ISA PDF into a searchable markdown corpus.

Output layout (relative to docs/isa/):
  inst/<mnemonic>.md      one file per instruction (ch 16)
  sections/<slug>.md      narrative sections (chapters 1-15 + family intros)
  INDEX.md                full map of the corpus
  parse_report.json       validation results

Deterministic rules (verified against the 15-Aug-2023 revision):
  * body text = SourceSerifPro 10.0; pseudocode = RobotoMono 7.5;
    headers/footers = 9.0pt -> dropped
  * instruction entry  = bold identifier line + opcode-number line + non-bold body line
  * opcode-table row   = same pair but followed by another bold line (no body)
  * entries are cross-checked against the opcode tables of their family section
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent.parent          # docs/isa
PDF = HERE / "source" / "rdna3-isa.pdf"
INST_DIR = HERE / "inst"
SECT_DIR = HERE / "sections"

IDENT = re.compile(
    r"(?:[A-Z][A-Z0-9_.]{0,60}[0-9A-Z])(?:[ \t]*/[ \t]*(?:[A-Z][A-Z0-9_.]{0,60}[0-9A-Z]))*"
)
OPNUM = re.compile(r"\d{1,4}|0[xX][0-9A-Fa-f]{1,6}")
PREFIX = re.compile(
    r"^(S_|V_|SMEM_|TGS|DS_|LDS_|BUFFER_|TBUFFER_|IMAGE_|TIMG_|TFB_|TFETCH|TCOMMIT|TFENCE|EXP|FLAT|SCRATCH|GLOBAL|GDS)"
)
MONO_F = re.compile(r"mono", re.I)
FOOTER = re.compile(r"^\d{1,3} of \d{1,3}$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class Line:
    __slots__ = ("pno", "text", "bold", "mono", "size")

    def __init__(self, pno, text, bold, mono, size):
        self.pno, self.text, self.bold, self.mono, self.size = pno, text, bold, mono, size


def load_lines(doc):
    lines = []
    for pno in range(doc.page_count):
        for block in doc[pno].get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for ln in block["lines"]:
                text = "".join(s["text"] for s in ln["spans"]).rstrip()
                if not text.strip():
                    continue
                size = max(s["size"] for s in ln["spans"])
                font = ln["spans"][0]["font"]
                if size <= 9.2 or FOOTER.match(text.strip()):
                    continue  # running header / footer / page number
                lines.append(
                    Line(pno + 1, text.strip(), "bold" in font.lower(),
                         bool(MONO_F.search(font)), round(size, 1))
                )
    return lines


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def find_heading(lines, page_starts, title, page, window=2):
    """Global index of the heading line `title` near `page`."""
    want = norm(title)
    p0 = max(1, page - window)
    p1 = page + window
    for i, ln in enumerate(lines):
        if p0 <= ln.pno <= p1 and norm(ln.text) == want:
            return i
    # last resort: same page, prefix match (heading may wrap in TOC copies)
    for i, ln in enumerate(lines):
        if ln.pno == page and norm(ln.text).startswith(want[:40]):
            return i
    return None


def slice_text(sub):
    """Render a list of Line into markdown; mono runs -> fences, bold -> ###."""
    out, para, code = [], [], []

    def flush_para():
        if para:
            out.append(" ".join(para))
            para.clear()

    def flush_code():
        if code:
            out.append("```")
            out.extend(code)
            out.append("```")
            code.clear()

    for ln in sub:
        if ln.mono:
            flush_para()
            code.append(ln.text)
        elif ln.bold and len(ln.text) <= 60 and not ln.text.endswith(":"):
            flush_para(); flush_code()
            out.append(f"### {ln.text}")
        else:
            flush_code()
            para.append(ln.text)
    flush_para(); flush_code()
    return "\n\n".join(out)


def fmt_tag(title: str) -> str:
    m = re.search(r"(VOPD X|VOPD Y|VOP3P|VOP3SD|VOP3|VOP2|VOP1|VOPC|VINTERP|SOP2|SOPK|SOP1|SOPC|SOPP|SMEM|MUBUF|MTBUF|MIMG|EXPORT|Flat|Scratch|Global|LDS & GDS|Parameter)", title)
    t = m.group(1) if m else ""
    return {"VOPD X": "VOPDX", "VOPD Y": "VOPDY", "Flat": "FLAT", "Scratch": "SCRATCH",
            "Global": "GLOBAL", "LDS & GDS": "LDS", "Parameter": "PARAM"}.get(t, t.upper())


def main():
    doc = pymupdf.open(PDF)
    toc = [(lvl, norm(t), pg) for lvl, t, pg in doc.get_toc(simple=True)]
    lines = load_lines(doc)

    # page -> first line index
    page_starts = {}
    for i, ln in enumerate(lines):
        page_starts.setdefault(ln.pno, i)

    # ---- locate headings ----
    heads = []  # (gi, lvl, title, page)
    missing = []
    for lvl, title, pg in toc:
        if title.lower() in ("contents",) or title.startswith('"RDNA3" Instruction Set Architecture: Reference Guide'):
            continue
        gi = find_heading(lines, page_starts, title, pg)
        if gi is None:
            missing.append(title)
            continue
        heads.append((gi, lvl, title, pg))
    heads.sort(key=lambda h: h[0])

    # chapter per heading
    chap_idx = [i for i, h in enumerate(heads) if h[1] == 1]
    reports = {"missing_headings": missing, "sections": [], "corpus": []}

    INST_DIR.mkdir(exist_ok=True)
    SECT_DIR.mkdir(exist_ok=True)
    index_rows = []
    used_slugs = set()

    def uniq(slug):
        s, n = slug, 2
        while s in used_slugs:
            s = f"{slug}-{n}"
            n += 1
        used_slugs.add(s)
        return s

    for hi, (gi, lvl, title, pg) in enumerate(heads):
        end = heads[hi + 1][0] if hi + 1 < len(heads) else len(lines)
        sub = lines[gi + 1:end]
        p_start = sub[0].pno if sub else pg
        p_end = sub[-1].pno if sub else pg
        prev_gi = max((heads[c][0] for c in chap_idx if heads[c][0] <= gi), default=None)
        chapter = next((h[2] for h in heads if h[0] == prev_gi), "")
        is_ch16 = chapter.startswith("Chapter 16")
        is_container = is_ch16 and lvl <= 3 and re.search(r"\d+\.\d+", title) \
            and title.lower().rstrip().endswith("instructions")

        if is_container:
            # instruction container: split entries vs intro/tables
            intro_i, entries = split_entries(sub, title, reports)
            sect_slug = uniq(slugify(title))
            (SECT_DIR / f"{sect_slug}.md").write_text(
                f"# {title}\n\n_(family intro + opcode table, pages {p_start}-{p_end})_\n\n"
                + slice_text(intro_i) + "\n")
            index_rows.append(("section", title, f"sections/{sect_slug}.md"))
            for name, opcode, body in entries:
                names = [n.strip() for n in name.split("/")]
                primary = names[0]
                fmt = fmt_tag(title)
                stem = slugify(primary).replace("-", "_")
                fname = stem + ".md"
                if fname in used_slugs:
                    fname = f"{stem}__{fmt.lower()}.md"
                    n2 = 2
                    while fname in used_slugs:
                        fname = f"{stem}__{fmt.lower()}-{n2}.md"
                        n2 += 1
                used_slugs.add(fname)
                pages = sorted({l.pno for l in body})
                aliases = names[1:]
                fm = ["---", f"mnemonic: {primary}"]
                if aliases:
                    fm.append(f"aliases: [{', '.join(aliases)}]")
                fm += [f'section: "{title}"', f"format: {fmt}", f"opcode: \"{opcode}\"",
                       f"pages: [{', '.join(map(str, pages))}]",
                       f"source: RDNA3 ISA PDF p.{pages[0]}-{pages[-1]}", "---"]
                body_md = slice_text(body)
                (INST_DIR / fname).write_text(
                    "\n".join(fm) + f"\n\n# {primary} (opcode {opcode})\n\n{body_md}\n")
                index_rows.append(("inst", f"{primary} (opcode {opcode})", f"inst/{fname}"))
        else:
            # narrative section
            if lvl == 1:
                slug = uniq(f"{pg:03d}-{slugify(title)}")
                kind = "chapter"
            else:
                slug = uniq(slugify(title))
                kind = "section"
            (SECT_DIR / f"{slug}.md").write_text(
                f"# {title}\n\n_(pages {p_start}-{p_end})_\n\n"
                + slice_text(sub) + "\n")
            index_rows.append((kind, title, f"sections/{slug}.md"))
            if lvl == 2:
                reports["sections"].append(title)

    with open(HERE / "INDEX.md", "w") as f:
        f.write("# RDNA3 ISA corpus index\n\n")
        for kind, title, path in index_rows:
            f.write(f"- [{kind}] {title} -> {path}\n")

    with open(HERE / "parse_report.json", "w") as f:
        json.dump(reports, f, indent=2)

    n_inst = len([r for r in index_rows if r[0] == "inst"])
    print(f"instructions: {n_inst}  sections/chapters: {len(index_rows) - n_inst}")
    print(f"missing headings: {missing}")


def split_entries(sub, section_title, reports):
    """Return (intro_lines, [(name, opcode, body_lines)])."""
    starts = []  # (i, name, opcode)
    table_rows = []
    i = 0
    while i + 2 < len(sub):
        a, b, c = sub[i], sub[i + 1], sub[i + 2]
        if (a.bold and IDENT.fullmatch(a.text) and b.bold and OPNUM.fullmatch(b.text)):
            if not c.bold and (IDENT.fullmatch(a.text) and PREFIX.match(a.text)):
                starts.append((i, a.text, b.text))
                i += 3
                continue
            if c.bold:  # opcode-table row
                table_rows.append((a.text, b.text))
                i += 2
                continue
        i += 1

    entries = []
    for k, (i, name, opcode) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(sub)
        entries.append((name, opcode, sub[i + 2:end]))

    # intro = everything before first entry minus table rows region kept verbatim
    intro = sub[:starts[0][0]] if starts else sub
    tnames = {n.split("/")[0].strip() for n, _ in table_rows}
    enames = {n.split("/")[0].strip() for n, _, _ in entries}
    reports["sections"].append(
        {"section": section_title, "table_rows": len(table_rows), "entries": len(entries),
         "in_table_not_described": sorted(tnames - enames),
         "described_not_in_table": sorted(enames - tnames)})
    return intro, entries


if __name__ == "__main__":
    main()
