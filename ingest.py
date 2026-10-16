import pymupdf, json, re, pathlib

HEAD = re.compile(r"^(\d+)\.\s+(.+)$", re.M)

def page_text(page):
    """Page text in reading order, with tables rebuilt as markdown rows."""
    tables = page.find_tables().tables
    boxes = [pymupdf.Rect(t.bbox) for t in tables]
    parts = []
    for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
        if any(pymupdf.Rect(x0, y0, x1, y1).intersects(b) for b in boxes):
            continue  # skip raw table cells; replaced by markdown below
        parts.append((y0, txt.strip()))
    for t, b in zip(tables, boxes):
        parts.append((b.y0, t.to_markdown().strip()))
    return "\n".join(t for _, t in sorted(parts, key=lambda p: p[0]))

chunks = []
for pdf in sorted(pathlib.Path("data").glob("*.pdf")):
    doc = pymupdf.open(pdf)
    title = " — ".join(doc[0].get_text().splitlines()[:2])

    for pno, page in enumerate(doc, 1):
        text = page_text(page)
        heads = list(HEAD.finditer(text))
        for j, h in enumerate(heads):
            end = heads[j + 1].start() if j + 1 < len(heads) else len(text)
            chunks.append({
                "id": f"{pdf.stem}-p{pno}-s{h.group(1)}",
                "doc": pdf.name,
                "page": pno,
                "section": f"§{h.group(1)} {h.group(2).strip()}",
                "text": f"{title}\n{text[h.start():end].strip()}",
            })

json.dump(chunks, open("chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for c in chunks:
    print("=" * 10, c["id"], "|", c["section"]); print(c["text"]); print()
print(len(chunks), "chunks saved to chunks.json")
