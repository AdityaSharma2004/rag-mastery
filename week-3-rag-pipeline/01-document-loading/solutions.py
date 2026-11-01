"""
Topic 01 - Document loading: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Everything here is pure Python and runs OFFLINE (sample files are generated in code).
Only Q5 can optionally call Gemini embeddings: it does so when GEMINI_API_KEY is set,
otherwise it uses a tiny local stand-in embedder so you can still watch the logic work.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from gemini_utils import *   # embed_texts, ...

# --- helpers ----------------------------------------------------------------
import hashlib
import io
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@dataclass
class Document:
    """The ONE shape every loader returns: text + metadata (where it came from)."""
    text: str
    metadata: dict = field(default_factory=dict)


def make_pdf(path, pages, header="ACME Corp - Confidential"):
    """pages = list of line-lists; a None page has no text layer (a 'scan')."""
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)   # invariant=1: no timestamps, so identical input = identical bytes
    for n, lines in enumerate(pages, 1):
        if lines is None:
            c.rect(60, 300, 400, 200, fill=1)
        else:
            c.setFont("Helvetica", 9); c.drawString(50, 800, header)
            c.setFont("Helvetica", 11)
            y = 740
            for line in lines:
                c.drawString(50, y, line); y -= 16
            c.setFont("Helvetica", 9); c.drawString(50, 40, f"Page {n} of {len(pages)}")
        c.showPage()
    c.save()


def load_pdf(path) -> list[Document]:
    path = pathlib.Path(path)
    docs = []
    for n, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        docs.append(Document(text, {"source": path.name, "page": n, "type": "pdf",
                                    "needs_ocr": len(text.strip()) < 20}))
    return docs


def table_to_text(rows: list[list[str]]) -> str:
    header, *body = rows
    return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(header, r)) for r in body)


def load_docx(path) -> list[Document]:
    path = pathlib.Path(path)
    d = DocxDocument(str(path))
    docs, heading, buf = [], "(start)", []

    def flush():
        if "".join(buf).strip():
            docs.append(Document("\n".join(buf), {"source": path.name, "heading": heading, "type": "docx"}))
        buf.clear()

    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, d)
            if p.style.name.startswith(("Heading", "Title")):
                flush(); heading = p.text.strip()
            elif p.text.strip():
                buf.append(p.text.strip())
        elif child.tag.endswith("}tbl"):
            buf.append(table_to_text([[c.text.strip() for c in row.cells] for row in Table(child, d).rows]))
    flush()
    return docs


NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
NOISE_CLASS = re.compile(r"cookie|banner|sidebar|advert|popup", re.I)


def load_html(html: bytes | str, url: str = "") -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=NOISE_CLASS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    title = soup.title.get_text(strip=True) if soup.title else url
    docs, heading, buf = [], title, []

    def flush():
        if buf:
            docs.append(Document("\n".join(buf), {"source": url or title, "heading": heading, "type": "html"}))
        buf.clear()

    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3"):
            flush(); heading = txt
        else:
            buf.append(txt)
    flush()
    return docs


def read_text_robust(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def load_csv(path) -> list[Document]:
    path = pathlib.Path(path)
    df = pd.read_csv(io.StringIO(read_text_robust(path.read_bytes())), dtype=str, keep_default_na=False)
    return [Document("; ".join(f"{c}: {v}" for c, v in row.items() if v != ""),
                     {"source": path.name, "row": int(i) + 1, "type": "csv"})
            for i, row in df.iterrows()]


def load_txt(path) -> list[Document]:
    path = pathlib.Path(path)
    return [Document(read_text_robust(path.read_bytes()), {"source": path.name, "type": "txt"})]


def collapse_whitespace(s: str) -> str:
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def dehyphenate(s: str) -> str:
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", s)


def clean(docs: list[Document], min_chars: int = 20) -> list[Document]:
    out = []
    for d in docs:
        t = collapse_whitespace(dehyphenate(unicodedata.normalize("NFKC", d.text)))
        if len(t) >= min_chars:
            out.append(Document(t, {**d.metadata, "cleaned": True}))
    return out


def fake_embed(texts, task_type="RETRIEVAL_DOCUMENT", dim=64, **_):
    """Offline stand-in for embed_texts: hashed bag-of-words, L2-normalised. NOT semantic - demo only."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


# region q1
# --- Q1 ---------------------------------------------------------------------
# A citation formatter that works for every loader's metadata.
def cite(doc: Document) -> str:
    """Turn metadata into a human-readable citation, whatever the source type was."""
    m = doc.metadata
    src = m.get("source", "unknown source")
    kind = m.get("type")
    if kind == "pdf" and "page" in m:
        return f"{src}, p. {m['page']}"
    if kind in ("docx", "html") and m.get("heading"):
        return f'{src}, section "{m["heading"]}"'
    if kind == "csv" and "row" in m:
        return f"{src}, row {m['row']}"
    return src                                   # graceful fallback - never crash on odd metadata


def q1():
    docs = [
        Document("...", {"source": "policy.pdf", "page": 7, "type": "pdf"}),
        Document("...", {"source": "handbook.docx", "heading": "Leave policy", "type": "docx"}),
        Document("...", {"source": "products.csv", "row": 3, "type": "csv"}),
        Document("...", {"source": "notes.txt", "type": "txt"}),
        Document("...", {}),
    ]
    for d in docs:
        print(cite(d))
    assert cite(docs[0]) == "policy.pdf, p. 7"
    assert cite(docs[1]) == 'handbook.docx, section "Leave policy"'
    assert cite(docs[2]) == "products.csv, row 3"
    assert cite(docs[3]) == "notes.txt" and cite(docs[4]) == "unknown source"


# endregion q1


# region q2
# --- Q2 ---------------------------------------------------------------------
# HTML tables -> one Document per row, as "Plan: Pro; Price: 999" with the caption as context.
def html_tables_to_docs(html: str, url: str = "page") -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    docs = []
    for t_idx, table in enumerate(soup.find_all("table"), start=1):
        cap = table.find("caption")
        if cap:
            title = cap.get_text(" ", strip=True)
        else:                                            # fall back to the nearest heading above the table
            h = table.find_previous(["h1", "h2", "h3"])
            title = h.get_text(" ", strip=True) if h else f"table {t_idx}"
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue                                     # header only / empty table: nothing to index
        header, *body = rows
        for r_idx, row in enumerate(body, start=1):
            row = (row + [""] * len(header))[: len(header)]      # pad or trim ragged rows
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
            docs.append(Document(f"{title} - " + "; ".join(pairs),
                                 {"source": url, "type": "html-table", "table": t_idx, "row": r_idx,
                                  "heading": title}))
    return docs


def q2():
    html = """<h2>Pricing</h2>
    <table><thead><tr><th>Plan</th><th>Price</th><th>Seats</th></tr></thead>
    <tbody><tr><td>Free</td><td>0</td><td>1</td></tr>
           <tr><td>Pro</td><td>999</td><td>5</td></tr>
           <tr><td>Team</td><td>2999</td></tr></tbody></table>
    <table><caption>Delivery times</caption>
      <tr><th>Zone</th><th>Days</th></tr><tr><td>Metro</td><td>2</td></tr></table>"""
    docs = html_tables_to_docs(html, "pricing.html")
    for d in docs:
        print(d.text, d.metadata["table"], d.metadata["row"])
    assert docs[1].text == "Pricing - Plan: Pro; Price: 999; Seats: 5"
    assert docs[2].text == "Pricing - Plan: Team; Price: 2999"          # ragged row survived
    assert docs[3].text == "Delivery times - Zone: Metro; Days: 2" and len(docs) == 4


# endregion q2


# region q3
# --- Q3 ---------------------------------------------------------------------
# A whole-folder loader that never dies on one bad file and tells you what happened.
EXT_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
    ".txt": load_txt,
    ".html": lambda p: load_html(pathlib.Path(p).read_bytes(), pathlib.Path(p).name),
}


def load_directory(folder) -> tuple[list[Document], dict]:
    folder = pathlib.Path(folder)
    report = {"loaded": [], "skipped_unsupported": [], "failed": [], "duplicates": [], "needs_ocr_pages": []}
    docs, seen_hashes = [], {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = str(path.relative_to(folder))
        loader = EXT_LOADERS.get(path.suffix.lower())
        if loader is None:
            report["skipped_unsupported"].append(rel)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:                        # byte-identical copy of an earlier file
            report["duplicates"].append((rel, seen_hashes[digest]))
            continue
        seen_hashes[digest] = rel
        try:
            file_docs = loader(path)
        except Exception as e:                           # one corrupt file must not kill a 10,000-file ingest
            report["failed"].append((rel, f"{type(e).__name__}: {e}"))
            continue
        for d in file_docs:
            d.metadata["source"] = rel                   # relative path = stable, unique identity
            if d.metadata.get("needs_ocr"):
                report["needs_ocr_pages"].append((rel, d.metadata["page"]))
        docs.extend(d for d in file_docs if not d.metadata.get("needs_ocr"))
        report["loaded"].append(rel)
    return docs, report


def make_messy_folder(folder: pathlib.Path) -> pathlib.Path:
    (folder / "sub").mkdir(parents=True, exist_ok=True)
    pages = [["Refunds are possible within 30 days.", "Contact support with your order id."], None]
    make_pdf(folder / "policy.pdf", pages)
    make_pdf(folder / "sub" / "policy_copy.pdf", pages)                       # exact duplicate
    (folder / "corrupt.pdf").write_bytes(b"this is not a pdf at all")
    (folder / "notes.txt").write_text("Support is open on weekdays from 9 to 6.", encoding="utf-8")
    (folder / "logo.png").write_bytes(b"\x89PNG fake")
    (folder / "sub" / "faq.html").write_text("<main><h1>FAQ</h1><p>We ship across India in 5 days.</p></main>")
    return folder


def q3():
    folder = make_messy_folder(pathlib.Path(tempfile.mkdtemp()))
    docs, report = load_directory(folder)
    print(json.dumps(report, indent=2))
    print([(d.metadata["source"], d.metadata.get("page")) for d in docs])
    assert report["loaded"] == ["notes.txt", "policy.pdf", "sub/faq.html"]
    assert report["skipped_unsupported"] == ["logo.png"]
    assert report["duplicates"] == [("sub/policy_copy.pdf", "policy.pdf")]
    assert report["failed"][0][0] == "corrupt.pdf"
    assert report["needs_ocr_pages"] == [("policy.pdf", 2)]
    assert len(docs) == 3


# endregion q3


# region q4
# --- Q4 ---------------------------------------------------------------------
# Debug: a "cleaner" that damages text. Find every bug, then fix it.
def bad_clean(text: str) -> str:
    text = text.replace("-\n", "")              # BUG 1: also kills real hyphens (COVID-19, state-of-the-art)
    text = re.sub(r"\s+", " ", text)            # BUG 2: destroys paragraph breaks (\n\n) too
    text = re.sub(r"Page \d+ of 3", "", text)   # BUG 3: hard-coded page count; also leaves a hole in the text
    return text


def build_vocab(texts: list[str]) -> set[str]:
    """Every plain (non-hyphen-broken) word that appears anywhere in the corpus, lower-cased."""
    return {w.lower() for t in texts for w in re.findall(r"[A-Za-z]+", re.sub(r"-\n", " ", t))}


def dehyphenate_vocab(text: str, vocab: set[str]) -> str:
    """Join 'cus-\\ntomer' only when the joined word is a word we have seen elsewhere ('customer').
    Otherwise keep the hyphen and just drop the newline ('COVID-\\n19' -> 'COVID-19')."""
    def fix(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        joined = left + right
        return joined if joined.lower() in vocab else f"{left}-{right}"
    return re.sub(r"([A-Za-z]+)-\n([A-Za-z0-9]+)", fix, text)


def good_clean(text: str, vocab: set[str]) -> str:
    text = re.sub(r"^\s*Page \d+ of \d+\s*$", "", text, flags=re.M)     # whole line only, any page count
    text = dehyphenate_vocab(text, vocab)
    return collapse_whitespace(text)                                     # keeps blank-line paragraph breaks


def q4():
    raw = ("Every cus-\ntomer must read this. COVID-\n19 rules apply.\n"
           "Page 1 of 7\n\n"
           "We use state-of-the-\nart scanners. Each customer gets a receipt.")
    print("BAD :", repr(bad_clean(raw)))
    vocab = build_vocab([raw])
    fixed = good_clean(raw, vocab)
    print("GOOD:", repr(fixed))
    assert "COVID19" in bad_clean(raw) and "theart" in bad_clean(raw) and "\n\n" not in bad_clean(raw)
    assert "customer must read" in fixed and "COVID-19" in fixed and "the-art" in fixed
    assert "\n\n" in fixed and "Page" not in fixed


# endregion q4


# region q5
# --- Q5 ---------------------------------------------------------------------
# Incremental ingestion: re-embed ONLY files that changed (embeddings cost quota and time).
class MiniStore:
    """Stand-in for a vector DB table: id -> (vector, text, metadata). Real DBs offer delete-by-filter too."""

    def __init__(self):
        self.rows: dict[str, tuple] = {}

    def delete_source(self, source: str) -> int:
        ids = [i for i, (_, _, m) in self.rows.items() if m["source"] == source]
        for i in ids:
            del self.rows[i]
        return len(ids)

    def upsert(self, ids, vecs, texts, metas):
        for i, v, t, m in zip(ids, vecs, texts, metas):
            self.rows[i] = (v, t, m)


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_id(d: Document) -> str:
    """Stable id = source + position. Re-ingesting the same file yields the same ids (idempotent)."""
    m = d.metadata
    locator = m.get("page") or m.get("row") or m.get("heading") or "0"
    return f"{m['source']}#{locator}"


def sync_folder(folder, manifest_path, store: MiniStore, embed_fn=embed_texts) -> dict:
    folder, manifest_path = pathlib.Path(folder), pathlib.Path(manifest_path)
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    current = {str(p.relative_to(folder)): p for p in sorted(folder.rglob("*"))
               if p.is_file() and p.suffix.lower() in EXT_LOADERS}
    stats = {"added": [], "updated": [], "unchanged": [], "deleted": [], "chunks_embedded": 0}
    new_manifest = {}

    for rel in sorted(set(old) - set(current)):          # file removed from disk -> remove its rows
        store.delete_source(rel)
        stats["deleted"].append(rel)

    for rel, path in current.items():
        digest = file_sha256(path)
        if rel in old and old[rel] == digest:
            stats["unchanged"].append(rel)
            new_manifest[rel] = digest
            continue
        docs = clean([d for d in EXT_LOADERS[path.suffix.lower()](path) if not d.metadata.get("needs_ocr")])
        for d in docs:
            d.metadata["source"] = rel
        store.delete_source(rel)                         # drop stale rows first (a page may have vanished)
        if docs:
            vecs = embed_fn([d.text for d in docs])
            store.upsert([doc_id(d) for d in docs], vecs, [d.text for d in docs], [d.metadata for d in docs])
            stats["chunks_embedded"] += len(docs)
        stats["updated" if rel in old else "added"].append(rel)
        new_manifest[rel] = digest

    manifest_path.write_text(json.dumps(new_manifest, indent=2))
    return stats


def q5():
    embed_fn = embed_texts if os.getenv("GEMINI_API_KEY") else fake_embed
    print("embedder:", embed_fn.__name__)
    work = pathlib.Path(tempfile.mkdtemp())
    docs_dir, manifest = work / "docs", work / "manifest.json"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Alpha policy: refunds within 30 days of delivery.")
    (docs_dir / "b.txt").write_text("Bravo policy: shipping is free above 999 rupees.")
    make_pdf(docs_dir / "c.pdf", [["Charlie handbook page one text goes here."], ["Page two has more handbook text."]])

    store = MiniStore()
    s1 = sync_folder(docs_dir, manifest, store, embed_fn)
    print("run 1:", s1)
    s2 = sync_folder(docs_dir, manifest, store, embed_fn)
    print("run 2 (nothing changed):", s2)
    (docs_dir / "a.txt").write_text("Alpha policy: refunds within 14 days of delivery.")     # edit
    (docs_dir / "b.txt").unlink()                                                             # delete
    s3 = sync_folder(docs_dir, manifest, store, embed_fn)
    print("run 3 (edit a, delete b):", s3)
    print("ids in store:", sorted(store.rows))

    assert s1["added"] == ["a.txt", "b.txt", "c.pdf"] and s1["chunks_embedded"] == 4
    assert s2["chunks_embedded"] == 0 and len(s2["unchanged"]) == 3
    assert s3["updated"] == ["a.txt"] and s3["deleted"] == ["b.txt"] and s3["chunks_embedded"] == 1
    assert sorted(store.rows) == ["a.txt#0", "c.pdf#1", "c.pdf#2"]
    assert "14 days" in store.rows["a.txt#0"][1]

# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__.upper()} =====")
        fn()
    print("\nAll solutions passed.")
