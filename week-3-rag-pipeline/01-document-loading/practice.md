# Practice - Document loading

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Everything runs offline with no API key:** sample PDFs/DOCX/HTML are generated in code, and Q5 uses a local stand-in embedder unless `GEMINI_API_KEY` is set. The `Document` dataclass and the loaders (`load_pdf`, `load_docx`, `load_html`, `load_csv`, `load_txt`, `clean`, `make_pdf`, `fake_embed`) are defined at the top of `solutions.py` and explained in the notes.

---

### Q1. Warm-up - a citation formatter for every source type

**Problem**

Write `cite(doc)` that turns a `Document`'s metadata into a human-readable citation: PDF → `policy.pdf, p. 7`; DOCX/HTML with a heading → `handbook.docx, section "Leave policy"`; CSV → `products.csv, row 3`; plain text → just the file name; empty metadata → `unknown source` (never crash).

<details><summary><b>Hint</b></summary>

Read `type` and the locator key (`page`, `heading`, `row`) from `metadata` with `.get(...)`. End with a graceful fallback.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Citations are only possible because every loader recorded *where* each piece of text came from. Make citation formatting tolerant of missing keys: one odd file must not crash answer generation.

**Expected output**
```text
policy.pdf, p. 7
handbook.docx, section "Leave policy"
products.csv, row 3
notes.txt
unknown source
```

---

### Q2. Apply - HTML tables into searchable sentences

**Problem**

A web page has two tables (one without a caption but under an `<h2>`, one with a `<caption>`), including a ragged row that has fewer cells than the header. Convert **each table row** into one `Document` such as `"Pricing - Plan: Pro; Price: 999; Seats: 5"`, using the caption or nearest heading as context, skipping header-only tables.

<details><summary><b>Hint</b></summary>

`table.find("caption")`, else `table.find_previous(["h1","h2","h3"])`. Take the first row as the header, pad/trim each body row to the header length, and join only non-empty cells.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
A bare cell like `999` is meaningless to an embedding model; `Pricing - Plan: Pro; Price: 999` carries its own context, so a question like "how much is the Pro plan?" can match it. Real HTML is ragged and messy, so loaders must pad, trim and skip empty tables.

**Expected output**
```text
Pricing - Plan: Free; Price: 0; Seats: 1 1 1
Pricing - Plan: Pro; Price: 999; Seats: 5 1 2
Pricing - Plan: Team; Price: 2999 1 3
Delivery times - Zone: Metro; Days: 2 2 1
```
(The trailing numbers are the table and row indexes printed by the demo.)

---

### Q3. Build - a directory loader that never dies

**Problem**

Write `load_directory(folder)` that walks a messy folder containing a good PDF (page 2 is a scan), an **exact duplicate** of that PDF in a subfolder, a corrupt `.pdf`, a `.txt`, an HTML file and an unsupported `.png`. It must: dispatch by extension, never crash, skip byte-identical duplicates, use relative paths as `source`, skip OCR-needed pages, and return a report (`loaded`, `skipped_unsupported`, `failed` with reason, `duplicates`, `needs_ocr_pages`).

<details><summary><b>Hint</b></summary>

A dict `{".pdf": load_pdf, ...}` for dispatch; `hashlib.sha256(path.read_bytes())` for duplicates; `try/except Exception` around each loader call. Generate PDFs with `canvas.Canvas(..., invariant=1)` so identical input gives identical bytes.

</details>

<details><summary><b>Solution</b></summary>

```python
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
    report = {
  "loaded": [
    "notes.txt",
    "policy.pdf",
    "sub/faq.html"
  ],
  "skipped_unsupported": [
    "logo.png"
  ],
  "failed": [
    [
      "corrupt.pdf",
      "PdfStreamError: Stream has ended unexpectedly"
    ]
  ],
  "duplicates": [
    [
      "sub/policy_copy.pdf",
      "policy.pdf"
    ]
  ],
  "needs_ocr_pages": [
    [
      "policy.pdf",
      2
    ]
  ]
}
[('notes.txt', None), ('policy.pdf', 1), ('sub/faq.html', None)]
```

---

### Q4. Debug - a cleaner that damages the text

**Problem**

This cleaner turns `COVID-\n19` into `COVID19`, `state-of-the-\nart` into `theart`, deletes paragraph breaks, and removes `Page 1 of 3` with a hard-coded page count. Find every bug and write a better cleaner.

```python
def bad_clean(text):
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+ of 3", "", text)
    return text
```

<details><summary><b>Hint</b></summary>

Join a hyphen-broken word only if the joined word appears elsewhere in the corpus (build a vocabulary); otherwise keep the hyphen. Remove only *whole lines* that match `Page \d+ of \d+`. Collapse single newlines but keep blank lines.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Three separate heuristics fail in three separate ways, and each hides: the output still *looks* like text. The lesson is to test cleaners on adversarial samples (real hyphens, paragraphs, varying page counts) and to prefer rules that are conservative: when unsure, keep the original text.

**Expected output**
```text
BAD : 'Every customer must read this. COVID19 rules apply. Page 1 of 7 We use state-of-theart scanners. Each customer gets a receipt.'
GOOD: 'Every customer must read this. COVID-19 rules apply.\n\nWe use state-of-the-art scanners. Each customer gets a receipt.'
```

---

### Q5. Production - incremental ingestion with a hash manifest

**Problem**

Write `sync_folder(folder, manifest_path, store, embed_fn)` that on each run: skips files whose hash is unchanged, re-processes new/changed files (deleting their old rows first), deletes rows of files that disappeared, and returns statistics including `chunks_embedded`. Use ids like `source#page`. Demonstrate three runs: first load, no change, then "edit one file and delete another".

<details><summary><b>Hint</b></summary>

Keep `manifest = {relative_path: sha256}` as JSON. Compare old vs current: `set(old) - set(current)` = deleted; same hash = unchanged; otherwise added/updated. Rewrite the manifest at the end.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Embedding is the expensive step, so make it proportional to what *changed*, not to corpus size. Run 2 embeds zero chunks; run 3 re-embeds one. Deterministic ids plus delete-before-insert guarantee no stale rows survive when a page disappears. In production, replace `MiniStore` with your vector database's upsert and delete-by-filter.

**Expected output**
```text
embedder: fake_embed
run 1: {'added': ['a.txt', 'b.txt', 'c.pdf'], 'updated': [], 'unchanged': [], 'deleted': [], 'chunks_embedded': 4}
run 2 (nothing changed): {'added': [], 'updated': [], 'unchanged': ['a.txt', 'b.txt', 'c.pdf'], 'deleted': [], 'chunks_embedded': 0}
run 3 (edit a, delete b): {'added': [], 'updated': ['a.txt'], 'unchanged': ['c.pdf'], 'deleted': ['b.txt'], 'chunks_embedded': 1}
ids in store: ['a.txt#0', 'c.pdf#1', 'c.pdf#2']
```
