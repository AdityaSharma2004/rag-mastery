# 01 - Document loading: PDF, DOCX, HTML, CSV, web pages

> Week 3 · Topic 1 of 7 · Time: ~90 min · Everything runs offline: the sample files are generated in code · Libraries: `pypdf`, `python-docx`, `beautifulsoup4`, `pandas`

## 🎯 What you will learn

- You will be able to load PDF, DOCX, HTML, CSV and TXT files into **one common `Document(text, metadata)` shape**.
- You will be able to keep the metadata (file, page, heading, row) that makes **citations** possible later.
- You will be able to clean extracted text without damaging it (hyphenation, whitespace, headers/footers, tables).
- You will be able to build a **fault-tolerant directory loader** with a report (skipped, failed, duplicates, scanned pages).
- You will be able to do **incremental ingestion**: re-embed only files that changed.
- You will be able to scrape web pages responsibly (robots.txt, rate limits) and know when OCR is needed.

## 🧠 The big idea in plain words

A RAG system is only as good as the text you feed it. "Garbage in, garbage out" applies brutally here: if your PDF loader glues two columns together, drops table rows, or leaves "Page 3 of 12" in the middle of every paragraph, then your embeddings are noisy, retrieval is worse, and the LLM quotes nonsense.

**Loading** has one job: turn messy real-world files into clean text **plus a note saying where each piece came from**. Everything later (chunking, embedding, citations, filters) depends on that.

Laravel analogy: this is your **import pipeline** for a CSV/Excel upload. You parse each source format into one normalised "row" shape (an Eloquent model), validate it, skip and report bad rows, and keep a record of where each row came from. Nobody writes the controller logic against raw spreadsheets, and nobody should write RAG logic against raw PDFs.

## 📚 Concepts, step by step

### 1. The one shape: `Document`

```python
@dataclass
class Document:
    text: str
    metadata: dict      # source, page / heading / row, type, ...
```

Every loader returns `list[Document]`. Everything downstream (chunker, embedder, vector DB) only knows this shape. Adding a new file format = adding one loader.

**Metadata conventions** (decide once):

| Key | Meaning | Used for |
|---|---|---|
| `source` | file name or URL (stable, relative path) | citations, delete-by-source |
| `type` | `pdf`, `docx`, `html`, `csv`, `txt` | formatting citations, filters |
| `page` | 1-based page number (PDF) | "policy.pdf, p. 7" |
| `heading` | nearest section heading (DOCX/HTML) | "section 'Leave policy'" |
| `row` | row number (CSV) | "products.csv, row 3" |
| `needs_ocr` | page had no text layer | routing scanned pages |

### 2. PDF (`pypdf`)

```python
from pypdf import PdfReader
for n, page in enumerate(PdfReader(path).pages, start=1):
    text = page.extract_text() or ""
```

Facts you must know:
- A PDF is a **drawing instruction set**, not a text document. There are no real paragraphs, only positioned strings. Extraction reconstructs reading order and can get it wrong (multi-column layouts, headers, footnotes).
- **One `Document` per page** is the right default: it preserves `page` for citations. Chunk later.
- **Scanned PDFs** (photos of paper) contain *no text layer*; `extract_text()` returns an empty string. Detect it (`len(text.strip()) < 20`), flag `needs_ocr`, and route those pages to OCR. Gemini 2.5 Flash can read page images (Week 4, multimodal), or use Tesseract.
- Tables come out as space-separated text; if tables matter, consider `pdfplumber` or `PyMuPDF` and convert rows to "Header: value" sentences.

### 3. DOCX (`python-docx`)

A Word file has real structure: paragraphs with **styles** (`Heading 1`, `Normal`) and tables. Walk the body in order, start a new `Document` at every heading, and turn tables into text:

```
Plan: Pro; Price: 999; Seats: 5
```

Why "Header: value" sentences instead of pipes? An embedding model understands "Price: 999" far better than `| Pro | 999 | 5 |` stripped of its header.

### 4. HTML and web pages (`BeautifulSoup`)

Real pages are ~80% noise: navigation, footers, cookie banners, ads, scripts.

1. Remove `script, style, nav, footer, header, aside, form` and elements whose class looks like `cookie|banner|sidebar|advert|popup`.
2. Prefer the `<main>` or `<article>` element.
3. Split into sections at `h1`-`h3`; store the heading as `metadata["heading"]`.
4. Turn tables into per-row sentences, with the caption/heading as context (Q2).

**Scraping etiquette (not optional):**
- Check `robots.txt` (`urllib.robotparser`) and the site's terms; do not scrape what you are not allowed to.
- Send an honest `User-Agent`, set a **timeout**, throttle (≥ 1 s between requests), and cache pages.
- Prefer an official API or export when one exists. Never scrape personal data you have no right to process.

```python
import requests, urllib.robotparser as rp
robots = rp.RobotFileParser("https://example.com/robots.txt"); robots.read()
if robots.can_fetch("MyRagBot/1.0", url):
    html = requests.get(url, headers={"User-Agent": "MyRagBot/1.0"}, timeout=15).text
```

(The sandbox used to write this course cannot reach arbitrary websites, so our exercises feed HTML as strings; the fetch code above is for your machine.)

### 5. CSV and tabular data

One row = one `Document`, rendered as `"Column: value; Column: value"`, with `row` in metadata. Gotchas: encoding (`utf-8-sig` for Excel exports, `cp1252` fallback), empty cells, and very wide tables (select the columns that carry meaning). For numeric analysis ("average price") do **not** use RAG: use SQL/pandas (topic 06 and Week 4 discuss routing).

### 6. Cleaning text (carefully!)

| Problem | Naive fix | Why it hurts | Better |
|---|---|---|---|
| Line-break hyphenation `cus-\ntomer` | remove every `-\n` | also kills real hyphens: `COVID-\n19`→`COVID19`, `state-of-the-\nart`→`theart` | join only when the merged word exists in the corpus vocabulary, else keep the hyphen (Q4) |
| Hard-wrapped lines | `\s+ → " "` | destroys paragraph breaks `\n\n` that chunkers use | collapse single newlines, **keep** blank lines |
| Page header/footer repeated | regex `Page \d of 3` | hard-coded count; leaves holes | remove whole lines matching `Page \d+ of \d+`; or drop lines repeating on most pages |
| Odd Unicode (ligatures, nbsp) | ignore | `ﬁ` ≠ `fi`, breaks BM25 | `unicodedata.normalize("NFKC", text)` |
| Empty/tiny pages | keep | pollute the index | drop below a minimum length |

Rule: **cleaning must be reversible in spirit**. If in doubt, keep the text.

### 7. A directory loader that never dies

Real folders contain unsupported files, corrupt PDFs, copies, scanned pages. A production loader:
- dispatches by extension,
- wraps each file in `try/except` so one bad file does not kill a 10,000-file job,
- hashes bytes to skip **exact duplicates**,
- uses the **relative path** as `source` (unique, stable),
- returns a **report**: loaded, skipped, failed (with reason), duplicates, OCR-needed pages.

The report is your ingestion's observability: you can tell the business "37 files failed, 12 scanned pages need OCR".

### 8. Incremental ingestion (don't pay twice)

Embeddings cost quota and time, so re-embed only what changed:

```
manifest.json  = {relative_path: sha256(file bytes)}
for each file now on disk:
    unchanged hash  -> skip
    new/changed     -> load, clean, delete old rows of that source, embed, upsert
for each file in manifest but not on disk -> delete its rows
```

Ids are `source#page` (or `#row`/`#heading`), so re-ingesting overwrites. Deleting old rows *before* re-adding handles pages that vanished. Q5 builds and tests it.

### 9. Citations start here

Because every `Document` knows its source and locator, a `cite()` function can print `policy.pdf, p. 7`, `handbook.docx, section "Leave policy"`, `products.csv, row 3`. Chunkers must **copy** this metadata onto every chunk (topic 02), otherwise answers cannot cite sources.

## 💻 Code walkthrough

### Stage 1 - the loaders (excerpt)

```python
def load_pdf(path):
    docs = []
    for n, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = page.extract_text() or ""
        docs.append(Document(text, {"source": path.name, "page": n, "type": "pdf",
                                    "needs_ocr": len(text.strip()) < 20}))
    return docs

def load_csv(path):
    df = pd.read_csv(io.StringIO(read_text_robust(path.read_bytes())), dtype=str, keep_default_na=False)
    return [Document("; ".join(f"{c}: {v}" for c, v in row.items() if v != ""),
                     {"source": path.name, "row": int(i) + 1, "type": "csv"}) for i, row in df.iterrows()]
```

`dtype=str` stops pandas turning `007` into `7`; `keep_default_na=False` keeps the text "NA" as text. `read_text_robust` tries `utf-8-sig` then `cp1252` then `latin-1`.

The full set (`load_pdf`, `load_docx`, `load_html`, `load_csv`, `load_txt`, `clean`, `cite`, `fake_embed`) is at the top of `solutions.py`; every sample file is generated in code with `reportlab` and `python-docx`, so nothing needs downloading.

### Stage 2 - DOCX walk

```python
for child in d.element.body.iterchildren():             # paragraphs and tables, in document order
    if child.tag.endswith("}p"):
        p = Paragraph(child, d)
        if p.style.name.startswith(("Heading", "Title")): flush(); heading = p.text.strip()
        elif p.text.strip(): buf.append(p.text.strip())
    elif child.tag.endswith("}tbl"):
        buf.append(table_to_text([[c.text.strip() for c in row.cells] for row in Table(child, d).rows]))
```

Iterating the body (not `d.paragraphs`) keeps tables in their reading position.

### Stage 3 - HTML section splitter

```python
for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]): tag.decompose()
root = soup.find("main") or soup.find("article") or soup.body
for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
    ...  # headings start a new Document; paragraphs/list items accumulate
```

### Stage 4 - citations

```python
def cite(doc):
    m = doc.metadata; src = m.get("source", "unknown source")
    if m.get("type") == "pdf" and "page" in m: return f"{src}, p. {m['page']}"
    if m.get("type") in ("docx", "html") and m.get("heading"): return f'{src}, section "{m["heading"]}"'
    if m.get("type") == "csv" and "row" in m: return f"{src}, row {m['row']}"
    return src
```

### Stage 5 - run it

`python solutions.py` loads a generated messy folder, prints the report, and demonstrates incremental ingestion. No API key needed (a hashed bag-of-words stands in for embeddings unless `GEMINI_API_KEY` is set).

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Empty text from a PDF | Scanned PDF (no text layer) | Detect and OCR (Gemini vision/Tesseract); report it |
| Words glued: `thedatabase` or split `d a t a` | Odd PDF font encoding / kerning | Try another extractor (`pdfplumber`/`PyMuPDF`); inspect raw output |
| Paragraphs interleaved from two columns | Multi-column layout | Layout-aware extractor, or crop columns |
| Same footer text in every chunk | Page headers/footers kept | Remove repeating lines; regex whole lines |
| `COVID19`, `theart` after cleaning | Blind de-hyphenation | Vocabulary-aware join (Q4) |
| Tables become word salad | Table rows flattened | Convert rows to `Header: value` sentences |
| `UnicodeDecodeError` on CSV | Excel `cp1252` / BOM | `utf-8-sig` then `cp1252` fallback |
| Duplicate answers in results | Same file ingested twice under different names | Hash file bytes; dedupe |
| Ingestion dies on file 4,213 | One corrupt file raised | `try/except` per file + a report |
| Citations missing page numbers | Metadata dropped at chunking | Copy metadata onto every chunk |
| Re-run re-embeds everything | No manifest | Hash manifest + deterministic ids |

## 🚀 Pro-level insights (what the top 2% know)

- **Extraction quality is the #1 hidden lever.** Teams spend weeks tuning embeddings while their PDF text is broken. Always *eyeball 20 random extracted pages* before tuning anything else.
- **Keep the raw file and the extraction version.** Store `extractor="pypdf 5.x"` in metadata; when you improve extraction you know which docs to reprocess.
- **Layout-aware parsing** (title/section/table detection, e.g. `unstructured`, `docling`, Gemini vision) gives structured chunks; use it for high-value documents (contracts, manuals) where a wrong table cell is costly.
- **Route by file quality:** text-layer PDFs → fast path; scanned/complex pages → vision-model OCR with `gemini-2.5-flash` (watch free-tier limits; cache results by file hash).
- **Security:** documents can contain **prompt-injection** ("ignore previous instructions...") and PII. Sanitise/flag at load time, keep an audit trail of sources, and respect access control via metadata (`allowed_groups`).
- **Idempotency everywhere:** deterministic ids, content hashes, manifest, delete-then-insert per source.
- **Large crawls:** respect robots, throttle, use conditional requests (`ETag`/`Last-Modified`) to skip unchanged pages, store raw HTML separately.
- **When NOT to load into a vector store:** structured data you query with SQL (orders, prices) belongs in a database; give the LLM a SQL/tool interface instead of embedding rows.

## 🧪 Mini lab (10-15 min)

1. Create a folder with: one real PDF you own, a DOCX with a table, a CSV, an HTML file saved from a website.
2. Load them with the course loaders and print `cite(doc)` plus the first 150 characters of 5 random documents.
3. Find at least one extraction defect and fix it in `clean()` without damaging other text.
4. Run the directory loader twice with the manifest; confirm run 2 embeds nothing.
5. Edit one file, delete another, re-run, and check the report.

## ✅ Recap checklist

- [ ] All my loaders return `Document(text, metadata)`.
- [ ] Every document carries `source` plus a page/heading/row locator.
- [ ] I flag scanned pages and do not silently index empty text.
- [ ] My cleaner keeps real hyphens and paragraph breaks.
- [ ] My directory loader survives bad files and returns a report.
- [ ] I ingest incrementally with a hash manifest and deterministic ids.
- [ ] I scrape only what I am allowed to, politely.

## 🎤 Interview questions

**1. Why one `Document` per PDF page rather than one per file?**
It preserves page numbers for citations and limits blast radius; chunking later splits pages further while copying the page metadata.

**2. How do you detect a scanned PDF?**
`extract_text()` returns little or no text for pages that are images; flag `needs_ocr` and route them to OCR/vision.

**3. What is wrong with `text.replace("-\\n", "")`?**
It deletes legitimate hyphens at line ends (`COVID-\n19`, `state-of-the-\nart`); only join when the merged word is a known word, else keep the hyphen.

**4. How do you avoid re-embedding unchanged files?**
Store a manifest of file hashes; re-process only added/changed files, delete rows of removed files, use deterministic ids so upserts overwrite.

**5. Why convert table rows to "Header: value" text?**
Embedding models understand sentences with labels far better than bare cells or pipe-separated rows detached from their header.

## 🔗 Next

**02 - Text chunking strategies:** split these cleaned documents into retrievable pieces (fixed, recursive, semantic, by section) while keeping the metadata, which is the biggest quality lever in a RAG pipeline.
