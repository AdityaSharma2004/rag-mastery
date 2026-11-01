# Practice - Multimodal RAG

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**How this runs offline.** `gemini-embedding-001` embeds *text only*, so every image is first turned into text. Sample images (a pricing slide, an error screenshot, a text-free "photo", a corrupt file) are **drawn in code with Pillow**. The text inside them is read by **real OCR** if `pytesseract` and the `tesseract` binary are installed (otherwise `visible_text` is empty and Q3/Q4 results will differ). The **vision caption** is a lookup table standing in for `gemini-2.5-flash`; the real call, `caption_with_gemini`, uses `types.Part.from_bytes(data=..., mime_type=...)` and is code-reviewed but cannot run here without an API key. Expected outputs are from the offline run (OCR output can vary slightly with the tesseract version).

---

### Q1. Warm-up - never trust an uploaded image

**Problem**

Write `load_asset(path)` that checks a file really is an image (not just named `.png`), allows only PNG/JPEG/WEBP, rejects images with too many pixels, and returns an asset with width, height and MIME type. Test it on a good slide, a photo-like image and a corrupt file.

<details><summary><b>Hint</b></summary>

`Image.open(...).verify()` raises on corrupt data, but it invalidates the object, so open the image again afterwards. Check `img.format`, not the filename.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: validate images and build assets (never trust uploaded files) ----------
ALLOWED = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
MAX_PIXELS = 4096 * 4096


def load_asset(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()                                   # detects corrupt files without decoding everything
        img = Image.open(io.BytesIO(raw))              # verify() invalidates the object: reopen
    except Exception as e:
        raise ValueError(f"not a valid image: {type(e).__name__}")
    if img.format not in ALLOWED:
        raise ValueError(f"unsupported format {img.format}")
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("image too large")
    return ImageAsset(path.name, hashlib.sha256(raw).hexdigest()[:12], img.width, img.height, ALLOWED[img.format])


def q1():
    with tempfile.TemporaryDirectory() as d:
        make_sample_images(d)
        for name in ["pricing_slide.png", "office_photo.png", "broken.png"]:
            try:
                a = load_asset(pathlib.Path(d) / name)
                print(f"{name:<20} OK   {a.width}x{a.height} {a.mime}")
            except ValueError as e:
                print(f"{name:<20} SKIP {e}")
```

</details>

**Why it works / what to learn**
Extensions lie; users upload anything. Validation protects the pipeline (and your API quota) from corrupt or oversized files, and the ingestion loop should *skip and report* a bad file rather than die.

**Expected output**
```text
pricing_slide.png    OK   760x330 image/png
office_photo.png     OK   640x360 image/png
broken.png           SKIP not a valid image: UnidentifiedImageError
```

---

### Q2. Apply - turn an image into structured text, cached by hash

**Problem**

Write a `Captioner` that returns `{type, caption, visible_text, key_facts}` for an image and caches results by content hash so the same bytes never cost a second model call. Offline: caption from the stand-in table plus real OCR for `visible_text`. Show the number of model calls for three requests where one image is repeated.

<details><summary><b>Hint</b></summary>

Compute `sha256(bytes)` in `load_asset`; check `self.cache[sha]` before calling. With a key set, use `caption_with_gemini(bytes, mime)` which sends `[Part.from_bytes(...), CAPTION_PROMPT]` and parses JSON.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: image -> structured text (caption + visible text), cached by hash -------
CAPTION_PROMPT = (
    "You are indexing images for a search system. Return JSON with keys: "
    '"type" (photo|chart|table|screenshot|diagram|other), "caption" (one factual sentence), '
    '"visible_text" (ALL text you can read in the image, verbatim), "key_facts" (list of short facts incl. every number).'
)


def caption_with_gemini(image_bytes, mime):
    """REAL call (needs GEMINI_API_KEY; reviewed against the google-genai API, not executed offline)."""
    from google.genai import types
    from gemini_utils import get_client, GEN_MODEL, _with_retry
    client = get_client()
    resp = _with_retry(lambda: client.models.generate_content(
        model=GEN_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), CAPTION_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    ))
    return json.loads(resp.text)


class Captioner:
    def __init__(self):
        self.cache, self.calls = {}, 0

    def describe(self, path):
        a = load_asset(path)
        if a.sha in self.cache:                        # same bytes => same description: never pay twice
            return self.cache[a.sha]
        self.calls += 1
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            data = caption_with_gemini(pathlib.Path(path).read_bytes(), a.mime)
        else:
            base = FAKE_CAPTIONS.get(a.path, {"type": "other", "caption": "an image", "key_facts": []})
            data = {**base, "visible_text": ocr_text(Image.open(path))}      # real OCR + stand-in caption
        self.cache[a.sha] = data
        return data


def q2():
    with tempfile.TemporaryDirectory() as d:
        make_sample_images(d)
        cap = Captioner()
        for name in ["pricing_slide.png", "office_photo.png"]:
            r = cap.describe(pathlib.Path(d) / name)
            print(name, "->", r["type"], "|", r["caption"][:60], "| visible_text:", repr(r["visible_text"][:60]))
        cap.describe(pathlib.Path(d) / "pricing_slide.png")
        print("model calls after 3 requests (1 repeated):", cap.calls)
```

</details>

**Why it works / what to learn**
The structured prompt asks for caption, *verbatim* visible text and every number, because retrieval depends on that text. The photo has no visible text (OCR is empty), so its caption carries all the meaning. The cache keyed by content hash makes re-ingestion free.

**Expected output**
```text
pricing_slide.png -> table | Slide listing subscription plans with monthly prices and an  | visible_text: 'Pricing plans Starter 499 rupees per month Growth 1299 rupee'
office_photo.png -> photo | Photograph of a bright yellow sun-like circle over a blue gr | visible_text: ''
model calls after 3 requests (1 repeated): 2
```

---

### Q3. Build - index images by their text and search them with a question

**Problem**

Build an `ImageIndex` that embeds `caption + visible_text + key_facts` of each image and returns the best image names for a text question. Ingest a folder (skip invalid files), then search three questions.

<details><summary><b>Hint</b></summary>

Concatenate the three fields into one text, embed it with the *document* task type, embed the question with the *query* type, and rank by dot product (unit vectors). Wrap `describe` in `try/except ValueError` so the corrupt file is skipped.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: index images by their text and retrieve them with a text question --------
def index_text(desc):
    return " ".join([desc["caption"], desc.get("visible_text", ""), " ".join(desc.get("key_facts", []))]).strip()


class ImageIndex:
    def __init__(self):
        self.items, self.vecs = [], np.zeros((0, 512), dtype=np.float32)

    def add(self, path, desc):
        self.items.append({"path": str(path), "name": pathlib.Path(path).name, "type": desc["type"], "text": index_text(desc)})
        self.vecs = toy_embed([it["text"] for it in self.items])      # (re)embed; with Gemini: embed_texts(..., RETRIEVAL_DOCUMENT)

    def search(self, question, k=2, min_score=0.05):
        scores = self.vecs @ toy_embed([question])[0]
        return [(self.items[i]["name"], round(float(scores[i]), 2)) for i in np.argsort(-scores)[:k] if scores[i] > min_score]


def build_index(folder):
    cap, idx = Captioner(), ImageIndex()
    for p in sorted(pathlib.Path(folder).glob("*.png")):
        try:
            idx.add(p, cap.describe(p))
        except ValueError:
            continue                                   # skip invalid files, keep ingesting
    return idx


def q3():
    with tempfile.TemporaryDirectory() as d:
        make_sample_images(d)
        idx = build_index(d)
        print("indexed:", [i["name"] for i in idx.items])
        for q in ["how much is the Growth plan", "why do I see a gateway error", "which image is the homepage banner"]:
            print(f"{q!r:<42} ->", idx.search(q, k=2))
```

</details>

**Why it works / what to learn**
A text question finds the right *image* because we indexed text derived from it, and the corrupt file was skipped without stopping ingestion. Note the photo is found through its caption alone (no text in the picture).

**Expected output**
```text
indexed: ['error_screenshot.png', 'office_photo.png', 'pricing_slide.png']
'how much is the Growth plan'              -> [('pricing_slide.png', 0.35), ('office_photo.png', 0.13)]
'why do I see a gateway error'             -> [('error_screenshot.png', 0.34), ('office_photo.png', 0.11)]
'which image is the homepage banner'       -> [('office_photo.png', 0.44)]
```

---

### Q4. Debug - "image search finds nothing"

**Problem**

After switching to a cheaper captioning prompt, images such as the pricing slide are never retrieved. Compare an index built from lazy captions (`"a slide"`, `"a web page"`) with the structured captions plus OCR. Explain the failure.

<details><summary><b>Hint</b></summary>

What text did you embed? If the caption does not contain the words or numbers a user would ask about, no embedding model can find the image.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: "image search finds nothing" - lazy captions vs informative captions -----
LAZY = {"pricing_slide.png": {"type": "other", "caption": "a slide", "key_facts": [], "visible_text": ""},
        "error_screenshot.png": {"type": "other", "caption": "a web page", "key_facts": [], "visible_text": ""}}


def q4():
    with tempfile.TemporaryDirectory() as d:
        make_sample_images(d)
        lazy_idx, good_idx = ImageIndex(), build_index(d)
        for name, desc in LAZY.items():
            lazy_idx.add(pathlib.Path(d) / name, desc)
        for q in ["how much is the Growth plan", "why do I see a gateway error"]:
            print(f"{q!r:<34} lazy captions: {lazy_idx.search(q)} | structured captions + OCR: {good_idx.search(q, 1)}")
        print("OCR available on this machine:", HAVE_OCR)
```

</details>

**Why it works / what to learn**
Retrieval can only match what the index text says. Lazy captions contain none of "Growth", "plan", "gateway" or the numbers, so nothing is found. The fix is in the **ingestion prompt**: ask for a factual caption, all visible text verbatim, and every number. Always test image retrieval with real questions after changing a captioning prompt.

**Expected output**
```text
'how much is the Growth plan'      lazy captions: [] | structured captions + OCR: [('pricing_slide.png', 0.35)]
'why do I see a gateway error'     lazy captions: [] | structured captions + OCR: [('error_screenshot.png', 0.34)]
OCR available on this machine: True
```

---

### Q5. Production - build the multimodal answer request with budgets

**Problem**

Write `build_request(question, hits, folder, max_images, max_bytes, min_score, image_types)` that always includes the retrieved captions/texts as sources, but attaches the **original image** only when it is a useful type (table, chart, screenshot, diagram), the match score is high enough, and count and byte budgets allow. Test with a pricing question and a "homepage banner" question.

<details><summary><b>Hint</b></summary>

Loop over hits in rank order; `continue` when the score is low or the type is not allowed; stop at `max_images`; track total bytes. In real code the image part is `types.Part.from_bytes(data=raw, mime_type=...)` and the whole list goes to `generate_content`.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: answer with the images: build the multimodal request safely -------------
@dataclass
class Part:
    kind: str                # "text" | "image"
    data: object


def build_request(question, hits, folder, max_images=2, max_bytes=3_000_000, min_score=0.2, image_types=("table", "chart", "screenshot", "diagram")):
    """Text parts always; attach the ORIGINAL image only for useful types, strong matches, within count and size budgets."""
    parts, used = [], 0
    ctx = "\n".join(f"[{i}] {h['name']}: {h['text']}" for i, h in enumerate(hits, 1))
    parts.append(Part("text", "Answer ONLY from the sources and attached images. Cite like [1].\n\nSOURCES:\n" + ctx + f"\n\nQUESTION: {question}"))
    attached = []
    for h in hits:
        if len([p for p in parts if p.kind == "image"]) >= max_images:
            break
        if h["score"] < min_score or h["type"] not in image_types:
            continue
        raw = (pathlib.Path(folder) / h["name"]).read_bytes()
        if used + len(raw) > max_bytes:
            continue
        used += len(raw)
        parts.append(Part("image", {"mime": "image/png", "bytes": len(raw)}))     # real code: types.Part.from_bytes(data=raw, mime_type=...)
        attached.append(h["name"])
    return parts, attached


def q5():
    with tempfile.TemporaryDirectory() as d:
        make_sample_images(d)
        idx = build_index(d)
        for q in ["how much is the Growth plan", "which image is the homepage banner"]:
            raw_hits = idx.search(q, k=3, min_score=0.0)
            hits = [{**next(i for i in idx.items if i["name"] == n), "score": s} for n, s in raw_hits]
            parts, attached = build_request(q, hits, d)
            print(f"{q!r}: parts={[p.kind for p in parts]} attached_images={attached}")
        print("real call:  client.models.generate_content(model='gemini-2.5-flash', contents=[Part.from_bytes(...), text])")
```

</details>

**Why it works / what to learn**
The pricing slide is a table with a strong score, so the model also *sees* it (it can read numbers directly and you can cite it). The decorative banner photo is not attached: it adds cost and no value. Budgets on count and bytes keep latency, quota and prompt size under control. Remember that numbers read from charts can be wrong, so cite the image and prefer source data when it exists.

**Expected output**
```text
'how much is the Growth plan': parts=['text', 'image'] attached_images=['pricing_slide.png']
'which image is the homepage banner': parts=['text'] attached_images=[]
real call:  client.models.generate_content(model='gemini-2.5-flash', contents=[Part.from_bytes(...), text])
```
