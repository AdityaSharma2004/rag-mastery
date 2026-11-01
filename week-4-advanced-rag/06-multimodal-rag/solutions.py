"""
Week 4 · Topic 06 - Multimodal RAG: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
HOW MULTIMODAL RAG WORKS HERE: gemini-embedding-001 embeds TEXT only. So we turn every image into text (a caption written
by gemini-2.5-flash + the text visible in the picture), embed that text, retrieve images by their text, and at answer time
(optionally) give the original image bytes back to gemini-2.5-flash together with the question.

OFFLINE: sample images are drawn with Pillow; the text inside them is read by real OCR (pytesseract + the tesseract binary,
if installed); the vision CAPTION is a lookup table standing in for gemini-2.5-flash. The real Gemini call
(`caption_with_gemini`) is included and code-reviewed against the google-genai API but cannot run without a key.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, io, json, os, re, tempfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up so as that this you your show".split())


def toy_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            if w in _STOP:
                continue
            w = w[:-1] if len(w) > 3 and w.endswith("s") else w
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False


def ocr_text(img):
    if not HAVE_OCR:
        return ""
    return re.sub(r"\s+", " ", pytesseract.image_to_string(img)).strip()


# ---- sample images drawn in code (no downloads) --------------------------------
def _font(size=26):
    return ImageFont.load_default(size=size)


def draw_text_image(lines, size=(760, 330)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((24, 20 + i * 52), line, fill="black", font=_font())
    return img


def draw_photo_like(size=(640, 360)):
    """No text at all: gradient + shapes (stands in for a product photo)."""
    img = Image.new("RGB", size)
    px = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = (40 + x * 150 // size[0], 90 + y * 100 // size[1], 160)
    d = ImageDraw.Draw(img)
    d.ellipse((220, 90, 420, 290), fill=(230, 200, 60))
    return img


def make_sample_images(folder):
    folder = pathlib.Path(folder)
    specs = {
        "pricing_slide.png": draw_text_image(["Pricing plans", "Starter 499 rupees per month", "Growth 1299 rupees per month", "Annual billing: 2 months free"]),
        "error_screenshot.png": draw_text_image(["502 Bad Gateway", "The upstream server did not respond", "Retry after 30 seconds"]),
        "office_photo.png": draw_photo_like(),
    }
    for name, img in specs.items():
        img.save(folder / name)
    (folder / "broken.png").write_bytes(b"this is not an image")
    return folder


# what gemini-2.5-flash would say about each image (stand-in for the vision call)
FAKE_CAPTIONS = {
    "pricing_slide.png": {"type": "table", "caption": "Slide listing subscription plans with monthly prices and an annual billing discount.",
                          "key_facts": ["Starter plan 499 rupees", "Growth plan 1299 rupees", "annual billing gives two months free"]},
    "error_screenshot.png": {"type": "screenshot", "caption": "Browser error page showing a gateway failure message and a retry hint.",
                             "key_facts": ["502 Bad Gateway error", "retry after 30 seconds"]},
    "office_photo.png": {"type": "photo", "caption": "Photograph of a bright yellow sun-like circle over a blue gradient, used as the website hero banner.",
                         "key_facts": ["hero banner image on the homepage"]},
}


@dataclass
class ImageAsset:
    path: str
    sha: str
    width: int
    height: int
    mime: str
    meta: dict = field(default_factory=dict)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
