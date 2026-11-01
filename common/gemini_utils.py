"""
common/gemini_utils.py
======================
One tiny, reusable Gemini client used by EVERY lesson in this course.
Think of it as your `app/Services/GeminiService.php` in Laravel: all the ugly
parts (API key, retries, batching, caching, rate limits) live here, and the
lessons just call `embed_texts()` / `generate()`.

Only FREE models are used:
    * Generation : gemini-2.5-flash
    * Embeddings : gemini-embedding-001   (a.k.a. "models/gemini-embedding-001")

Setup:
    pip install -r requirements.txt
    cp .env.example .env      # then paste your free key from https://aistudio.google.com/apikey
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
load_dotenv()  # reads GEMINI_API_KEY from .env (like Laravel's env())

GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"      # "models/gemini-embedding-001" also works
DEFAULT_DIM = 768                         # 3072 is native; 768 / 1536 are supported via MRL
BATCH_SIZE = 50                           # embed_content accepts many texts per call
CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", ".embed_cache"))

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Create the Gemini client once (singleton, like a Laravel service container binding)."""
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY missing. Copy .env.example to .env and add your free key "
                "from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=key)
    return _client


# --------------------------------------------------------------------------
# Retry helper (free tier = rate limits, so 429s WILL happen)
# --------------------------------------------------------------------------
def _with_retry(fn, max_tries: int = 6):
    delay = 2.0
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            retryable = code in (429, 500, 502, 503, 504)
            if not retryable or attempt == max_tries:
                raise
            print(f"[gemini] {code} - retry {attempt}/{max_tries} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalise rows. REQUIRED for dims < 3072 (only the 3072 vector comes pre-normalised)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _cache_path(text: str, task_type: str, dim: int) -> Path:
    h = hashlib.sha256(f"{EMBED_MODEL}|{task_type}|{dim}|{text}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.npy"


def embed_texts(
    texts: Sequence[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    dim: int = DEFAULT_DIM,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Turn a list of strings into an (N, dim) float32 numpy array of unit vectors.

    task_type (tell the model WHY you embed - it changes the vector, improves recall):
        RETRIEVAL_DOCUMENT  -> use for the chunks you store
        RETRIEVAL_QUERY     -> use for the user's question
        SEMANTIC_SIMILARITY -> comparing two texts symmetrically
        CLASSIFICATION / CLUSTERING -> self explanatory
    Results are cached on disk, so re-running a lesson costs 0 API calls.
    """
    texts = list(texts)
    out: list[np.ndarray | None] = [None] * len(texts)
    todo_idx: list[int] = []

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(texts):
            p = _cache_path(t, task_type, dim)
            if p.exists():
                out[i] = np.load(p)
            else:
                todo_idx.append(i)
    else:
        todo_idx = list(range(len(texts)))

    client = get_client()
    for start in range(0, len(todo_idx), BATCH_SIZE):
        idxs = todo_idx[start : start + BATCH_SIZE]
        batch = [texts[i] for i in idxs]
        resp = _with_retry(
            lambda: client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim),
            )
        )
        vecs = _normalize(np.array([e.values for e in resp.embeddings], dtype=np.float32))
        for i, v in zip(idxs, vecs):
            out[i] = v
            if use_cache:
                np.save(_cache_path(texts[i], task_type, dim), v)
        time.sleep(0.7)  # be polite to the free-tier rate limit

    return np.vstack(out) if out else np.zeros((0, dim), dtype=np.float32)


def embed_query(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Embed ONE user question. Returns shape (dim,)."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY", dim=dim)[0]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate(
    prompt: str | Sequence,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    schema=None,
    max_output_tokens: int | None = None,
    thinking_budget: int | None = 0,
) -> str:
    """
    Call gemini-2.5-flash and return plain text.

    thinking_budget=0 turns "thinking" off: faster, cheaper, and plenty for RAG answers.
    Pass thinking_budget=-1 for dynamic thinking on hard reasoning questions.
    json_mode=True (+ optional pydantic `schema`) forces valid JSON output.
    """
    cfg = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        if thinking_budget is not None
        else None,
    )
    if json_mode or schema is not None:
        cfg["response_mime_type"] = "application/json"
    if schema is not None:
        cfg["response_schema"] = schema

    client = get_client()
    resp = _with_retry(
        lambda: client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
    )
    return resp.text or ""


def generate_json(prompt: str, system: str | None = None, **kw):
    """generate() + json.loads(). Strips accidental ```json fences."""
    raw = generate(prompt, system=system, json_mode=True, **kw).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.startswith("json") else raw
    return json.loads(raw)


def generate_stream(prompt: str, system: str | None = None, temperature: float = 0.2):
    """Yield text pieces as they arrive (for typing-effect UIs)."""
    client = get_client()
    for chunk in client.models.generate_content_stream(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
    ):
        if chunk.text:
            yield chunk.text


# --------------------------------------------------------------------------
# Similarity math (pure numpy - no API needed)
# --------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity. a: (d,) or (n,d); b: (m,d). Returns (m,) or (n,m)."""
    a = np.atleast_2d(a).astype(np.float32)
    b = np.atleast_2d(b).astype(np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sims = a @ b.T
    return sims[0] if sims.shape[0] == 1 else sims


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Return [(row_index, score), ...] of the k most similar rows, best first."""
    scores = cosine_sim(query_vec, matrix)
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


# --------------------------------------------------------------------------
# Sample corpus used by many lessons (so examples are consistent and offline-friendly)
# --------------------------------------------------------------------------
SAMPLE_DOCS: list[str] = [
    "Laravel uses Eloquent ORM to map database tables to PHP classes.",
    "MySQL FULLTEXT indexes allow fast keyword search on text columns.",
    "Redis is an in-memory key-value store often used for caching and queues.",
    "Qdrant is an open-source vector database written in Rust.",
    "Cosine similarity measures the angle between two vectors, ignoring their length.",
    "The Indian Constitution guarantees fundamental rights under Part III.",
    "A cat sat quietly on the warm windowsill watching the birds outside.",
    "Kittens are young cats that sleep most of the day and play in short bursts.",
    "Retrieval-Augmented Generation grounds an LLM's answer in documents you supply.",
    "AWS RDS is a managed relational database service for MySQL and PostgreSQL.",
]


if __name__ == "__main__":
    # Smoke test:  python common/gemini_utils.py
    v = embed_texts(["hello world"], dim=768)
    print("embedding shape:", v.shape, "norm:", float(np.linalg.norm(v[0])))
    print(generate("Reply with exactly: pong", thinking_budget=0))
