"""
Week 4 · Topic 07 - LlamaIndex with Gemini: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Needs:  pip install llama-index-core llama-index-llms-google-genai llama-index-embeddings-google-genai
(versions this was written against: llama-index-core 0.14.x, llama-index-llms-google-genai 0.11.x,
 llama-index-embeddings-google-genai 0.6.x. LlamaIndex changes quickly: if an import fails, check its current docs.)

OFFLINE by default: a hashed bag-of-words `HashEmbedding` and an `ExtractiveLLM` are plugged in through LlamaIndex's own
extension points (BaseEmbedding / CustomLLM) so the framework's real machinery (nodes, index, retriever, query engine,
persistence, auto-merging) runs without an API key. `gemini_settings()` shows the real Gemini wiring; its constructors can't run
offline (the LLM wrapper validates the model name over the network), so that function is code-reviewed, not executed here.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib, logging, os, re, tempfile, warnings

warnings.filterwarnings("ignore")
logging.getLogger("llama_index").setLevel(logging.ERROR)
import numpy as np

from llama_index.core import (Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever

_STOP = set("a an the is are of to in on for and or my i me do does how what when it be at can with from by up so as that this you your".split())


def _vec(text, dim=256):
    v = np.zeros(dim)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class HashEmbedding(BaseEmbedding):
    """Offline stand-in for GoogleGenAIEmbedding."""
    def _get_query_embedding(self, query):
        return _vec(query)

    async def _aget_query_embedding(self, query):
        return _vec(query)

    def _get_text_embedding(self, text):
        return _vec(text)


class ExtractiveLLM(CustomLLM):
    """Offline stand-in for GoogleGenAI: answers with the SOURCES sentence that best matches the QUESTION."""
    @property
    def metadata(self):
        return LLMMetadata(model_name="extractive-toy", context_window=8192, num_output=256)

    def complete(self, prompt, formatted=False, **kwargs):
        if "QUESTION:" not in prompt:
            return CompletionResponse(text="(no question)")
        body, question = prompt.split("QUESTION:", 1)
        qwords = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOP}
        best, score = None, 0
        for sent in re.split(r"(?<=[.!?])\s+|\n", body.split("SOURCES:")[-1]):
            s = len({w for w in re.findall(r"[a-z0-9]+", sent.lower())} & qwords)
            if s > score:
                best, score = sent.strip(), s
        return CompletionResponse(text=best if score >= 2 else "I don't know based on the provided documents.")

    def stream_complete(self, prompt, formatted=False, **kwargs):
        yield self.complete(prompt)


def gemini_settings():
    """The REAL wiring (not executed offline). Explicit model names: the library defaults are different models!"""
    from google.genai import types
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)           # reads GEMINI_API_KEY / GOOGLE_API_KEY
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001", embed_batch_size=50,
        embedding_config=types.EmbedContentConfig(output_dimensionality=768))       # task types handled by the wrapper
    # the wrapper uses RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions automatically


if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    gemini_settings()
    MODE = "gemini"
else:
    Settings.embed_model = HashEmbedding()
    Settings.llm = ExtractiveLLM()
    MODE = "offline toys"

QA_TEMPLATE = PromptTemplate(
    "Answer ONLY from the sources. If they do not contain the answer say: I don't know based on the provided documents.\n"
    "SOURCES:\n{context_str}\nQUESTION: {query_str}\nANSWER:"
)

TEXTS = {
    "refunds.md": "KiranaBooks refunds a subscription within 14 days of purchase. After 14 days payments are non-refundable. "
                  "Approved refunds reach your bank account in 5 to 7 working days. You can cancel at any time to stop renewal.",
    "plans.md": "The Starter plan costs 499 rupees per month and includes one user. The Growth plan costs 1,299 rupees per month "
                "and includes five users and unlimited invoices. Annual billing gives you two months free.",
    "export.md": "Owners can export all company data as a ZIP of CSV files from Settings > Data. Exports are ready within 10 minutes. "
                 "The download link expires after 24 hours. Backups are kept for 30 days.",
}


def make_docs(texts=TEXTS):
    return [Document(text=t, metadata={"file_name": n}, doc_id=n) for n, t in texts.items()]


# region q1
# --- Q1: documents -> nodes -> index -> retriever ----------------------------------
def q1():
    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=0)          # sizes are in TOKENS
    nodes = splitter.get_nodes_from_documents(make_docs())
    print("mode:", MODE, "| nodes:", len(nodes))
    print("node 0 metadata:", nodes[0].metadata, "| text:", nodes[0].get_content()[:50] + "...")
    index = VectorStoreIndex(nodes)
    for hit in index.as_retriever(similarity_top_k=2).retrieve("how many days for a refund"):
        print(f"  score={hit.score:.2f} {hit.node.metadata['file_name']:<11} {hit.node.get_content()[:55]}...")
# endregion q1


# region q2
# --- Q2: a query engine with your own prompt and visible sources ---------------------
def q2():
    index = VectorStoreIndex.from_documents(make_docs(), transformations=[SentenceSplitter(chunk_size=60, chunk_overlap=0)])
    engine = index.as_query_engine(similarity_top_k=2, text_qa_template=QA_TEMPLATE)
    for q in ["How long is the export download link valid?", "Who won the cricket world cup?"]:
        r = engine.query(q)
        print("Q:", q)
        print("A:", str(r).strip())
        top = r.source_nodes[0]
        print("   best source:", top.node.metadata["file_name"], round(top.score, 2))
# endregion q2


# region q3
# --- Q3: persist, reload and refresh only what changed ---------------------------------
def q3():
    with tempfile.TemporaryDirectory() as d:
        splitter = SentenceSplitter(chunk_size=60, chunk_overlap=0)
        index = VectorStoreIndex.from_documents(make_docs(), transformations=[splitter])
        index.storage_context.persist(persist_dir=d)
        print("persisted files:", sorted(os.listdir(d))[:4], "...")
        reloaded = load_index_from_storage(StorageContext.from_defaults(persist_dir=d))
        print("reloaded docs:", sorted(reloaded.ref_doc_info))
        changed = make_docs({**TEXTS, "plans.md": TEXTS["plans.md"].replace("1,299", "1,499")})
        flags = reloaded.refresh_ref_docs(changed)                            # same doc_id, compares content hashes
        print("refresh flags per document (True = re-embedded):", dict(zip([x.doc_id for x in changed], flags)))
        reloaded.delete_ref_doc("export.md", delete_from_docstore=True)
        print("after deleting export.md:", sorted(reloaded.ref_doc_info))
        hit = reloaded.as_retriever(similarity_top_k=1).retrieve("how much is the growth plan")[0]
        print("fresh content retrieved:", "1,499" in hit.node.get_content())
# endregion q3


# region q4
# --- Q4: the two errors every beginner hits ------------------------------------------------
def q4():
    # (a) metadata is part of the text that gets embedded and counted against chunk_size
    big_meta = {"file_name": "policy.md", "description": "long internal notes " * 40}
    doc = Document(text="Refunds take 5 to 7 days. " * 6, metadata=big_meta)
    try:
        VectorStoreIndex.from_documents([doc], transformations=[SentenceSplitter(chunk_size=64, chunk_overlap=0)])
    except ValueError as e:
        print("bug  :", str(e)[:75] + "...")
    doc.excluded_embed_metadata_keys = ["description"]                      # keep it for display, not for embedding
    doc.excluded_llm_metadata_keys = ["description"]
    idx = VectorStoreIndex.from_documents([doc], transformations=[SentenceSplitter(chunk_size=64, chunk_overlap=0)])
    print("fixed:", len(idx.docstore.docs), "nodes indexed; description still stored:", "description" in idx.docstore.docs[next(iter(idx.docstore.docs))].metadata)
    # (b) forgetting to configure Settings -> LlamaIndex would silently pick an OpenAI default
    from llama_index.core.settings import Settings as S
    print("embed model in use:", type(S.embed_model).__name__, "| llm in use:", type(S.llm).__name__, "(set both explicitly!)")
# endregion q4


# region q5
# --- Q5: parent-child (auto-merging) in 15 lines, and when to use the framework ---------------
def q5():
    long_texts = {"handbook.md": " ".join(f"Section {i} rule {i}: staff must do task {i} within {i + 2} days of the request. "
                                          f"Managers approve exceptions for task {i} in writing." for i in range(1, 9))}
    nodes = HierarchicalNodeParser.from_defaults(chunk_sizes=[256, 64]).get_nodes_from_documents(make_docs(long_texts))
    leaves = get_leaf_nodes(nodes)
    storage = StorageContext.from_defaults()
    storage.docstore.add_documents(nodes)                                    # the retriever needs ALL levels in the docstore
    index = VectorStoreIndex(leaves, storage_context=storage)                # but only LEAVES are embedded
    base = index.as_retriever(similarity_top_k=4)
    merging = AutoMergingRetriever(base, storage, verbose=False)
    q = "what must staff do within days for each task"
    b, m = base.retrieve(q), merging.retrieve(q)
    print(f"nodes: {len(nodes)} total, {len(leaves)} leaves")
    print("plain retriever   ->", len(b), "leaf chunks, avg chars ~", int(round(np.mean([len(x.node.get_content()) for x in b]), -1)))
    print("auto-merging      ->", len(m), "chunks, avg chars ~", int(round(np.mean([len(x.node.get_content()) for x in m]), -1)), "(siblings merged into parents)")
    print()
    print("DIY pipeline (Weeks 1-3)        vs  LlamaIndex")
    print("  full control, tiny dependency |  loaders, parsers, retrievers, evaluators ready-made")
    print("  you debug your own code       |  deep abstraction stack, fast-moving API")
    print("  best for a focused product    |  best for prototypes and many data sources")
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
