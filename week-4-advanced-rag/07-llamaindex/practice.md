# Practice - LlamaIndex with Gemini

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all).

**Setup.** `pip install llama-index-core llama-index-llms-google-genai llama-index-embeddings-google-genai` (written against `llama-index-core 0.14.x`; LlamaIndex changes quickly, so if an import fails check its current docs). **Offline by default:** a hashed `HashEmbedding` (a `BaseEmbedding` subclass) and an `ExtractiveLLM` (a `CustomLLM` subclass) are plugged into LlamaIndex's extension points so its real machinery (nodes, index, retriever, query engine, persistence, auto-merging) runs without an API key. With `GEMINI_API_KEY` set the file calls `gemini_settings()` instead, which wires `GoogleGenAI(model="gemini-2.5-flash")` and `GoogleGenAIEmbedding(model_name="gemini-embedding-001", ...)`. That function is code-reviewed but not executed here, because the `GoogleGenAI` constructor checks the model over the network. `TEXTS`/`make_docs` (3 small support documents), `QA_TEMPLATE`, `HashEmbedding` and `ExtractiveLLM` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - documents, nodes, index, retriever

**Problem**

Split the three documents into nodes with `SentenceSplitter`, print the node count and the first node's metadata, build a `VectorStoreIndex`, and retrieve the top 2 chunks for "how many days for a refund" with their scores and file names.

<details><summary><b>Hint</b></summary>

`SentenceSplitter(chunk_size=..., chunk_overlap=...).get_nodes_from_documents(docs)`, then `VectorStoreIndex(nodes)` and `index.as_retriever(similarity_top_k=2).retrieve(question)`. `chunk_size` is counted in tokens.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: documents -> nodes -> index -> retriever ----------------------------------
def q1():
    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=0)          # sizes are in TOKENS
    nodes = splitter.get_nodes_from_documents(make_docs())
    print("mode:", MODE, "| nodes:", len(nodes))
    print("node 0 metadata:", nodes[0].metadata, "| text:", nodes[0].get_content()[:50] + "...")
    index = VectorStoreIndex(nodes)
    for hit in index.as_retriever(similarity_top_k=2).retrieve("how many days for a refund"):
        print(f"  score={hit.score:.2f} {hit.node.metadata['file_name']:<11} {hit.node.get_content()[:55]}...")
```

</details>

**Why it works / what to learn**
Each `Document` carries metadata (here the file name) which every node inherits, exactly the metadata propagation you built by hand in Week 3. `retrieve` returns `NodeWithScore` objects: score, node text and node metadata. The framework hid the embedding and similarity loops you wrote in Weeks 1-2.

**Expected output**
```text
mode: offline toys | nodes: 3
node 0 metadata: {'file_name': 'refunds.md'} | text: KiranaBooks refunds a subscription within 14 days ...
  score=0.51 refunds.md  KiranaBooks refunds a subscription within 14 days of pu...
  score=0.09 export.md   Owners can export all company data as a ZIP of CSV file...
```

---

### Q2. Apply - a query engine with your own prompt

**Problem**

Create a query engine from the index with `similarity_top_k=2` and your own grounding prompt (`text_qa_template` using `{context_str}` and `{query_str}`). Ask an answerable and an unanswerable question and print the answer plus the source file names and scores.

<details><summary><b>Hint</b></summary>

`index.as_query_engine(similarity_top_k=2, text_qa_template=QA_TEMPLATE)`; `response.source_nodes` is the list of retrieved nodes with scores.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The template is the Week 3 grounding prompt expressed in LlamaIndex terms. `source_nodes` is your debugging log: for the unanswerable question it shows weak scores (0.17, 0.00) and the answer refuses. (The offline "LLM" is extractive; with Gemini the wording will differ.)

**Expected output**
```text
Q: How long is the export download link valid?
A: The download link expires after 24 hours.
   best source: export.md 0.34
Q: Who won the cricket world cup?
A: I don't know based on the provided documents.
   best source: plans.md 0.17
```

---

### Q3. Build - persist, reload and refresh only what changed

**Problem**

Persist the index to a temporary directory, reload it, then refresh it with a new list of documents in which only `plans.md` changed (Growth price 1,299 → 1,499). Show which documents were re-embedded, delete `export.md`, and prove the new price is retrievable.

<details><summary><b>Hint</b></summary>

Documents need stable `doc_id`s (we use the file name). `index.refresh_ref_docs(docs)` returns one boolean per document (True = changed and re-indexed). Configure `Settings` before `load_index_from_storage`.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Only `plans.md` was re-embedded, saving embedding calls and quota: this is the framework's version of the hash-manifest incremental ingestion from Week 3 · 01. The embedding model used when reloading must be the one that built the index.

**Expected output**
```text
persisted files: ['default__vector_store.json', 'docstore.json', 'graph_store.json', 'image__vector_store.json'] ...
reloaded docs: ['export.md', 'plans.md', 'refunds.md']
refresh flags per document (True = re-embedded): {'refunds.md': False, 'plans.md': True, 'export.md': False}
after deleting export.md: ['plans.md', 'refunds.md']
fresh content retrieved: True
```

---

### Q4. Debug - the metadata error and the hidden defaults

**Problem**

Indexing a document raises `ValueError: Metadata length (...) is longer than chunk size`. Reproduce it, explain why metadata counts against the chunk size, and fix it without losing the metadata. Then print which embedding model and LLM `Settings` actually uses, and explain why you must set both explicitly.

<details><summary><b>Hint</b></summary>

LlamaIndex prepends metadata to the text that is embedded and sent to the LLM. Use `doc.excluded_embed_metadata_keys` and `doc.excluded_llm_metadata_keys` for keys that should be stored but not embedded.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
The long `description` was embedded into every chunk, leaving no room for the actual text. Excluding it keeps it in the node metadata (for display or filtering) but out of the embedded text. Separately, if `Settings` is not configured LlamaIndex falls back to defaults that point at other providers or models, so always set `Settings.llm` and `Settings.embed_model` explicitly (with Gemini: model names `gemini-2.5-flash` and `gemini-embedding-001`).

**Expected output**
```text
bug  : Metadata length (129) is longer than chunk size (64). Consider increasing t...
fixed: 2 nodes indexed; description still stored: True
embed model in use: HashEmbedding | llm in use: ExtractiveLLM (set both explicitly!)
```

---

### Q5. Production - auto-merging (parent-child) retrieval and the framework decision

**Problem**

Build a hierarchical node structure (parents of 256 tokens, leaves of 64 tokens), embed only the leaves, keep all nodes in the docstore, and wrap the retriever in `AutoMergingRetriever`. Compare the plain retriever with the auto-merging one for a broad question that matches several sibling leaves. Then, in your own words, decide when you would use the framework and when your own pipeline.

<details><summary><b>Hint</b></summary>

`HierarchicalNodeParser.from_defaults(chunk_sizes=[256, 64])`, `get_leaf_nodes(nodes)`, `storage.docstore.add_documents(nodes)`, `VectorStoreIndex(leaves, storage_context=storage)`, `AutoMergingRetriever(base_retriever, storage)`. Merging happens when more than about half of a parent's children are retrieved, so raise `similarity_top_k` if nothing merges.

</details>

<details><summary><b>Solution</b></summary>

```python
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
```

</details>

**Why it works / what to learn**
Four small leaves from the same section were retrieved, so the retriever replaced them by their parent: precise matching, larger context, exactly the idea of Week 4 · 03 in a few lines. The comparison table at the end is the real lesson: frameworks give breadth and speed, your own code gives control, security and clarity. Pin versions, keep a smoke test, and reuse your own components (cached embeddings, tenant filters, evaluation) inside whatever framework you choose.

**Expected output**
```text
nodes: 8 total, 7 leaves
plain retriever   -> 4 leaf chunks, avg chars ~ 170
auto-merging      -> 1 chunks, avg chars ~ 960 (siblings merged into parents)

DIY pipeline (Weeks 1-3)        vs  LlamaIndex
  full control, tiny dependency |  loaders, parsers, retrievers, evaluators ready-made
  you debug your own code       |  deep abstraction stack, fast-moving API
  best for a focused product    |  best for prototypes and many data sources
```
