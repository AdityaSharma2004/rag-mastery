# 07 - LlamaIndex with Gemini

> Week 4 · Topic 7 of 7 · Time: ~75 min · Written against `llama-index-core 0.14.x`, `llama-index-llms-google-genai 0.11.x`, `llama-index-embeddings-google-genai 0.6.x`. The offline demo plugs toy embedding/LLM classes into LlamaIndex's real machinery.

## 🎯 What you will learn

- You will be able to map every concept in this course to its **LlamaIndex equivalent** (Document, node parser, embedding, index, retriever, query engine).
- You will be able to wire **`gemini-2.5-flash` and `gemini-embedding-001`** into LlamaIndex correctly (explicit model names, task types, dimensions).
- You will be able to build, persist, reload and **incrementally refresh** an index.
- You will be able to use **auto-merging (parent-child) retrieval** from the framework.
- You will be able to avoid the **classic beginner errors** and decide when a framework is worth it versus your own code.

## 🧠 The big idea in plain words

You have now built every part of a RAG system by hand: loaders, chunkers, embeddings, stores, retrieval, prompts, evaluation. A **framework** packages the same parts into reusable classes so you write less glue code.

Laravel analogy: after writing raw PDO queries, sessions and routing yourself, you use **Laravel**. You are not learning new ideas; you are learning names for ideas you already know, plus a lot of ready-made integrations. And like Laravel, a framework helps most when you understand what it does under the hood, which is why this lesson comes last.

## 📚 Concepts, step by step

### 1. Vocabulary map

| Our course | LlamaIndex |
|---|---|
| `Document(text, metadata)` (Week 3 · 01) | `Document(text=..., metadata=..., doc_id=...)` |
| loaders | `SimpleDirectoryReader`, hundreds of readers on LlamaHub |
| chunker (Week 3 · 02-03) | `SentenceSplitter(chunk_size, chunk_overlap)` (sizes in **tokens**), other node parsers |
| a chunk | `Node` / `TextNode` (text + metadata + relationships) |
| embedding helper | `Settings.embed_model` (`BaseEmbedding`) |
| LLM helper | `Settings.llm` |
| vector store | `VectorStoreIndex` (default in-memory store; Chroma, Qdrant, pgvector integrations) |
| top-k search | `index.as_retriever(similarity_top_k=k)` |
| RAG pipeline (Week 3 · 04) | `index.as_query_engine(...)` |
| prompt template (Week 3 · 05) | `PromptTemplate` passed as `text_qa_template` |
| parent-child (Week 4 · 03) | `HierarchicalNodeParser` + `AutoMergingRetriever` |
| evaluation (Week 4 · 04) | `llama_index.core.evaluation` modules |

### 2. Wiring Gemini (the part people get wrong)

```python
from google.genai import types
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)          # key from GEMINI_API_KEY / GOOGLE_API_KEY
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embed_batch_size=50,
    embedding_config=types.EmbedContentConfig(output_dimensionality=768),
)
```
Important details verified against the installed packages:
- **Always pass the model names.** The library defaults point to *other* models, not the free ones this course uses.
- The embedding wrapper automatically uses `RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions.
- `embedding_config` is where `output_dimensionality` (MRL truncation) goes. Truncated vectors are not pre-normalised; LlamaIndex's default similarity is cosine, so ranking is still correct.
- The `GoogleGenAI` constructor looks the model up over the network, so it needs a valid key and connectivity at construction time. That is why the offline demo uses LlamaIndex's extension points (`BaseEmbedding`, `CustomLLM`) instead.
- Set `Settings` **explicitly**, otherwise LlamaIndex may try its default (OpenAI) and fail or bill the wrong provider.

### 3. The core flow

```python
docs  = SimpleDirectoryReader("data").load_data()                 # or [Document(...)]
index = VectorStoreIndex.from_documents(docs, transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)])
engine = index.as_query_engine(similarity_top_k=4, text_qa_template=QA_TEMPLATE)
resp = engine.query("How long is the refund window?")
print(resp, [n.node.metadata for n in resp.source_nodes])
```
`resp.source_nodes` carries the retrieved chunks with scores and metadata: your citations and debugging log.

### 4. Metadata is part of the embedded text

LlamaIndex prepends metadata to each node's text before embedding and before sending to the LLM, and it **counts metadata tokens against `chunk_size`**. Two consequences:
- Long metadata → `ValueError: Metadata length (N) is longer than chunk size`.
- Metadata you only want for display/filtering should be excluded: `doc.excluded_embed_metadata_keys = ["description"]` and `doc.excluded_llm_metadata_keys = [...]`.

### 5. Persistence and refresh

```python
index.storage_context.persist(persist_dir="storage")
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage"))   # needs Settings configured
index.refresh_ref_docs(new_docs)      # same doc_id: re-embeds only documents whose content hash changed
index.delete_ref_doc("old.md", delete_from_docstore=True)
```
Stable `doc_id`s (we use the file name) are the framework version of our deterministic ids (Week 3 · 01).

### 6. Parent-child in the framework

```python
nodes  = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128]).get_nodes_from_documents(docs)
leaves = get_leaf_nodes(nodes)
storage = StorageContext.from_defaults(); storage.docstore.add_documents(nodes)    # ALL levels in the docstore
index = VectorStoreIndex(leaves, storage_context=storage)                          # only leaves embedded
retriever = AutoMergingRetriever(index.as_retriever(similarity_top_k=6), storage)
```
If enough sibling leaves of one parent are retrieved (more than about half by default), the retriever **swaps them for the parent**: small chunks for matching, larger context for answering (Week 4 · 03). In Q5 four sibling leaves (174 characters each) merge into one 960-character parent.

### 7. Evaluation, async and streaming

- **Streaming:** `index.as_query_engine(streaming=True)`; iterate `response.response_gen`.
- **Async:** `aquery`, `aretrieve`; use with a rate limiter on the free tier.
- **Evaluators:** `FaithfulnessEvaluator`, `RelevancyEvaluator`, `RetrieverEvaluator` exist; verify against our hand-written metrics (topic 04) before trusting them.
- **Callbacks/instrumentation** give you traces of every LLM and embedding call, useful for cost tracking.

### 8. Framework or your own code?

| Choose LlamaIndex (or LangChain) when | Choose your own code when |
|---|---|
| prototyping quickly, many data sources/readers | a focused product with stable requirements |
| you want ready-made retrievers (auto-merging, hybrid, routers) | you need exact control of prompts, caching, filtering and security |
| a team already knows it | few dependencies, easier debugging, smaller attack surface |
| you will swap vector stores/LLMs often | latency/cost tuning at the byte level |

Honest advice: learn the concepts (this course), prototype with a framework if it saves time, and **replace pieces with your own code where you need control**. The skills transfer either way; frameworks change names every few months.

## 💻 Code walkthrough

### Stage 1 - plug points (offline)

```python
class HashEmbedding(BaseEmbedding):            # implement _get_query_embedding, _aget_query_embedding, _get_text_embedding
class ExtractiveLLM(CustomLLM):                # implement metadata, complete, stream_complete
Settings.embed_model = HashEmbedding(); Settings.llm = ExtractiveLLM()
```
These are the same extension points you would use to plug in a custom or local model.

### Stage 2 - nodes, index, retriever (Q1)

`SentenceSplitter(...).get_nodes_from_documents(docs)` returns nodes with `metadata`; `VectorStoreIndex(nodes)` embeds them; `as_retriever(similarity_top_k=2).retrieve(question)` returns `NodeWithScore` objects.

### Stage 3 - query engine with your prompt (Q2)

`as_query_engine(similarity_top_k=2, text_qa_template=QA_TEMPLATE)`; the template uses `{context_str}` and `{query_str}`. `response.source_nodes` shows what was retrieved, including for the unanswerable question.

### Stage 4 - persist/refresh (Q3)

Persist, reload, `refresh_ref_docs` (only `plans.md` is re-embedded), `delete_ref_doc`.

### Stage 5 - pitfalls (Q4) and auto-merging (Q5)

See the exercises. Both are real behaviours of the installed version.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Errors about OpenAI / missing API key | `Settings` not configured, library default used | Set `Settings.llm` and `Settings.embed_model` explicitly |
| `Metadata length ... longer than chunk size` | Long metadata counted in the chunk | Shorten metadata or exclude keys from embed/LLM text |
| Wrong/other Gemini model used | Relying on defaults | Pass `model="gemini-2.5-flash"`, `model_name="gemini-embedding-001"` |
| Index loads but queries fail | `Settings` not set before `load_index_from_storage` | Configure `Settings` first; embedding model must match what built the index |
| Embeddings from different models mixed | Changed model, kept old index | Re-embed (versioned collections; Week 2) |
| Chunk sizes look wrong | `chunk_size` is in tokens, not characters | Convert; check `len(node.get_content())` |
| Re-ingestion duplicates everything | Documents without stable `doc_id` | Set `doc_id`; use `refresh_ref_docs` |
| `GoogleGenAI(...)` fails at construction | Needs network/valid key to look up the model | Set `GEMINI_API_KEY`; check connectivity and model name |
| Auto-merging returns unmerged leaves | Not enough siblings retrieved / parents not in docstore | Add all nodes to the docstore; raise `similarity_top_k` |
| Breaks after upgrading | Fast-moving API | Pin versions; read release notes; keep a smoke test |
| 429 errors | Free-tier limits | Smaller `embed_batch_size`, retries, caching |

## 🚀 Pro-level insights (what the top 2% know)

- **Understand the abstraction by reading `source_nodes` and the final prompt** (turn on debug logging or use callbacks). When quality is bad, look at what actually reached the LLM.
- **Pin versions and write one smoke test** (index three documents, ask one question, assert a citation). Framework upgrades are the most common cause of "it worked yesterday".
- **Reuse your own components inside the framework:** a custom `BaseEmbedding` that calls your cached `embed_texts` (topic 05), a custom retriever that applies your tenant filters, a custom node postprocessor that dedupes and budgets context.
- **Keep security outside the framework's defaults:** tenant/ACL filters must still be enforced explicitly (Week 3 · 06).
- **Use it for breadth, not magic:** readers (Notion, Drive, Slack, databases) are the biggest time-saver; retrieval quality still comes from chunking, metadata and evaluation.
- **Measure framework overhead:** extra abstraction layers add latency and tokens (verbose default prompts). Compare against your hand-built pipeline on your gold set.
- **Portability:** store metadata and ids in a framework-neutral shape so you can migrate off any library.

## 🧪 Mini lab (10-15 min)

1. `pip install llama-index-core llama-index-llms-google-genai llama-index-embeddings-google-genai`.
2. With your `GEMINI_API_KEY` set, call `gemini_settings()` and index three of your own documents.
3. Ask five questions; print `source_nodes` metadata and scores; compare answers with your Week 3 pipeline.
4. Change a document, call `refresh_ref_docs`, and confirm only that document is re-embedded.
5. Replace `SentenceSplitter` with the hierarchical parser and check how many chunks are merged for a broad question.

## ✅ Recap checklist

- [ ] I can map each course concept to its LlamaIndex name.
- [ ] I configure `Settings` explicitly with Gemini model names.
- [ ] I know metadata counts toward chunk size and how to exclude keys.
- [ ] I persist, reload and refresh by `doc_id`.
- [ ] I can use auto-merging retrieval and explain what it does.
- [ ] I can argue when to use a framework and when not to.

## 🎤 Interview questions

**1. What does LlamaIndex give you that a hand-written pipeline does not?**
Ready-made readers, node parsers, retrievers and evaluators, and swappable integrations, so prototypes are faster; the costs are abstraction depth, API churn and less control.

**2. What is a node?**
A chunk of a document with its text, metadata and relationships (parent, previous, next), the unit that gets embedded and retrieved.

**3. How do you update an index when documents change?**
Give documents stable `doc_id`s and call `refresh_ref_docs`, which re-embeds only changed content; use `delete_ref_doc` for removed documents.

**4. How does auto-merging retrieval work?**
Documents are split hierarchically; only leaf nodes are embedded; if enough sibling leaves of a parent are retrieved, they are replaced by the parent, giving precise matching with larger context.

**5. When would you avoid a framework?**
For a narrow, security-sensitive product where you need exact control over prompts, filtering, caching and latency, or when you want minimal dependencies.

## 🔗 Next

**Capstone:** combine everything in this course into one project: load documents, chunk with parent-child, embed with `gemini-embedding-001` (cached), store in a vector database with metadata filters, retrieve with hybrid search plus reranking, expand queries, generate grounded answers with `gemini-2.5-flash`, and gate releases with an evaluation harness. Use the topic READMEs as your checklist and rebuild it from a blank folder without looking.
