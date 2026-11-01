# 06 - Metadata filtering in RAG

> Week 3 · Topic 6 of 7 · Time: ~75 min · Pure Python + numpy, runs offline

## 🎯 What you will learn

- You will be able to design a **metadata schema** for chunks (tenant, department, year, ACL groups, ...).
- You will be able to write and use a **filter language** (`$eq`, `$in`, `$gte`, `$and`, `$or`, ...) and map it to Chroma, Qdrant and pgvector.
- You will be able to explain **pre-filtering vs post-filtering** and why post-filtering silently returns too few results.
- You will be able to derive filters from a user's question safely, with a fallback.
- You will be able to build a **secure retriever** that enforces tenant isolation and permissions on the server, and test it for leaks.

## 🧠 The big idea in plain words

Similarity search answers "what is *closest in meaning*?". But real questions come with hard constraints that meaning cannot express:

- "Only **my company's** documents" (multi-tenancy)
- "Only **2024** policies" (freshness)
- "Only documents **I'm allowed to read**" (permissions)
- "Only the **HR** handbook" (scope)

Those constraints are stored as **metadata** next to each vector and applied as **filters**. Similarity ranks; filters decide who is even allowed to be ranked.

Laravel/MySQL analogy: this is `WHERE tenant_id = ? AND year >= ?` plus `ORDER BY similarity LIMIT k`. The vector part is the `ORDER BY`; the metadata part is the `WHERE`, and in a multi-tenant app a missing `WHERE tenant_id` is a data breach, exactly as in a Laravel global scope.

## 📚 Concepts, step by step

### 1. What metadata to store

| Field | Type | Typical use |
|---|---|---|
| `source`, `page`, `chunk_id` | str/int | citations, delete by source |
| `tenant` / `org_id` | str | multi-tenant isolation (mandatory filter) |
| `dept`, `doc_type`, `language` | str | scoping |
| `year`, `updated_at`, `valid_to` | int/date | freshness, versions |
| `groups` / `allowed_roles` | list[str] | access control (ACL) |
| `version`, `is_current` | int/bool | hide superseded documents |

Rules: use **stable simple types** (ints for years, ISO strings or epoch numbers for dates); keep values **lowercase and normalised**; do not put big text in metadata; decide the schema before ingestion because backfilling millions of rows is painful.

### 2. The filter language

Most vector databases share a Mongo-like language:

```python
{"tenant": "acme", "year": {"$gte": 2023}}                       # implicit AND
{"$and": [{"dept": {"$in": ["hr", "it"]}}, {"year": 2024}]}
{"$or": [{"dept": "finance"}, {"year": 2022}]}
{"groups": {"$overlap": ["managers", "all"]}}                    # list fields (name varies by DB)
```

| Idea | Chroma | Qdrant | pgvector (SQL) |
|---|---|---|---|
| equals | `{"dept": "hr"}` | `FieldCondition(key="dept", match=MatchValue(value="hr"))` | `WHERE dept = 'hr'` |
| in list | `{"dept": {"$in": [...]}}` | `MatchAny(any=[...])` | `dept = ANY(...)` |
| range | `{"year": {"$gte": 2023}}` | `Range(gte=2023)` | `year >= 2023` |
| and / or | `$and`, `$or` | `must`, `should`, `must_not` | `AND`, `OR` |

(Chroma requires `$and` for two or more conditions; Week 2 topic 05 covers each database in detail.) In Q1 you implement the language yourself so you know exactly what a database does.

### 3. Pre-filtering vs post-filtering

```
POST-filter:  rank ALL by similarity -> take top k -> drop rows that fail the filter
PRE-filter:   drop rows that fail the filter -> rank the rest -> take top k
```

Post-filtering is dangerous: if your tenant's documents are not among the global top-k, you get **fewer than k results, even zero**, although matching documents exist (Q2: ask for 3, get 1). Over-fetching (`k x 10`) mitigates but does not eliminate it, and wastes work.

Real databases do pre-filtering inside the index (Qdrant filterable HNSW, pgvector `WHERE` with iterative scans/partitions, Chroma `where`). Caveat for approximate (HNSW) indexes: very selective filters can reduce recall, so check your database's guidance and **index the payload fields you filter on** (Qdrant payload indexes, SQL B-tree indexes). For hard isolation of big tenants consider a separate collection or table partition per tenant.

### 4. Deriving filters from the question

Users say "leave rules 2023" or "VPN setup". You can extract filters automatically:

1. **Rules/regex:** years (`20\d\d`), known product or department keywords.
2. **LLM extraction:** ask `gemini-2.5-flash` with `generate_json` for `{"year":..., "dept":...}` using a strict schema and a list of allowed values.

Safety rules:
- Extract only what is **explicit**. A wrong guess hides the right answer.
- **Never** let extracted filters touch security fields; tenant and permissions come from the *authenticated session*, not from the question text.
- **Fallback:** if the filtered search returns nothing, retry with the guessed filters relaxed (but security filters kept) and tell the user ("no 2019 documents; showing the closest").

### 5. Security: tenant isolation and ACLs

The golden rule: **the server adds security filters; the client can never remove them.**

```python
where = {"$and": [ {"tenant": user.tenant},                  # from the session
                   {"year": {"$gte": cutoff}},               # business rule
                   caller_filter ]}                          # optional, AND-ed on top
```
and permissions are checked with **list/set logic, not string tricks** (Q4's bug: saving groups as `"hr_admin"` text and testing `"hr" in groups` leaks admin data to anyone in `hr`).

Also: enforce at **retrieval time** (not only at display), because anything that reaches the prompt can be leaked by the model. Log every query and the ids returned (audit trail), and write an automated **leak test**: for each user × many queries, assert that no forbidden chunk id is ever returned (Q5).

### 6. Updating and deleting by metadata

Metadata makes lifecycle operations possible: delete all chunks `source == "policy_v1.pdf"`, mark `is_current=False` on old versions, expire documents with `valid_to < today`. Idempotent ids from topic 01/02 plus metadata filters give you safe re-ingestion.

### 7. Common filter pitfalls

- **Type mismatch:** year stored as `"2024"` (string) but filtered with `2024` (int) returns nothing, silently.
- **Missing field:** a chunk without `year` fails `year >= 2023` (our `matches` returns False). Decide whether "unknown" should pass or fail; default to fail for security fields.
- **High cardinality** fields (user id per chunk) bloat indexes; use groups/roles instead.
- **Case/whitespace:** `"HR"` vs `"hr"`. Normalise at ingestion.

## 💻 Code walkthrough

### Stage 1 - the matcher

```python
def matches(meta, where):
    for key, cond in where.items():
        if key == "$and":   if not all(matches(meta, w) for w in cond): return False
        elif key == "$or":  if not any(matches(meta, w) for w in cond): return False
        else:
            val = meta.get(key)
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            for op, arg in ops.items():
                ok = {"$eq": val == arg, "$in": val in arg, ...}[op]    # (the real code uses if/elif)
                if not ok: return False
    return True
```
Recursive for `$and`/`$or`; a bare value means `$eq`; unknown operators **raise** (a silently ignored operator is a security bug).

### Stage 2 - pre vs post

```python
if mode == "pre":
    allowed = [i for i, m in enumerate(store.metas) if matches(m, where)]
    order = sorted(allowed, key=lambda i: -scores[i])[:k]
else:
    top = np.argsort(-scores)[: k * overfetch]
    order = [i for i in top if matches(store.metas[i], where)][:k]
```

### Stage 3 - filter extraction with fallback

`extract_filters(question)` returns `{"year": 2023}` and/or `{"dept": "it"}` only when unambiguous. `filtered_search_with_fallback` always starts with `{"tenant": tenant}` and relaxes only the guessed parts.

### Stage 4 - the secure retriever

```python
mandatory = [{"tenant": user["tenant"]}, {"year": {"$gte": current_year - max_age}}]
where = {"$and": mandatory + ([extra_where] if extra_where else [])}
allowed = [i for i, m in enumerate(metas) if matches(m, where) and fixed_allowed(m, user["groups"])]
```
Because `extra_where` is AND-ed, an attacker passing `{"tenant": "acme"}` as a globex user gets an empty result, not acme data (printed by Q5).

### Stage 5 - with a real database

The same `where` dictionaries work in Chroma: `collection.query(query_embeddings=[qv], n_results=5, where={"$and": [{"tenant": "acme"}, {"year": {"$gte": 2023}}]})`. In Qdrant use `Filter(must=[...])`; in pgvector add `WHERE tenant = %s AND year >= %s` before `ORDER BY embedding <=> %s LIMIT 5`.

## ⚠️ Common mistakes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| Fewer than k results (or none) | Post-filtering | Pre-filter in the database; over-fetch only as a stopgap |
| Filter returns nothing although data exists | Type mismatch (`"2024"` vs `2024`), case mismatch | Normalise types/case at ingestion; log the filter |
| Employee sees restricted content | Substring checks on strings, filter only in UI | Store ACL as lists, intersect sets, filter at retrieval |
| Cross-tenant leak | Tenant filter taken from request body | Take tenant from the session; AND it server-side |
| Filter extraction hides the right answer | LLM/regex guessed wrong | Extract only explicit values; fall back without guessed filters |
| Chroma error on two conditions | Missing `$and` | Wrap in `{"$and": [...]}` |
| Slow filtered queries | No payload/SQL index on filter fields | Create payload indexes (Qdrant) or B-tree indexes (Postgres) |
| Stale answers after a policy update | Old versions still indexed | `is_current` flag or delete by `source` |
| Unknown operator silently ignored | Lenient matcher | Raise on unknown operators |

## 🚀 Pro-level insights (what the top 2% know)

- **Security filters are not optional features; they are invariants.** Put them in one function (a "scope") that every query path goes through, and add a CI leak test.
- **Prefer roles/groups over per-user ids**, and compute a user's groups at login. Per-user ACL lists on every chunk explode in size and are hard to update.
- **Filter selectivity matters.** A filter matching 0.1% of vectors with an approximate index may hurt recall; databases handle this differently (Qdrant plans around it, pgvector has iterative scans in newer versions). Benchmark recall with your real filters.
- **Time-aware RAG:** combine a hard `valid_from/valid_to` filter with a soft recency boost for ranking.
- **Metadata can improve ranking too:** boost chunks from `doc_type="policy"` over `"email"`, or demote old versions, as a post-retrieval re-score.
- **Never trust the model with access control.** "Only answer if the user may see it" in a prompt is not a control. Filter before the model ever sees the text.
- **Test negative cases:** the most important RAG security tests assert what must *not* come back.

## 🧪 Mini lab (10-15 min)

1. Extend `CORPUS` with a third tenant and a `confidential=True` flag.
2. Add a filter `{"confidential": False}` to `SecureRetriever` for users without the `legal` group.
3. Write a loop that tries 20 queries for each user and asserts `leaks == 0`.
4. Add a deliberately buggy variant (substring ACL) and confirm your leak test catches it.
5. Change `max_age_years` and observe which chunks disappear.

## ✅ Recap checklist

- [ ] I can design a metadata schema with tenant, time, scope and ACL fields.
- [ ] I can write filters with `$and`, `$or`, `$in`, ranges and list conditions.
- [ ] I can explain why post-filtering returns too few results.
- [ ] I extract filters only when explicit and fall back safely.
- [ ] Tenant and permissions are enforced server-side at retrieval time.
- [ ] I have an automated leak test with negative assertions.

## 🎤 Interview questions

**1. Pre-filtering vs post-filtering?**
Pre-filtering restricts candidates before ranking, so you always get k valid results; post-filtering ranks first and may leave fewer than k (or none) after filtering.

**2. How do you implement multi-tenant RAG safely?**
Store `tenant` on every chunk; add a server-side mandatory tenant filter from the authenticated session; AND any caller filters; consider per-tenant collections for large/regulated tenants; test for leaks.

**3. How would you derive filters from a natural-language question?**
Regex or an LLM with a strict schema extracts explicit constraints (year, department). Never extract security filters from the text, and fall back to unfiltered-by-guess search when the filtered result is empty.

**4. Why is `"hr" in "hr_admin"` a security bug?**
Substring matching treats different roles as equal. Store groups as lists and use set intersection or exact `$in`/overlap operators.

**5. How do you keep filtered ANN search fast and accurate?**
Index the filtered fields, use a database with filter-aware ANN, benchmark recall under realistic filters, and use partitions/collections for very selective or large tenants.

## 🔗 Next

**07 - Handling long documents:** books, contracts and manuals that are far bigger than any context window: hierarchical summaries, map-reduce and structure-aware retrieval.
