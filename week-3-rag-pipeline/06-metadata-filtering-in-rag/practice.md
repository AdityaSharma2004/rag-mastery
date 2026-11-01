# Practice - Metadata filtering in RAG

Five questions, rising difficulty. Attempt each before opening the hint, and the hint before the solution. Every solution is the exact code in `solutions.py` (`python solutions.py` runs all). **Everything runs offline, no API key**: a hashed bag-of-words stands in for embeddings. `Store`, `CORPUS` (10 chunks of two companies, `acme` and `globex`, with `tenant`, `dept`, `year` and `groups` metadata) and `fake_embed` are defined at the top of `solutions.py`.

---

### Q1. Warm-up - write the filter language

**Problem**

Implement `matches(meta, where)` supporting equality, `$ne`, `$in`, `$nin`, `$gt/$gte/$lt/$lte`, list operators `$contains` and `$overlap`, and `$and` / `$or` (nested). Unknown operators must raise an error. Test four queries over the corpus.

<details><summary><b>Hint</b></summary>

A bare value means `$eq`. For `$and` / `$or`, call `matches` recursively on each sub-filter. Missing fields should fail range checks.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q1: a tiny filter language ($eq $ne $in $nin $gt $gte $lt $lte $contains $overlap $and $or) ------
def matches(meta, where):
    for key, cond in where.items():
        if key == "$and":
            if not all(matches(meta, w) for w in cond):
                return False
        elif key == "$or":
            if not any(matches(meta, w) for w in cond):
                return False
        else:
            val = meta.get(key)
            ops = cond if isinstance(cond, dict) else {"$eq": cond}
            for op, arg in ops.items():
                if op == "$eq":        ok = val == arg
                elif op == "$ne":      ok = val != arg
                elif op == "$in":      ok = val in arg
                elif op == "$nin":     ok = val not in arg
                elif op == "$gt":      ok = val is not None and val > arg
                elif op == "$gte":     ok = val is not None and val >= arg
                elif op == "$lt":      ok = val is not None and val < arg
                elif op == "$lte":     ok = val is not None and val <= arg
                elif op == "$contains": ok = arg in (val or [])            # list field contains a value
                elif op == "$overlap": ok = bool(set(val or []) & set(arg))  # list field shares any value
                else:
                    raise ValueError(f"unknown operator {op}")
                if not ok:
                    return False
    return True


def q1():
    s = Store()
    queries = {
        "tenant=acme, year>=2024": {"tenant": "acme", "year": {"$gte": 2024}},
        "dept in (hr, it), tenant globex": {"$and": [{"tenant": "globex"}, {"dept": {"$in": ["hr", "it"]}}]},
        "finance OR year 2022": {"$or": [{"dept": "finance"}, {"year": 2022}]},
        "readable by managers": {"groups": {"$overlap": ["managers", "all"]}, "tenant": "acme"},
    }
    for name, where in queries.items():
        print(f"{name:<34}", [i for i, m in zip(s.ids, s.metas) if matches(m, where)])
```

</details>

**Why it works / what to learn**
This is exactly the semantics of Chroma/Mongo-style `where` clauses; writing it yourself makes their behaviour predictable. Raising on unknown operators prevents a silently ignored security filter.

**Expected output**
```text
tenant=acme, year>=2024            ['a1', 'a3', 'a6', 'a7']
dept in (hr, it), tenant globex    ['g1', 'g3']
finance OR year 2022               ['a2', 'a4', 'a6', 'g2']
readable by managers               ['a1', 'a2', 'a3', 'a5', 'a7']
```

---

### Q2. Apply - pre-filter vs post-filter

**Problem**

Implement `search(store, query, k, where, mode, overfetch)`. For the query "leave policy paid leave days" restricted to tenant `globex` with `k=3`, compare: no filter, post-filter, post-filter with `overfetch=10`, and pre-filter.

<details><summary><b>Hint</b></summary>

Pre: compute allowed row indexes first, sort those by score. Post: take `argsort(-scores)[:k*overfetch]`, then drop rows failing the filter.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q2: pre-filtering vs post-filtering -------------------------------------
def search(store, query, k=3, where=None, mode="pre", overfetch=1):
    scores = store.matrix @ fake_embed([query])[0]
    if mode == "pre":                                     # filter FIRST, then rank only the allowed rows
        allowed = [i for i, m in enumerate(store.metas) if not where or matches(m, where)]
        order = sorted(allowed, key=lambda i: -scores[i])[:k]
    else:                                                 # post: rank everything, take top k*overfetch, THEN filter
        top = np.argsort(-scores)[: k * overfetch]
        order = [int(i) for i in top if not where or matches(store.metas[i], where)][:k]
    return [(store.ids[i], round(float(scores[i]), 2)) for i in order]


def q2():
    s = Store()
    q, where = "leave policy paid leave days", {"tenant": "globex"}
    print("no filter    :", search(s, q, 3))
    print("post-filter  :", search(s, q, 3, where, mode="post"))
    print("post x10     :", search(s, q, 3, where, mode="post", overfetch=10))
    print("pre-filter   :", search(s, q, 3, where, mode="pre"))
```

</details>

**Why it works / what to learn**
Most top matches belong to the other tenant, so post-filtering keeps only 1 of the 3 requested results. Over-fetching helps but is a guess; pre-filtering returns all 3 valid rows every time. Real databases do pre-filtering inside the index.

**Expected output**
```text
no filter    : [('g1', 0.76), ('a1', 0.73), ('a2', 0.73)]
post-filter  : [('g1', 0.76)]
post x10     : [('g1', 0.76), ('g2', 0.43), ('g3', 0.0)]
pre-filter   : [('g1', 0.76), ('g2', 0.43), ('g3', 0.0)]
```

---

### Q3. Build - filters from the question, with a safe fallback

**Problem**

Write `extract_filters(question)` that returns `{"year": ...}` for an explicit 4-digit year and `{"dept": ...}` only when exactly one department wins on keywords. Then write `filtered_search_with_fallback(store, question, tenant)` that **always** enforces the tenant and, when the guessed filters return nothing, retries with the tenant filter alone and reports `relaxed=True`.

<details><summary><b>Hint</b></summary>

Build `where = {"tenant": tenant, **extract_filters(q)}`. If the result is empty, rebuild `where = {"tenant": tenant}`; never drop the tenant.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q3: turn the user's question into a filter (and fall back safely) ------
DEPT_WORDS = {"hr": ["leave", "holiday", "hiring", "policy"], "finance": ["salary", "payroll", "encashment", "tax"],
              "it": ["vpn", "laptop", "password", "wifi"]}


def extract_filters(question):
    """Rule-based: explicit year -> {'year': y}; strong department keywords -> {'dept': d}.
    Returns only what is explicit, so a wrong guess cannot hide the right answer."""
    where = {}
    m = re.search(r"\b(20\d\d)\b", question)
    if m:
        where["year"] = int(m.group(1))
    words = set(re.findall(r"[a-z]+", question.lower()))
    scored = {d: len(words & set(ws)) for d, ws in DEPT_WORDS.items()}
    best = max(scored, key=scored.get)
    if scored[best] >= 1 and list(scored.values()).count(scored[best]) == 1:      # unambiguous winner only
        where["dept"] = best
    return where


def filtered_search_with_fallback(store, question, tenant, k=3):
    where = {"tenant": tenant, **extract_filters(question)}       # tenant is ALWAYS enforced, even on fallback
    hits = search(store, question, k, where)
    if hits:
        return hits, where, False
    where = {"tenant": tenant}                                    # relax the *guessed* filters only
    return search(store, question, k, where), where, True


def q3():
    s = Store()
    for q in ["leave encashment rules 2023", "how do I set up the VPN", "tell me about the 2019 policy", "what is the dress code"]:
        hits, where, relaxed = filtered_search_with_fallback(s, q, "acme")
        print(f"{q!r:<34} filter={where} relaxed={relaxed} -> {[h[0] for h in hits]}")
```

</details>

**Why it works / what to learn**
"2019" matches no document, so the guessed filters are relaxed and the user still gets the closest chunks (tell them so in the UI). The tenant filter is security, not a guess, so it survives the fallback. For ambiguous questions (the dress code) no filter is guessed at all.

**Expected output**
```text
'leave encashment rules 2023'      filter={'tenant': 'acme', 'year': 2023} relaxed=False -> ['a4', 'a5']
'how do I set up the VPN'          filter={'tenant': 'acme', 'dept': 'it'} relaxed=False -> ['a7']
'tell me about the 2019 policy'    filter={'tenant': 'acme'} relaxed=True -> ['a3', 'a7', 'a1']
'what is the dress code'           filter={'tenant': 'acme'} relaxed=False -> ['a7', 'a1', 'a2']
```

---

### Q4. Debug - "an HR clerk saw salary bands"

**Problem**

ACL groups were saved as a comma string and checked with substring matching:

```python
groups_str = ",".join(meta["groups"])
return any(g in groups_str for g in user_groups)
```
User `["hr"]` is a normal clerk. Find *two* bugs (one leaks data, one hides data), and write the fixed check. Which chunk leaked?

<details><summary><b>Hint</b></summary>

Is `"hr"` a substring of `"hr_admin"`? And what does the special group `"all"` mean for a user who is not literally in a group called `all`?

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q4: "an employee saw salary data" - find the ACL bug -------------------
def buggy_allowed(meta, user_groups):
    """BUG: groups were saved as a comma string and checked with substring `in`."""
    groups_str = ",".join(meta["groups"])
    return any(g in groups_str for g in user_groups)          # "hr" in "hr_admin" -> True !


def fixed_allowed(meta, user_groups):
    """FIX: real list + set intersection; 'all' means everyone."""
    allowed = set(meta["groups"])
    return "all" in allowed or bool(allowed & set(user_groups))


def q4():
    s = Store()
    user = ["hr"]                                          # a normal HR clerk, NOT hr_admin
    leak = [i for i, m in zip(s.ids, s.metas) if m["tenant"] == "acme" and buggy_allowed(m, user)]
    safe = [i for i, m in zip(s.ids, s.metas) if m["tenant"] == "acme" and fixed_allowed(m, user)]
    print("buggy sees :", leak)
    print("fixed sees :", safe)
    print("LEAKED chunks:", sorted(set(leak) - set(safe)))
```

</details>

**Why it works / what to learn**
(1) Substring matching treats `hr_admin` as containing `hr`, so the clerk can read the salary bands chunk `a6` (leak). (2) The buggy check does not understand the public group `all`, so the clerk is *denied* the public chunks `a1, a2, a5, a7` (over-blocking). Exact set logic fixes both. Always test security code with negative AND positive cases.

**Expected output**
```text
buggy sees : ['a6']
fixed sees : ['a1', 'a2', 'a5', 'a7']
LEAKED chunks: ['a6']
```

---

### Q5. Production - a secure retriever and a leak test

**Problem**

Build `SecureRetriever.retrieve(question, user, k, extra_where)` that AND-s caller filters with mandatory server-side constraints: the user's tenant, a freshness rule (year ≥ current_year − 1 = 2023), and the ACL check from Q4. Log each query in an audit list. Then write a leak test over 4 users × 4 queries asserting no forbidden chunk is ever returned, and try a "sneaky" caller who passes `{"tenant": "acme"}` as a globex user.

<details><summary><b>Hint</b></summary>

`where = {"$and": mandatory + [extra_where]}`. Because everything is AND-ed, an override can only narrow results, never widen them. Make the leak test list, for each user, the chunk ids they must never see.

</details>

<details><summary><b>Solution</b></summary>

```python
# --- Q5: a retriever that ENFORCES tenant + ACL + freshness, with a leak test -
class SecureRetriever:
    """Caller-supplied filters are AND-ed with mandatory server-side constraints; the caller cannot remove them."""
    def __init__(self, store, current_year=2024, max_age_years=1):
        self.store, self.current_year, self.max_age = store, current_year, max_age_years
        self.audit = []

    def retrieve(self, question, user, k=3, extra_where=None):
        mandatory = [
            {"tenant": user["tenant"]},
            {"year": {"$gte": self.current_year - self.max_age}},
        ]
        where = {"$and": mandatory + ([extra_where] if extra_where else [])}
        scores = self.store.matrix @ fake_embed([question])[0]
        allowed = [i for i, m in enumerate(self.store.metas)
                   if matches(m, where) and fixed_allowed(m, user["groups"])]
        order = sorted(allowed, key=lambda i: -scores[i])[:k]
        ids = [self.store.ids[i] for i in order]
        self.audit.append({"user": user["name"], "q": question, "returned": ids})
        return ids


def q5():
    s = Store()
    r = SecureRetriever(s)
    users = {
        "ravi (acme, employee)": dict(name="ravi", tenant="acme", groups=["employee"]),
        "meera (acme, managers)": dict(name="meera", tenant="acme", groups=["managers"]),
        "kiran (acme, hr_admin)": dict(name="kiran", tenant="acme", groups=["hr_admin"]),
        "gita (globex, employee)": dict(name="gita", tenant="globex", groups=["employee"]),
    }
    forbidden = {"ravi": {"a3", "a4", "a6"}, "meera": {"a4", "a6"}, "kiran": {"a3", "a4"}, "gita": {f"a{i}" for i in range(1, 8)}}
    queries = ["leave policy paid leave", "salary bands per year", "leave encashment", "VPN setup"]
    leaks = 0
    for label, u in users.items():
        ids_all = set()
        for q in queries:
            ids_all |= set(r.retrieve(q, u, k=5))
        bad = ids_all & forbidden[u["name"]]
        leaks += len(bad)
        print(f"{label:<26} sees {sorted(ids_all)}  leaks={sorted(bad)}")
    # the caller tries to override the tenant: it must still be ANDed, so nothing from acme appears
    sneaky = r.retrieve("leave policy", users["gita (globex, employee)"], k=5, extra_where={"tenant": "acme"})
    print("override attempt returns:", sneaky)
    print("TOTAL LEAKS:", leaks)
```

</details>

**Why it works / what to learn**
Security constraints are built on the server from the session and cannot be removed by callers. The stale 2022 chunk `a2` never appears (freshness rule), managers see `a3`, admins see `a6`, and the globex user sees only globex data. The override attempt returns `[]` because `tenant=globex AND tenant=acme` is empty. Keep this leak test in CI.

**Expected output**
```text
ravi (acme, employee)      sees ['a1', 'a5', 'a7']  leaks=[]
meera (acme, managers)     sees ['a1', 'a3', 'a5', 'a7']  leaks=[]
kiran (acme, hr_admin)     sees ['a1', 'a5', 'a6', 'a7']  leaks=[]
gita (globex, employee)    sees ['g1', 'g2', 'g3']  leaks=[]
override attempt returns: []
TOTAL LEAKS: 0
```
