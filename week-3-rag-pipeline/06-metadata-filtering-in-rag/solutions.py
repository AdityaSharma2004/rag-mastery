"""
Topic 06 - Metadata filtering in RAG: solutions to practice.md (Q1..Q5).

Run everything:   python solutions.py
Pure Python + numpy, runs OFFLINE. A hashed bag-of-words stands in for embeddings (swap in
`embed_texts(..., task_type="RETRIEVAL_DOCUMENT")` / `embed_query` for real Gemini vectors).
The filter language mimics Chroma/Mongo (`$eq $in $gte $and $or ...`) so ideas transfer to Chroma, Qdrant and pgvector.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))

import hashlib
import re

import numpy as np


def fake_embed(texts, dim=256):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


# A small multi-tenant corpus. `groups` = who may read the chunk (ACL).
CORPUS = [
    ("a1", "Leave policy: employees get 24 days of paid leave per year.", dict(tenant="acme", dept="hr", year=2024, groups=["all"])),
    ("a2", "Leave policy: employees got 20 days of paid leave per year.", dict(tenant="acme", dept="hr", year=2022, groups=["all"])),
    ("a3", "Leave policy for managers: approve leave requests within 2 days.", dict(tenant="acme", dept="hr", year=2024, groups=["managers"])),
    ("a4", "Leave encashment: unused leave days are paid out at basic salary.", dict(tenant="acme", dept="finance", year=2023, groups=["finance"])),
    ("a5", "Leave of absence rules for contractors: unpaid leave only.", dict(tenant="acme", dept="hr", year=2023, groups=["all"])),
    ("a6", "Salary bands: grade L5 earns 18 to 24 lakh per year.", dict(tenant="acme", dept="finance", year=2024, groups=["hr_admin"])),
    ("a7", "VPN setup: install the client and sign in with your work email.", dict(tenant="acme", dept="it", year=2024, groups=["all"])),
    ("g1", "Leave policy: Globex staff get 18 days of paid leave.", dict(tenant="globex", dept="hr", year=2024, groups=["all"])),
    ("g2", "Globex leave encashment is paid in December.", dict(tenant="globex", dept="finance", year=2023, groups=["all"])),
    ("g3", "Globex VPN uses hardware tokens for login.", dict(tenant="globex", dept="it", year=2024, groups=["all"])),
]


class Store:
    def __init__(self):
        self.ids = [c[0] for c in CORPUS]
        self.texts = [c[1] for c in CORPUS]
        self.metas = [c[2] for c in CORPUS]
        self.matrix = fake_embed(self.texts)


# region q1
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
# endregion q1


# region q2
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
# endregion q2


# region q3
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
# endregion q3


# region q4
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
# endregion q4


# region q5
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
# endregion q5


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        print(f"\n===== {fn.__name__} =====")
        fn()
