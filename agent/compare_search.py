"""Compare keyword (ILIKE) vs semantic search using hit@5."""
import requests
import psycopg2
from .test_queries import TEST_QUERIES

DB = dict(host="localhost", port=5432, dbname="jobsdb",
          user="postgres", password="postgres")
BASE = "http://localhost:8000"
LIMIT = 5


def keyword_search(query):
    conn = psycopg2.connect(**DB)
    try:
        cur = conn.cursor()
        pat = f"%{query}%"
        cur.execute(
            "SELECT id, title FROM jobs "
            "WHERE title ILIKE %s OR description ILIKE %s "
            "ORDER BY (title ILIKE %s) DESC, id LIMIT %s",
            (pat, pat, pat, LIMIT),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]


def semantic_search(query):
    r = requests.get(f"{BASE}/search", params={"q": query, "limit": LIMIT}, timeout=30)
    r.raise_for_status()
    return r.json()


def hit_at_5(results, needle):
    n = needle.lower()
    return any(n in (r.get("title") or "").lower() for r in results)


def main():
    kw_hits = sem_hits = 0
    print(f"{'Query':<32} {'KW':>3} {'SEM':>4}  Expected substring")
    print("-" * 78)
    for case in TEST_QUERIES:
        q = case["query"]
        exp = case["expected_title_contains"]
        kw = keyword_search(q)
        sem = semantic_search(q)
        kh = hit_at_5(kw, exp)
        sh = hit_at_5(sem, exp)
        kw_hits += kh
        sem_hits += sh
        kw_mark = "✓" if kh else "✗"
        sem_mark = "✓" if sh else "✗"
        print(f"{q:<32} {kw_mark:>3} {sem_mark:>4}  {exp}")

    n = len(TEST_QUERIES)
    print("-" * 78)
    print(f"Keyword (ILIKE) hit@5 : {kw_hits}/{n} = {kw_hits/n:.1%}")
    print(f"Semantic search hit@5 : {sem_hits}/{n} = {sem_hits/n:.1%}")
    print(f"Semantic advantage    : {sem_hits - kw_hits:+d} queries")
    print(f"Relative improvement  : {((sem_hits - kw_hits) / max(kw_hits, 1)) * 100:+.1f}%")


if __name__ == "__main__":
    main()
