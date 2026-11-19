# Semantic Job Search

Semantic search over 1,000 job postings using pgvector, FastAPI, and sentence-transformers.

## Setup

1. `docker run --name pgvector-jobs -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=jobsdb -p 5432:5432 -v pgvector_data:/var/lib/postgresql/data -d pgvector/pgvector:pg16`
2. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. `pip install -r requirements.txt`
4. `docker exec -i pgvector-jobs psql -U postgres -d jobsdb < schema.sql`
5. `python ingest.py`   (also creates the HNSW index — idempotent)
6. `uvicorn main:app --host 0.0.0.0 --port 8000`

## Endpoints
- `GET /search?q=term&limit=5` — semantic search
- `GET /jobs/{id}` — full job details
- `POST /save` — save job for user
- `GET /health` — health check

## Example

```
curl -s 'http://localhost:8000/search?q=machine+learning+engineer&limit=3'
```

Returns top-3 ML roles with similarity 0.69–0.74.

## Schema
See `schema.sql` for the jobs, users, saved_jobs tables and the HNSW index
(`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` — idempotent).
---

# Project 2: Job Agent

LLM-powered agent that searches, inspects, and saves jobs through Project 1's API, using **Gemini 2.5 Flash** with tool calling and structured output.

## Architecture

- **Tools**: `search_jobs`, `get_job`, `save_job` (thin wrappers over Project 1's HTTP API)
- **Two-phase pattern**: Phase 1 = tool calling only; Phase 2 = structured output (`AgentResponse` Pydantic schema). Gemini 2.5 does not allow `response_schema` together with `tools`, so we separate them.
- **Memory**: last 10 turns per `session_id` loaded from the `messages` table (PostgreSQL).
- **Model**: `gemini-2.5-flash` via `google-genai` SDK, `api_version="v1beta"`.

## Running the Agent

Project 1 API must be running on port 8000. Start it first:

    uvicorn main:app --host 0.0.0.0 --port 8000 &

Then install deps and start the agent on port 8001:

    pip install google-genai python-dotenv
    echo "GEMINI_API_KEY=your-key-here" > .env
    uvicorn agent.agent_main:app --host 0.0.0.0 --port 8001 &

## Chat Example

    curl -s -X POST "http://localhost:8001/chat" \
      -H "Content-Type: application/json" \
      -d '{"session_id":"s1","message":"Find me a machine learning engineer job in New York","user_id":1}' \
      | python -m json.tool

Real output — the agent called `search_jobs` and reasoned about the results:

    {
      "recommendations": [
        {"job_id": 319, "title": "Data Scientist Machine Learning Engineer",
         "reason": "Direct match for ML engineer role in New York.", "match_score": 0.702},
        {"job_id": 293, "title": "Machine Learning Engineer",
         "reason": "Located in New York, aligns with your title + location.", "match_score": 0.647},
        {"job_id": 100, "title": "Machine Learning Engineer/Scientist",
         "reason": "Jane Street, ML engineer/scientist in New York.", "match_score": 0.632}
      ],
      "summary": "Found three machine learning engineer jobs in New York."
    }

## Multi-Turn Memory

Same `session_id` continues the conversation. The agent recalls prior turns:

    curl -s -X POST "http://localhost:8001/chat" \
      -H "Content-Type: application/json" \
      -d '{"session_id":"s1","message":"Save the second one for me","user_id":1}'
    # → agent calls save_job(user_id=1, job_id=293) because turn 1 recommended [319, 293, 100]

## Semantic vs Keyword: hit@5 Evaluation

`agent/compare_search.py` runs 20 queries through both keyword (ILIKE) and semantic (pgvector) search and reports hit@5. Real results from this dataset:

    Keyword (ILIKE) hit@5 : 16/20 = 80.0%
    Semantic search hit@5 : 20/20 = 100.0%
    Semantic advantage    : +4 queries
    Relative improvement  : +25.0%

The 4 queries where keyword failed and semantic won:

    machine learning intern            ✗    ✓   (no literal "machine learning intern" title)
    deep learning researcher           ✗    ✓   (no literal "deep learning researcher" title)
    artificial intelligence engineer   ✗    ✓   (titles say "Machine Learning Engineer")
    entry level analyst                ✗    ✓   ("entry level" not in titles — semantic understands concept)

Reproduce:

    python -m agent.compare_search


## Index Verification

At 1,000 rows the query planner defaults to `Seq Scan` — that's correct (a full
scan of 1,000 tiny rows is cheaper than graph traversal). The HNSW index is
present and wired correctly; PostgreSQL just chooses not to use it at this size.

To force index use and confirm it works:

    docker exec -it pgvector-jobs psql -U postgres -d jobsdb -c "
    SET LOCAL enable_seqscan = off;
    EXPLAIN ANALYZE SELECT id FROM jobs
    ORDER BY embedding <=> (SELECT embedding FROM jobs WHERE id=1) LIMIT 5;
    "

Expected: `Index Scan using jobs_embedding_hnsw on jobs`. At ~10k+ rows the
planner starts preferring the index on its own; at 100k+ it's an order of
magnitude faster than sequential scan.

## Dockerfile

Build and run (requires Project 1 API on :8000 and a Postgres container):

    docker build -t job-agent .
    docker run --network host -e GEMINI_API_KEY=$GEMINI_API_KEY job-agent
