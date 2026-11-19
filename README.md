# Semantic Job Search

Semantic search over 1,000 job postings using pgvector, FastAPI, and sentence-transformers.

## Setup

1. `docker run --name pgvector-jobs -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=jobsdb -p 5432:5432 -v pgvector_data:/var/lib/postgresql/data -d pgvector/pgvector:pg16`
2. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. `pip install -r requirements.txt`
4. `docker exec -i pgvector-jobs psql -U postgres -d jobsdb < schema.sql`
5. `python ingest.py`
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
See `schema.sql` for the jobs, users, saved_jobs tables and the HNSW index.
