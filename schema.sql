-- ============================================================
-- Semantic Job Search — Database Schema
-- ============================================================

-- 1. Enable the pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Jobs table
--    embedding: 384-dimensional vector from all-MiniLM-L6-v2
CREATE TABLE IF NOT EXISTS jobs (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    company     TEXT,
    location    TEXT,
    embedding   vector(384)
);

-- 3. Users table (minimal for the project)
CREATE TABLE IF NOT EXISTS users (
    id    SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL
);

-- 4. Saved jobs (many-to-many between users and jobs)
CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id   INTEGER NOT NULL REFERENCES jobs(id)  ON DELETE CASCADE,
    saved_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, job_id)
);

-- 5. Seed a default user so /save works immediately
INSERT INTO users (email) VALUES ('test@example.com')
ON CONFLICT (email) DO NOTHING;

-- 6. Helpful index for saved_jobs lookups by user
CREATE INDEX IF NOT EXISTS idx_saved_jobs_user ON saved_jobs(user_id);



-- 7. HNSW index on the embedding column (cosine distance)
--    Note: with an empty table this is created instantly; for very large
--    datasets, consider running this after ingest.py for faster bulk load.
CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw
    ON jobs
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 7. HNSW index on the embedding column (cosine distance)
--    Idempotent: safe to re-run on an existing DB.
--    Note: on a large table, prefer running this after bulk-loading data.
CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw
    ON jobs
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
