-- ============================================================
-- Semantic Job Search — Database Schema
-- ============================================================

-- 1. Enable the pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Jobs table
--    embedding: 384-dimensional vector from all-MiniLM-L6-v2
CREATE TABLE jobs (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    company     TEXT,
    location    TEXT,
    embedding   vector(384)
);

-- 3. Users table (minimal for the project)
CREATE TABLE users (
    id    SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL
);

-- 4. Saved jobs (many-to-many between users and jobs)
CREATE TABLE saved_jobs (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id   INTEGER NOT NULL REFERENCES jobs(id)  ON DELETE CASCADE,
    saved_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, job_id)
);

-- 5. Seed a default user so /save works immediately
INSERT INTO users (email) VALUES ('test@example.com')
ON CONFLICT (email) DO NOTHING;

-- 6. Helpful index for saved_jobs lookups by user
CREATE INDEX idx_saved_jobs_user ON saved_jobs(user_id);
