"""Semantic Job Search API."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB = dict(host="localhost", port=5432, dbname="jobsdb",
          user="postgres", password="postgres")

model: SentenceTransformer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading embedding model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded. API ready.")
    yield

app = FastAPI(title="Semantic Job Search", lifespan=lifespan)


def get_conn():
    conn = psycopg2.connect(**DB)
    register_vector(conn)
    return conn


class SaveRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    job_id:  int = Field(..., gt=0)


class JobSummary(BaseModel):
    id: int
    title: str
    company: str | None
    location: str | None
    similarity: float


class JobDetail(BaseModel):
    id: int
    title: str
    description: str
    company: str | None
    location: str | None


@app.get("/search", response_model=list[JobSummary])
def search(q: str, limit: int = 5):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty")
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be 1..50")
    qvec = model.encode(q, normalize_embeddings=True).tolist()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, company, location, "
            "1 - (embedding <=> %s::vector) AS similarity "
            "FROM jobs ORDER BY embedding <=> %s::vector LIMIT %s",
            (qvec, qvec, limit),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return [JobSummary(id=r[0], title=r[1], company=r[2],
                       location=r[3], similarity=float(r[4])) for r in rows]


@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: int):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, description, company, location "
            "FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobDetail(id=row[0], title=row[1], description=row[2],
                     company=row[3], location=row[4])


@app.post("/save")
def save_job(req: SaveRequest):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE id = %s", (req.user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404,
                                detail=f"user {req.user_id} not found")
        cur.execute("SELECT 1 FROM jobs WHERE id = %s", (req.job_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404,
                                detail=f"job {req.job_id} not found")
        cur.execute(
            "INSERT INTO saved_jobs (user_id, job_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (req.user_id, req.job_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()
    return {"status": "saved", "user_id": req.user_id, "job_id": req.job_id}


@app.get("/health")
def health():
    return {"status": "ok"}
