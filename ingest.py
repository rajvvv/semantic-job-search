"""
Ingest jobs from jobs_1000.csv into Postgres with 384-d embeddings.
Model: sentence-transformers/all-MiniLM-L6-v2
"""
import time
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
CSV_PATH   = "jobs_1000.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB = dict(host="localhost", port=5432, dbname="jobsdb",
          user="postgres", password="postgres")

# ---------- 1. Load CSV ----------
print("[1/5] Loading CSV ...")
df = pd.read_csv(CSV_PATH)
print(f"      Loaded {len(df)} rows")
assert len(df) > 0, "CSV is empty"

# ---------- 2. Load model ----------
print(f"[2/5] Loading model {MODEL_NAME}")
print("      (First run downloads ~90 MB from HuggingFace; expect 1-3 min)")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
dim = model.get_sentence_embedding_dimension()
print(f"      Model loaded in {time.time()-t0:.1f}s. Embedding dim = {dim}")
assert dim == 384, f"Expected dim 384, got {dim}"

# ---------- 3. Embed ----------
print(f"[3/5] Embedding {len(df)} texts ...")
# Title + space + description. The model truncates to 256 tokens internally.
texts = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()

t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalize -> cosine similarity via <-> works perfectly
    convert_to_numpy=True,
)
print(f"      Done in {time.time()-t0:.1f}s. Shape = {embeddings.shape}")
assert embeddings.shape == (len(df), 384), "Embedding shape mismatch"

# ---------- 4. Insert ----------
print("[4/5] Connecting to Postgres and inserting ...")
conn = psycopg2.connect(**DB)
register_vector(conn)
cur = conn.cursor()

rows = []
for i, row in df.iterrows():
    rows.append((
        str(row["title"]),
        str(row["description"]),
        str(row["company"])  if pd.notna(row["company"])  else None,
        str(row["location"]) if pd.notna(row["location"]) else None,
        embeddings[i],
    ))

t0 = time.time()
cur.executemany(
    "INSERT INTO jobs (title, description, company, location, embedding) "
    "VALUES (%s, %s, %s, %s, %s)",
    rows,
)
conn.commit()
print(f"      Inserted {len(rows)} rows in {time.time()-t0:.1f}s")

# ---------- 5. Verify ----------
print("[5/5] Verifying ...")
cur.execute("SELECT COUNT(*) FROM jobs")
count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM jobs WHERE embedding IS NOT NULL")
non_null = cur.fetchone()[0]
cur.close()
conn.close()

print(f"      Jobs in DB       : {count}")
print(f"      With embeddings  : {non_null}")
print("Done.")
