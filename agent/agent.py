"""Gemini-powered job agent with two-phase tool-calling + structured output."""
import os
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg2

from .models import AgentResponse
from .tools import TOOL_FUNCTIONS

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL = "gemini-2.5-flash"
DB = dict(host="localhost", port=5432, dbname="jobsdb",
          user="postgres", password="postgres")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)

SYSTEM_PROMPT = (
    "You are a job-search assistant. Use the provided tools to help the user find and save jobs. "
    "Always use search_jobs to find candidate jobs before recommending anything. "
    "Never fabricate job IDs — only use IDs returned by the tools. "
    "After your tool work is done, you will be asked to produce a structured response."
)


search_jobs_decl = types.FunctionDeclaration(
    name="search_jobs",
    description="Semantic search over jobs. Returns list of jobs with id, title, company, location, similarity.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING,
                description="Natural-language query, e.g. 'entry-level ML engineer in NY'"),
            "limit": types.Schema(type=types.Type.INTEGER,
                description="Max results, default 5"),
        },
        required=["query"],
    ),
)

get_job_decl = types.FunctionDeclaration(
    name="get_job",
    description="Fetch full details for one job by ID.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "job_id": types.Schema(type=types.Type.INTEGER,
                description="Job ID from search results"),
        },
        required=["job_id"],
    ),
)

save_job_decl = types.FunctionDeclaration(
    name="save_job",
    description="Save a job for the user.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "user_id": types.Schema(type=types.Type.INTEGER,
                description="User ID (default 1)"),
            "job_id": types.Schema(type=types.Type.INTEGER,
                description="Job ID to save"),
        },
        required=["user_id", "job_id"],
    ),
)

JOB_TOOL = types.Tool(
    function_declarations=[search_jobs_decl, get_job_decl, save_job_decl]
)


def _conn():
    return psycopg2.connect(**DB)


def load_history(session_id: str, limit: int = 10) -> List[dict]:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
            (session_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_message(session_id: str, role: str, content: str) -> None:
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def _build_contents(history: List[dict]) -> List[types.Content]:
    contents = []
    for h in history:
        role = "model" if h["role"] == "assistant" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=h["content"])],
        ))
    return contents


def run_agent(session_id: str, user_message: str, user_id: int = 1) -> AgentResponse:
    save_message(session_id, "user", user_message)
    contents = _build_contents(load_history(session_id, 10))

    # ---------- Phase 1: tools only (no response_schema) ----------
    for _ in range(6):
        r = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[JOB_TOOL],
                temperature=0.2,
            ),
        )
        if not r.function_calls:
            break
        contents.append(r.candidates[0].content)
        for fc in r.function_calls:
            args = dict(fc.args or {})
            if fc.name == "save_job" and "user_id" not in args:
                args["user_id"] = user_id
            try:
                result = TOOL_FUNCTIONS[fc.name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=fc.name, response=result)],
            ))

    # ---------- Phase 2: structured output (no tools) ----------
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=(
            "Produce the final AgentResponse based only on the tool results above. "
            "Include 1-5 recommendations ranked by fit. "
            "Only use job_ids that actually appeared in search results — never invent IDs. "
            "Set match_score in [0,1]. 'reason' 1-2 sentences. 'summary' 1-2 sentences."
        ))],
    ))
    r2 = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgentResponse,
            temperature=0.1,
        ),
    )
    parsed = r2.parsed or AgentResponse.model_validate_json(r2.text)
    save_message(session_id, "assistant", parsed.model_dump_json())
    return parsed
