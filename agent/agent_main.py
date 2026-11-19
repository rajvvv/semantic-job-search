"""FastAPI app exposing POST /chat backed by the Gemini job agent."""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from .agent import run_agent
from .models import AgentResponse

app = FastAPI(title="Job Agent")


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    user_id: int = 1


@app.post("/chat", response_model=AgentResponse)
def chat(req: ChatRequest):
    try:
        return run_agent(req.session_id, req.message, req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
