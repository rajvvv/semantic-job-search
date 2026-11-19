"""Tool functions the agent can call. Each hits Project 1's API at :8000."""
import os
import requests

BASE_URL = os.getenv("PROJECT1_URL", "http://localhost:8000")
TIMEOUT = 30


def search_jobs(query: str, limit: int = 5) -> dict:
    r = requests.get(f"{BASE_URL}/search",
                     params={"q": query, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return {"jobs": r.json()}


def get_job(job_id: int) -> dict:
    r = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def save_job(user_id: int, job_id: int) -> dict:
    r = requests.post(f"{BASE_URL}/save",
                      json={"user_id": user_id, "job_id": job_id}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


TOOL_FUNCTIONS = {
    "search_jobs": search_jobs,
    "get_job": get_job,
    "save_job": save_job,
}
