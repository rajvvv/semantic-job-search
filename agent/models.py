from pydantic import BaseModel, Field
from typing import List


class JobRecommendation(BaseModel):
    job_id: int = Field(..., description="The job's ID from search results")
    title: str = Field(..., description="The job's title")
    reason: str = Field(..., description="Why this job fits the user's request (1-2 sentences)")
    match_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score of fit")


class AgentResponse(BaseModel):
    recommendations: List[JobRecommendation]
    summary: str = Field(..., description="1-2 sentence summary of what the agent did")
