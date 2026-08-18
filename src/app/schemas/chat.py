from typing import Any, List

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: List[Message] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[Any] = []
    processing_time: float = 0.0
    cached: bool = False
    learned_from_feedback: bool = False
    correction_type: str = ""
    similarity_score: float = 0.0
    matched_question: str = ""
    correction_usage_id: int = 0
    correction_intent: str = ""
