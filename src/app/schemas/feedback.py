from typing import List, Optional
from pydantic import BaseModel

class FeedbackItemRequest(BaseModel):
    selected_text: str
    feedback: str
    original_question: str
    full_response: str
    intent: str = "correction"  # correction | expansion | tone

class FeedbackRequest(BaseModel):
    items: List[FeedbackItemRequest]
