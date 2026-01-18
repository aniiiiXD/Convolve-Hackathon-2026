from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    query_text: str
    clinic_id: str
    limit: int = 10
    modality_filter: Optional[str] = None # "text" or "image"

class SearchResponse(BaseModel):
    score: float
    text_content: Optional[str] = None
    metadata: Dict[str, Any]

class ChatRequest(BaseModel):
    message_history: List[Dict[str, str]] # [{"role": "user", "content": "..."}]
    current_query: str
    clinic_id: str
