from typing import Optional
from pydantic import BaseModel

class MarvelState(BaseModel):
    query: str
    query_type: Optional[str] = "default"
    raw_result: Optional[str] = ""
    final_response: Optional[str] = ""
