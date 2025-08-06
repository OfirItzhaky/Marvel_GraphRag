from typing import Optional
from pydantic import BaseModel

class MarvelState(BaseModel):
    """
    Represents the state passed between nodes in the LangGraph flow.

    Tracks the user query, its classified type, raw graph response, and the final formatted LLM output.

    Fields:
    - query: The original user question.
    - query_type: The routing category determined from the query.
    - raw_result: The unformatted response returned from the graph query.
    - final_response: The final response to return to the user after formatting.
    """

    query: str
    query_type: Optional[str] = "default"
    raw_result: Optional[str] = ""
    final_response: Optional[str] = ""
