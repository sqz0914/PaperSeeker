from pydantic import BaseModel
from typing import Optional

class QueryResponse(BaseModel):
    """
    Model representing a response to a paper query.
    Contains the response text and optional citation.
    """
    response: str
    citation: Optional[str] = None 