from pydantic import BaseModel
from typing import Optional

class QueryResponse(BaseModel):
    """
    Model representing a response to a paper query
    """
    response: str
    citation: Optional[str] = None 