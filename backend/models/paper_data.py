from pydantic import BaseModel
from typing import Optional

class PaperData(BaseModel):
    """
    Model representing paper data for storage and retrieval
    """
    query: str
    response: str
    citation: Optional[str] = None 