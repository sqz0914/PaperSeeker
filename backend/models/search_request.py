from pydantic import BaseModel

class SearchRequest(BaseModel):
    """
    Request model for paper search.
    Defines search parameters including query, result count, and augmentation flag.
    """
    query: str
    top_k: int = 5
    augment: bool = False 