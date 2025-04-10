from pydantic import BaseModel
from typing import List, Optional
from .paper import Paper

class SearchResponse(BaseModel):
    """
    Response model for search results.
    Contains a list of papers and optional augmented content from the RAG system.
    """
    papers: List[Paper]
    augmented_content: Optional[str] = None 