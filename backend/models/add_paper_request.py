from pydantic import BaseModel, Field
from typing import List, Optional

class AddPaperRequest(BaseModel):
    """
    Request model for adding a new paper to the system.
    Contains all necessary metadata for a paper.
    """
    title: str
    authors: List[str]
    abstract: str
    year: Optional[str] = None
    url: Optional[str] = None
    topic: Optional[str] = Field(None, description="Research topic or category")
    text: Optional[str] = None
    pdf_url: Optional[str] = None
    publication_date: Optional[str] = None
    citations: Optional[int] = None
    keywords: Optional[List[str]] = None 