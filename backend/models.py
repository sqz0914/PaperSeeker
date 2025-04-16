from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ExternalId(BaseModel):
    """Model for external identifiers of a paper"""
    source: str
    id: str

    def __str__(self):
        return f"{self.source}: {self.id}"

class PaperMetadata(BaseModel):
    """Model for paper metadata"""
    year: Optional[int] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    sha1: Optional[str] = None
    sources: Optional[List[str]] = None
    s2fieldsofstudy: Optional[List[str]] = None
    extfieldsofstudy: Optional[List[str]] = None
    external_ids: Optional[List[ExternalId]] = None

class Paper(BaseModel):
    """
    Model for a research paper with detailed metadata.
    Used in search results and paper management.
    """
    id: Optional[str] = None
    source: Optional[str] = None
    version: Optional[str] = None
    added: Optional[datetime] = None
    created: Optional[datetime] = None
    text: Optional[str] = None
    metadata: Optional[PaperMetadata] = None
    
    # Fields used in internal models
    title: Optional[str] = None
    authors: Optional[List[str]] = []
    abstract: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    topic: Optional[str] = Field(None, description="Research topic or category")
    publication_date: Optional[str] = None
    citations: Optional[int] = None
    keywords: Optional[List[str]] = None
    
    @classmethod
    def parse_data(cls, data: Dict[str, Any]) -> 'Paper':
        """
        Parse a paper data dictionary into a Paper object
        
        Args:
            data: Dictionary containing paper data
            
        Returns:
            Paper object with populated fields
        """
        paper = cls(
            id=data.get("id"),
            source=data.get("source"),
            version=data.get("version"),
            text=data.get("text")
        )
        
        # Parse date fields if present
        if "added" in data:
            try:
                paper.added = datetime.fromisoformat(data["added"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
                
        if "created" in data:
            try:
                paper.created = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        
        # Parse metadata if present
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata = data["metadata"]
            
            # Create metadata object
            paper.metadata = PaperMetadata(
                year=metadata.get("year"),
                title=metadata.get("title"),
                abstract=metadata.get("abstract"),
                sha1=metadata.get("sha1"),
                sources=metadata.get("sources"),
                s2fieldsofstudy=metadata.get("s2fieldsofstudy"),
                extfieldsofstudy=metadata.get("extfieldsofstudy")
            )
            
            # Parse external IDs if present
            if "external_ids" in metadata and isinstance(metadata["external_ids"], list):
                external_ids = []
                for ext_id in metadata["external_ids"]:
                    if isinstance(ext_id, dict) and "source" in ext_id and "id" in ext_id:
                        external_ids.append(ExternalId(
                            source=ext_id["source"],
                            id=ext_id["id"]
                        ))
                paper.metadata.external_ids = external_ids
            
            # Populate simplified fields for compatibility
            paper.title = metadata.get("title", "")
            paper.abstract = metadata.get("abstract", "")
            if "year" in metadata:
                paper.year = str(metadata["year"])
            
            # Set topic from fields of study if available
            if metadata.get("s2fieldsofstudy") and len(metadata["s2fieldsofstudy"]) > 0:
                paper.topic = metadata["s2fieldsofstudy"][0]
            elif metadata.get("extfieldsofstudy") and len(metadata["extfieldsofstudy"]) > 0:
                paper.topic = metadata["extfieldsofstudy"][0]
                
            # Extract DOI URL if available
            if metadata.get("external_ids"):
                for ext_id in metadata["external_ids"]:
                    if ext_id.get("source") == "DOI" and ext_id.get("id"):
                        paper.url = f"https://doi.org/{ext_id['id']}"
                        break
        
        return paper
    
    def prepare_paper_embedding_content(self) -> str:
        """
        Create a structured text representation of a paper for embedding.
        
        Args:
            paper: Paper object
            
        Returns:
            Formatted text optimized for embedding generation
        """
        # Create a structured text with clear sections
        sections = []
        
        # Title is most important - repeat and emphasize
        sections.append(f"TITLE: {self.title}")
        
        # Abstract provides a good summary
        if self.abstract:
            sections.append(f"ABSTRACT: {self.abstract}")
        
        # Add year if available
        if self.year:
            sections.append(f"YEAR: {self.year}")
        
        # Add authors if available
        if hasattr(self, 'authors') and self.authors:
            author_text = ", ".join(self.authors) if isinstance(self.authors, list) else str(self.authors)
            sections.append(f"AUTHORS: {author_text}")
        
        # Add main text if available - this will be truncated by the tokenizer
        if self.text:
            sections.append(f"CONTENT: {self.text}")
        
        # Join with newlines to create clear section separation
        return "\n\n".join(sections)

# Define request model
class SearchRequest(BaseModel):
    """
    Schema for search requests
    """
    query: str
    top_k: Optional[int] = 10
    use_llm: Optional[bool] = True