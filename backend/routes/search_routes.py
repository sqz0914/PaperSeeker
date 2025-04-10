from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from ..models import SearchResponse
import logging

# Setup logging
logger = logging.getLogger("search_routes")

router = APIRouter(
    prefix="/search",
    tags=["search"],
)

# Helper function to get data manager and RAG engine
def get_dependencies():
    from ..app import data_manager, rag_engine
    return data_manager, rag_engine

@router.get("/", response_model=Dict[str, Any])
async def search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return"),
    augment: bool = Query(False, description="Whether to augment results with LLM-generated content")
):
    """
    Search for papers matching the given query.
    
    Args:
        query: Search query string
        top_k: Number of results to return (1-10)
        augment: Whether to augment results with LLM-generated content
        
    Returns:
        List of papers matching the query or augmented response
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        # Check if we're using the data manager or direct RAG engine
        if data_manager and data_manager.use_real_data:
            if augment:
                # Use the RAG search and augment
                papers, augmented_content = data_manager.rag_engine.search_and_augment(query, top_k)
                return {"papers": papers, "augmented_content": augmented_content}
            else:
                # Use regular search
                papers = data_manager.rag_engine.search_papers(query, top_k)
                return {"papers": papers}
        elif rag_engine:
            # Direct use of RAG engine
            results = rag_engine.search(query, top_k=top_k, augment=augment)
            return results
        else:
            raise HTTPException(status_code=500, detail="Search engine not initialized")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics", response_model=List[str])
def get_topics():
    """Return a list of all available paper topics"""
    data_manager, _ = get_dependencies()
    
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    return data_manager.get_all_topics() 