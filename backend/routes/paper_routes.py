from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from ..models import Paper, AddPaperRequest
from ..rag.rag_engine import RAGEngine
from ..data.data_manager import DataManager
import logging

# Setup logging
logger = logging.getLogger("paper_routes")

router = APIRouter(
    prefix="/papers",
    tags=["papers"],
    responses={404: {"description": "Paper not found"}}
)

# Helper function to get data manager and RAG engine
def get_dependencies():
    from ..app import data_manager, rag_engine
    return data_manager, rag_engine

@router.get("/", response_model=Dict[str, List[Dict[str, Any]]])
async def get_papers():
    """
    Get all papers in the database.
    
    Returns:
        List of all papers
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        if data_manager:
            papers = data_manager.get_all_papers()
        elif rag_engine:
            papers = rag_engine.get_all_papers()
        else:
            raise HTTPException(status_code=500, detail="Data manager not initialized")
        
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error retrieving papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{paper_id}")
async def get_paper(paper_id: str):
    """
    Get a specific paper by its ID.
    
    Args:
        paper_id: ID of the paper to retrieve
        
    Returns:
        Paper details
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        paper = None
        if data_manager:
            # Try to get paper from data manager
            paper = data_manager.get_paper_by_id(paper_id) if hasattr(data_manager, 'get_paper_by_id') else None
        
        if paper is None and rag_engine:
            # Try to get paper from RAG engine
            paper = rag_engine.get_paper_by_id(paper_id)
        
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        return paper
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", status_code=201)
async def add_paper(paper: AddPaperRequest):
    """
    Add a new paper to the database.
    
    Args:
        paper: Paper details to add
        
    Returns:
        Success message
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        paper_dict = paper.dict()
        success = False
        
        if data_manager:
            # Try to add through data manager
            success = data_manager.add_paper(paper_dict)
        elif rag_engine:
            # Add directly through RAG engine
            rag_engine.add_paper(paper_dict)
            success = True
        else:
            raise HTTPException(status_code=500, detail="Data manager not initialized")
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to add paper. It may already exist.")
            
        return {"message": "Paper added successfully", "paper_id": paper_dict.get("id")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{paper_id}")
async def update_paper(paper_id: str, paper: AddPaperRequest):
    """
    Update an existing paper.
    
    Args:
        paper_id: ID of the paper to update
        paper: Updated paper details
        
    Returns:
        Success message
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        paper_dict = paper.dict()
        success = False
        
        if data_manager and hasattr(data_manager, 'update_paper'):
            # Try to update through data manager
            success = data_manager.update_paper(paper_id, paper_dict)
        elif rag_engine:
            # Update directly through RAG engine
            success = rag_engine.update_paper(paper_id, paper_dict)
        else:
            raise HTTPException(status_code=500, detail="Data manager not initialized")
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        return {"message": "Paper updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{paper_id}")
async def delete_paper(paper_id: str):
    """
    Delete a paper by its ID.
    
    Args:
        paper_id: ID of the paper to delete
        
    Returns:
        Success message
    """
    data_manager, rag_engine = get_dependencies()
    
    try:
        success = False
        
        if data_manager and hasattr(data_manager, 'delete_paper'):
            # Try to delete through data manager
            success = data_manager.delete_paper(paper_id)
        elif rag_engine:
            # Delete directly through RAG engine
            success = rag_engine.delete_paper(paper_id)
        else:
            raise HTTPException(status_code=500, detail="Data manager not initialized")
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        return {"message": "Paper deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
def get_topics():
    """Get all paper topics"""
    data_manager, _ = get_dependencies()
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    return {"topics": data_manager.get_all_topics()}

@router.get("/search")
def search_papers(query: str = Query(..., description="Search query")):
    """Search for papers matching the query"""
    data_manager, _ = get_dependencies()
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = data_manager.search_papers(query)
    if result:
        return result
    
    return JSONResponse(
        status_code=404,
        content={"message": f"No papers found for query: {query}"}
    )

@router.get("/config")
def get_config():
    """Get the current configuration of the search system"""
    data_manager, _ = get_dependencies()
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
        
    return {
        "use_real_data": data_manager.use_real_data,
        "use_rag": True,
        "use_milvus": True,
        "embedding_model": data_manager.rag_engine.embedding_model_name,
        "llm_model": data_manager.rag_engine.llm_model_name
    } 