from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

# Setup logging
logger = logging.getLogger("config_routes")

router = APIRouter(
    prefix="/api",
    tags=["configuration"],
)

# Helper function to get data manager and RAG engine
def get_dependencies():
    from ..app import data_manager
    return data_manager

@router.get("/data-source")
def get_data_source():
    """Return information about the data source being used"""
    data_manager = get_dependencies()
    
    if not data_manager:
        return {
            "using_real_data": False,
            "using_rag": True,
            "source": "not initialized"
        }
    
    return {
        "using_real_data": data_manager.use_real_data,
        "using_rag": True,
        "source": "real paper database" if data_manager.use_real_data else "mock data"
    }

@router.get("/config")
def get_config():
    """Get the current configuration of the search system"""
    data_manager = get_dependencies()
    
    if not data_manager:
        return {
            "use_real_data": False,
            "use_rag": True,
            "use_milvus": True,
            "embedding_model": None,
            "llm_model": None
        }
    
    return {
        "use_real_data": data_manager.use_real_data,
        "use_rag": True,
        "use_milvus": True,
        "embedding_model": data_manager.rag_engine.embedding_model_name,
        "llm_model": data_manager.rag_engine.llm_model_name
    } 