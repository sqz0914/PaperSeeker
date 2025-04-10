from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
from typing import Optional

# Import route modules
from routes import paper_routes, search_routes, chat_routes, config_routes
from rag.rag_engine import RAGEngine
from data.data_manager import DataManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PaperSeeker API", 
    description="Search and explore academic papers with RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the data manager and RAG engine
data_manager: Optional[DataManager] = None
rag_engine: Optional[RAGEngine] = None

try:
    # Always use RAG and Milvus since that's our production approach
    data_manager = DataManager(use_rag=True, use_milvus=True)
    rag_engine = data_manager.rag_engine
    logger.info("Data manager and RAG engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize data manager or RAG engine: {e}")

# Include all routers
app.include_router(paper_routes.router)
app.include_router(search_routes.router)
app.include_router(chat_routes.router)
app.include_router(config_routes.router)

@app.get("/")
def read_root():
    """Root endpoint providing basic information about the API"""
    return {
        "message": "Welcome to PaperSeeker API", 
        "using_real_data": data_manager.use_real_data if data_manager else False,
        "using_rag": True
    }

# Run the API server when this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 