from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from typing import Dict, Any
from pydantic import BaseModel
from models import SearchRequest
from vector_search import VectorSearch
from llm_processor import LLMProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PaperSeeker API", 
    description="Search and explore academic papers with vector search",
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

# Initialize vector search
vector_search = None
llm_processor = None
try:
    vector_search = VectorSearch()
    logger.info("Vector search engine initialized successfully")
    llm_processor = LLMProcessor()
    logger.info("LLM processor initialized successfully")
except ValueError as e:
    if "Collection 'paper_collection' not found" in str(e):
        logger.warning("Zilliz Cloud collection not found. Please run generate_embeddings.py first to create the vector database.")
    elif "ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set" in str(e):
        logger.error("Zilliz Cloud credentials not configured. Please update the .env file with your Zilliz Cloud information.")
    else:
        logger.error(f"Failed to initialize vector search engine: {e}")
except Exception as e:
    logger.error(f"Failed to initialize vector search engine: {e}")

vector_search_status = "available" if vector_search else "unavailable"

@app.get("/")
def read_root():
    """Root endpoint providing basic information about the API"""
    return {
        "message": "Welcome to PaperSeeker API",
        "vector_search": vector_search_status
    }

@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for papers matching the query using vector similarity.
    Returns papers where the query matches semantically using embeddings.
    """
    query = request.query
    top_k = request.top_k if hasattr(request, 'top_k') else 10  # Default to 10 papers
    use_llm = request.use_llm if hasattr(request, 'use_llm') else False
    
    # Use vector search if available
    if vector_search:
        try:
            # Get more results if we're using LLM reranking
            limit = 30 if use_llm and llm_processor else top_k  # Increased from 20 to 30 for better diversity
            
            matched_papers = vector_search.search(query, limit=limit)
            logger.info(f"Vector search found {len(matched_papers)} papers matching query: '{query}'")
            
            # Filter out papers without essential data
            matched_papers = [p for p in matched_papers if p.get("metadata", {}).get("title")]
            
            if not matched_papers:
                return {
                    "introduction": "",
                    "papers": [],
                    "conclusion": f"I couldn't find any papers matching your query '{query}'."
                }
            
            # If LLM reranking is requested and available
            if use_llm and llm_processor and matched_papers:
                try:
                    logger.info("Starting LLM reranking and response generation")
                    structured_response = llm_processor.rerank_and_generate(
                        query=query,
                        papers=matched_papers,
                        max_papers_for_rerank=limit,
                        max_papers_for_response=top_k  # Return top 10 papers after reranking
                    )
                    
                    # Log the raw response for debugging
                    logger.info(f"LLM structured response format: {structured_response.keys()}")
                    if "papers" in structured_response:
                        logger.info(f"Number of papers in response: {len(structured_response['papers'])}")
                    
                    # Format the response in the exact structure requested
                    introduction = structured_response.get("introduction", "")
                    conclusion = structured_response.get("conclusion", "")
                    raw_papers = structured_response.get("papers", [])
                    
                    # Format each paper with only the required fields
                    simplified_papers = []
                    seen_titles = set()  # Track titles to prevent duplicates
                    
                    for paper_info in raw_papers:
                        # Skip papers without essential data
                        if not paper_info.get("title") or not paper_info.get("paper_data"):
                            logger.warning(f"Skipping paper with missing title or data")
                            continue
                            
                        paper_data = paper_info.get("paper_data", {})
                        metadata = paper_data.get("metadata", {})
                        
                        # Create simplified paper object with clear indicators for missing data
                        title = paper_info.get("title", "").strip()
                        
                        # Skip papers with missing title or duplicates
                        if not title or title.lower() in seen_titles:
                            continue
                            
                        # Add title to seen set to prevent duplicates
                        seen_titles.add(title.lower())
                        
                        abstract = metadata.get("abstract", "").strip() if metadata.get("abstract") else "[Abstract not available]"
                        summary = paper_info.get("summary", "").strip()
                        year = metadata.get("year", "") or paper_data.get("year", "")
                        
                        # Extract citation information
                        sources = ""
                        if metadata.get("external_ids"):
                            ext_ids = []
                            for ext_id in metadata["external_ids"]:
                                if ext_id.get("source") and ext_id.get("id"):
                                    ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                            if ext_ids:
                                sources = ", ".join(ext_ids)
                        
                        simplified_paper = {
                            "title": title,
                            "abstract": abstract,
                            "summary": summary if summary else "[Summary not available]",
                            "year": year if year else "[Year not available]",
                            "sources": sources if sources else "[Sources not available]"
                        }
                        simplified_papers.append(simplified_paper)
                    
                    # If we have no valid papers, return an appropriate message
                    if not simplified_papers:
                        return {
                            "introduction": "",
                            "papers": [],
                            "conclusion": f"I couldn't find any papers matching your query '{query}'."
                        }
                    
                    # Return the simplified structured response
                    return {
                        "introduction": "",  # Empty introduction, focus on papers
                        "papers": simplified_papers,
                        "conclusion": conclusion if conclusion else f"Found {len(simplified_papers)} papers related to your query about '{query}'."
                    }
                except Exception as e:
                    logger.error(f"LLM reranking failed: {e}")
                    # Fall back to standard results but format them in the required structure
            
            # Format standard results in the required structure for consistency
            simplified_papers = []
            seen_titles = set()  # Track titles to prevent duplicates
            
            for paper in matched_papers[:top_k]:
                metadata = paper.get("metadata", {})
                
                # Get title and skip if missing or duplicate
                title = metadata.get("title", "").strip()
                if not title or title.lower() in seen_titles:
                    continue
                    
                # Add to seen titles to prevent duplicates
                seen_titles.add(title.lower())
                
                sources = ""
                if metadata.get("external_ids"):
                    ext_ids = []
                    for ext_id in metadata["external_ids"]:
                        if ext_id.get("source") and ext_id.get("id"):
                            ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                    if ext_ids:
                        sources = ", ".join(ext_ids)
                
                simplified_paper = {
                    "title": title,
                    "abstract": metadata.get("abstract", "").strip() if metadata.get("abstract") else "[Abstract not available]",
                    "summary": f"This paper appears relevant to your query about '{query}'.",
                    "year": metadata.get("year") if metadata.get("year") else "[Year not available]",
                    "sources": sources if sources else "[Sources not available]"
                }
                simplified_papers.append(simplified_paper)
            
            if not simplified_papers:
                return {
                    "introduction": "",
                    "papers": [],
                    "conclusion": f"I couldn't find any papers matching your query '{query}'."
                }
            
            return {
                "introduction": "",
                "papers": simplified_papers,
                "conclusion": f"Found {len(simplified_papers)} papers related to your query about '{query}'."
            }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.info("No search results found")
    
    # Return empty results if vector search is not available
    return {
        "introduction": "",
        "papers": [],
        "conclusion": f"I couldn't find any papers matching your query '{query}'. Vector search is not available."
    }

# Run the API server when this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 