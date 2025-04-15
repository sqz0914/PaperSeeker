from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import os
import logging
import asyncio
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from models import Paper, SearchRequest, QueryRequest
from vector_search import VectorSearch

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

# Welcome message for the chatbot
WELCOME_MESSAGE = "Hello! I'm PaperSeeker. Ask me about research papers using vector similarity search powered by Llama 3.2."

# Initialize vector search
vector_search = None
try:
    vector_search = VectorSearch()
    logger.info("Vector search engine initialized successfully")
except ValueError as e:
    if "Collection 'paper_collection' not found" in str(e):
        logger.warning("Zilliz Cloud collection not found. Please run generate_embeddings.py first to create the vector database.")
    elif "ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set" in str(e):
        logger.error("Zilliz Cloud credentials not configured. Please update the .env file with your Zilliz Cloud information.")
    else:
        logger.error(f"Failed to initialize vector search engine: {e}")
except Exception as e:
    logger.error(f"Failed to initialize vector search engine: {e}")

# Load papers from sample_papers.json as fallback
def load_papers():
    """Load papers from JSONL or JSON file as fallback when vector search is unavailable."""
    try:
        papers = []
        papers_path = os.path.join(os.path.dirname(__file__), "sample_papers.json")
        
        with open(papers_path, 'r', encoding='utf-8') as f:
            # Try to detect if it's a JSONL file (each line is a JSON object)
            first_line = f.readline().strip()
            # Reset file pointer to beginning
            f.seek(0)
            
            if first_line.startswith('{') and first_line.endswith('}'):
                # Likely JSONL format - each line is a separate JSON object
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            paper = json.loads(line.strip())
                            papers.append(paper)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing paper JSON: {e}")
            else:
                # Try standard JSON format (array of objects)
                try:
                    papers = json.load(f)
                except json.JSONDecodeError:
                    # If that fails, try parsing line by line
                    f.seek(0)
                    for line in f:
                        if line.strip():  # Skip empty lines
                            try:
                                paper = json.loads(line.strip())
                                papers.append(paper)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing JSON line: {e}")
        
        logger.info(f"Loaded {len(papers)} papers from sample_papers.json")
        return papers
    except Exception as e:
        logger.error(f"Error loading papers: {e}")
        return []

# Initialize papers as fallback
papers = load_papers()
vector_search_status = "available" if vector_search else "unavailable"
if not vector_search:
    if papers:
        logger.info(f"Vector search unavailable, using text-based search with {len(papers)} loaded papers")
    else:
        logger.warning("Neither vector search nor fallback papers are available. API will have limited functionality.")

@app.get("/")
def read_root():
    """Root endpoint providing basic information about the API"""
    return {
        "message": "Welcome to PaperSeeker API",
        "papers_loaded": len(papers),
        "vector_search": vector_search_status
    }

@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for papers matching the query using vector similarity.
    Returns papers where the query matches semantically using embeddings.
    """
    query = request.query
    top_k = request.top_k if hasattr(request, 'top_k') else 5
    
    # Use vector search if available
    if vector_search:
        try:
            matched_papers = vector_search.search(query, limit=top_k)
            logger.info(f"Vector search found {len(matched_papers)} papers matching query: '{query}'")
            return {"papers": matched_papers}
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.info("Falling back to text-based search")
    
    # Fallback to text-based search
    matched_papers = []
    for paper in papers:
        # Search in title
        title = paper.get("metadata", {}).get("title", "").lower()
        # Search in abstract
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        # Search in full text
        text = paper.get("text", "").lower()
        
        # If query is found in any of these fields, add the paper to results
        if query.lower() in title or query.lower() in abstract or query.lower() in text:
            matched_papers.append(paper)
            if len(matched_papers) >= top_k:
                break
    
    logger.info(f"Text search found {len(matched_papers)} papers matching query: '{query}'")
    return {"papers": matched_papers}

@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get a specific paper by its ID, parsed using the Paper model"""
    for paper_data in papers:
        if paper_data.get("id") == paper_id:
            try:
                # Parse the paper using our Paper model
                paper = Paper.parse_data(paper_data)
                # Convert to dict for JSON response
                return paper.dict(exclude_none=True)
            except Exception as e:
                logger.error(f"Error parsing paper: {e}")
                return {"error": "Failed to parse paper"}
    
    return {"error": "Paper not found"}

@app.get("/papers")
async def get_papers():
    """Get all available papers"""
    return {"papers": papers}

@app.get("/api/topics")
async def get_topics():
    """Get all available topics/fields of study from the papers"""
    topics = set()
    
    for paper in papers:
        # Get fields of study from metadata
        s2fields = paper.get("metadata", {}).get("s2fieldsofstudy", [])
        extfields = paper.get("metadata", {}).get("extfieldsofstudy", [])
        
        # Add all fields to our set of topics
        for field in s2fields:
            topics.add(field)
        for field in extfields:
            topics.add(field)
    
    return {"topics": sorted(list(topics))}

# Helper function for streaming text with typing effect
async def stream_text(text: str, citation: str = None):
    """Helper function to stream any text with typing effect"""
    # First yield the type of message
    yield json.dumps({"type": "start"}) + "\n"
    
    # Stream the response character by character
    buffer = ""
    for char in text:
        buffer += char
        # Send in small chunks to simulate typing
        if len(buffer) >= 3 or char in ['.', ',', '!', '?', ' ']:
            # Random delay to simulate realistic typing
            await asyncio.sleep(random.uniform(0.01, 0.05))
            yield json.dumps({"type": "chunk", "content": buffer}) + "\n"
            buffer = ""
    
    # Send any remaining characters
    if buffer:
        yield json.dumps({"type": "chunk", "content": buffer}) + "\n"
    
    # Send citation separately
    if citation:
        await asyncio.sleep(0.5)  # Pause before citation
        yield json.dumps({"type": "citation", "content": citation}) + "\n"
    
    # Send end message
    yield json.dumps({"type": "end"}) + "\n"

# Stream the response for a query
async def stream_response(query: str):
    """Stream a response character by character with realistic typing delays"""
    if vector_search:
        try:
            # Try to use vector search first
            matched_papers = vector_search.search(query, limit=1)
            if matched_papers:
                paper = matched_papers[0]
                title = paper.get("metadata", {}).get("title", "")
                abstract = paper.get("metadata", {}).get("abstract", "")
                year = paper.get("metadata", {}).get("year", "")
                score = paper.get("similarity_score", 0)
                
                # Get citation from external IDs
                citation = None
                if paper.get("metadata", {}).get("external_ids"):
                    ext_ids = []
                    for ext_id in paper["metadata"]["external_ids"]:
                        if ext_id.get("source") and ext_id.get("id"):
                            ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                    
                    if ext_ids:
                        citation = ", ".join(ext_ids)
                
                # Generate a more informative response using the paper data
                response = f"I found a paper that might be relevant (similarity score: {score:.2f}): '{title}' ({year}). {abstract[:300]}..."
                
                async for chunk in stream_text(response, citation):
                    yield chunk
                return
        except Exception as e:
            logger.error(f"Vector search failed in streaming: {e}")
            # Fall back to text search
    
    # Text-based search fallback
    for paper in papers:
        title = paper.get("metadata", {}).get("title", "").lower()
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        text = paper.get("text", "").lower()
        
        if query.lower() in title or query.lower() in abstract or query.lower() in text:
            # Format the response in a way the frontend expects
            citation = None
            if paper.get("metadata", {}).get("external_ids"):
                ext_ids = []
                for ext_id in paper["metadata"]["external_ids"]:
                    if ext_id.get("source") and ext_id.get("id"):
                        ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                
                if ext_ids:
                    citation = ", ".join(ext_ids)
            
            title = paper.get("metadata", {}).get("title", "")
            abstract = paper.get("metadata", {}).get("abstract", "")
            year = paper.get("metadata", {}).get("year", "")
            
            response = f"I found a paper that might be relevant: '{title}' ({year}). {abstract[:300]}..."
            
            async for chunk in stream_text(response, citation):
                yield chunk
            return
    
    # Default response if no match found
    response = "I couldn't find any papers matching your query. Could you try a different search term?"
    async for chunk in stream_text(response):
        yield chunk

@app.post("/api/search")
def api_search_papers(request: QueryRequest):
    """Simple search API for the chat interface"""
    query = request.query
    
    if vector_search:
        try:
            # Try vector search first
            matched_papers = vector_search.search(query, limit=1)
            if matched_papers:
                paper = matched_papers[0]
                title = paper.get("metadata", {}).get("title", "")
                abstract = paper.get("metadata", {}).get("abstract", "")
                year = paper.get("metadata", {}).get("year", "")
                score = paper.get("similarity_score", 0)
                
                # Get citation from external IDs
                citation = None
                if paper.get("metadata", {}).get("external_ids"):
                    ext_ids = []
                    for ext_id in paper["metadata"]["external_ids"]:
                        if ext_id.get("source") and ext_id.get("id"):
                            ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                    
                    if ext_ids:
                        citation = ", ".join(ext_ids)
                
                return {
                    "response": f"I found a paper that might be relevant (similarity score: {score:.2f}): '{title}' ({year}). {abstract[:300]}...",
                    "citation": citation
                }
        except Exception as e:
            logger.error(f"Vector search failed in API search: {e}")
            # Fall back to text search
    
    # Text-based search as fallback
    for paper in papers:
        title = paper.get("metadata", {}).get("title", "").lower()
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        text = paper.get("text", "").lower()
        
        if query.lower() in title or query.lower() in abstract or query.lower() in text:
            # Format the response in a way the frontend expects
            citation = None
            if paper.get("metadata", {}).get("external_ids"):
                ext_ids = []
                for ext_id in paper["metadata"]["external_ids"]:
                    if ext_id.get("source") and ext_id.get("id"):
                        ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                
                if ext_ids:
                    citation = ", ".join(ext_ids)
            
            title = paper.get("metadata", {}).get("title", "")
            abstract = paper.get("metadata", {}).get("abstract", "")
            year = paper.get("metadata", {}).get("year", "")
            
            return {
                "response": f"I found a paper that might be relevant: '{title}' ({year}). {abstract[:300]}...",
                "citation": citation
            }
    
    # Default response if no match found
    return {
        "response": "I couldn't find any papers matching your query. Could you try a different search term?",
        "citation": None
    }

@app.post("/api/stream")
async def stream_search(request: QueryRequest):
    """Endpoint that streams the response"""
    return StreamingResponse(
        stream_response(request.query),
        media_type="text/event-stream"
    )

@app.get("/api/welcome")
async def welcome_message():
    """Stream the welcome message"""
    return StreamingResponse(
        stream_text(WELCOME_MESSAGE),
        media_type="text/event-stream"
    )

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    if vector_search:
        try:
            vector_search.close()
            logger.info("Vector search resources released")
        except Exception as e:
            logger.error(f"Error closing vector search: {e}")

# Run the API server when this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 