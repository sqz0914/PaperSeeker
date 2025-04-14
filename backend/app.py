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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PaperSeeker API", 
    description="Search and explore academic papers",
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
WELCOME_MESSAGE = "Hello! I'm PaperSeeker. Ask me about research papers, and I'll help you find relevant information."

# Load papers from sample_papers.json
def load_papers():
    try:
        papers_path = os.path.join(os.path.dirname(__file__), "sample_papers.json")
        with open(papers_path, 'r') as file:
            raw_papers = json.load(file)
            
        # Parse papers using the Paper model
        parsed_papers = []
        for paper_data in raw_papers:
            try:
                paper = Paper.parse_data(paper_data)
                parsed_papers.append(paper_data)  # Keep original data for search
                logger.debug(f"Parsed paper: {paper.title}")
            except Exception as e:
                logger.error(f"Error parsing paper: {e}")
                
        logger.info(f"Loaded {len(parsed_papers)} papers from sample_papers.json")
        return raw_papers  # Return raw papers for now to maintain compatibility
    except Exception as e:
        logger.error(f"Error loading papers: {e}")
        return []

# Initialize papers
papers = load_papers()

@app.get("/")
def read_root():
    """Root endpoint providing basic information about the API"""
    return {
        "message": "Welcome to PaperSeeker API",
        "papers_loaded": len(papers)
    }

@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for papers matching the query.
    Returns papers where the query matches part of the title, abstract, or full text.
    """
    query = request.query.lower()
    matched_papers = []
    
    for paper in papers:
        # Search in title
        title = paper.get("metadata", {}).get("title", "").lower()
        # Search in abstract
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        # Search in full text
        text = paper.get("text", "").lower()
        
        # If query is found in any of these fields, add the paper to results
        if query in title or query in abstract or query in text:
            matched_papers.append(paper)
    
    logger.info(f"Found {len(matched_papers)} papers matching query: '{query}'")
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
    query = query.lower()
    
    # Search for a matching paper
    for paper in papers:
        title = paper.get("metadata", {}).get("title", "").lower()
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        text = paper.get("text", "").lower()
        
        if query in title or query in abstract or query in text:
            # Format the response in a way the frontend expects
            citation = None
            
            # Add external IDs
            if paper.get("metadata", {}).get("external_ids"):
                # Get all external IDs
                ext_ids = []
                for ext_id in paper["metadata"]["external_ids"]:
                    if ext_id.get("source") and ext_id.get("id"):
                        ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                
                # Join them with commas
                if ext_ids:
                    citation = ", ".join(ext_ids)
            
            title = paper.get("metadata", {}).get("title", "")
            abstract = paper.get("metadata", {}).get("abstract", "")
            year = paper.get("metadata", {}).get("year", "")
            
            # Generate a more informative response using the paper data
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
    query = request.query.lower()
    
    # Search for a matching paper
    for paper in papers:
        title = paper.get("metadata", {}).get("title", "").lower()
        abstract = paper.get("metadata", {}).get("abstract", "").lower()
        text = paper.get("text", "").lower()
        
        if query in title or query in abstract or query in text:
            # Format the response in a way the frontend expects
            citation = None
            
            # Add external IDs
            if paper.get("metadata", {}).get("external_ids"):
                # Get all external IDs
                ext_ids = []
                for ext_id in paper["metadata"]["external_ids"]:
                    if ext_id.get("source") and ext_id.get("id"):
                        ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                
                # Join them with commas
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

# Run the API server when this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 