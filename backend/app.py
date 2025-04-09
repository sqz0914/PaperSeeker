from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import asyncio
import json
import time
import random
import os
from data import DataManager
from models import QueryRequest, QueryResponse, PaperData

app = FastAPI(title="PaperSeeker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the data manager
data_manager = DataManager()

# Welcome message for the chatbot
WELCOME_MESSAGE = "Hello! I'm PaperSeeker. Ask me about research papers, and I'll help you find relevant information."

@app.get("/")
def read_root():
    return {"message": "Welcome to PaperSeeker API"}

@app.post("/api/search")
def search_papers(request: QueryRequest):
    query = request.query.lower()
    
    # Search for a matching paper
    paper = data_manager.search_papers(query)
    
    if paper:
        return {
            "response": paper["response"],
            "citation": paper.get("citation")
        }
    
    # Default response if no match found
    available_topics = ", ".join(data_manager.get_all_topics())
    return {
        "response": f"I don't have specific information about that yet. Try asking about {available_topics}.",
        "citation": None
    }

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

async def stream_response(query: str):
    """Stream a response character by character with realistic typing delays"""
    # Find a matching paper
    paper = data_manager.search_papers(query)
    
    # Use default response if no match
    if paper is None:
        # Get a list of available topics
        available_topics = ", ".join(data_manager.get_all_topics())
        response = f"I don't have specific information about that yet. Try asking about {available_topics}."
        citation = None
    else:
        response = paper["response"]
        citation = paper.get("citation")
    
    # Use the shared streaming function
    async for chunk in stream_text(response, citation):
        yield chunk

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

# API endpoint to get all available paper topics
@app.get("/api/topics", response_model=List[str])
def get_topics():
    """Return a list of all available paper topics"""
    return data_manager.get_all_topics()

# Endpoints for managing papers (for future admin interface)
@app.get("/api/papers", response_model=List[Dict[str, Any]])
def get_all_papers():
    """Get all papers"""
    return data_manager.get_all_papers()

@app.post("/api/papers", status_code=201)
def add_paper(paper: PaperData):
    """Add a new paper"""
    success = data_manager.add_paper(paper.model_dump())
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add paper. It may already exist.")
    return {"success": True, "message": "Paper added successfully"}

@app.put("/api/papers/{query}")
def update_paper(query: str, paper: PaperData):
    """Update an existing paper"""
    success = data_manager.update_paper(query, paper.model_dump())
    if not success:
        raise HTTPException(status_code=404, detail=f"Paper with query '{query}' not found")
    return {"success": True, "message": "Paper updated successfully"}

@app.delete("/api/papers/{query}")
def delete_paper(query: str):
    """Delete a paper"""
    success = data_manager.delete_paper(query)
    if not success:
        raise HTTPException(status_code=404, detail=f"Paper with query '{query}' not found")
    return {"success": True, "message": "Paper deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 