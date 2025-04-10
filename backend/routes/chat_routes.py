from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Optional, Any
import json
import asyncio
import random
import logging
from ..models import QueryRequest

# Setup logging
logger = logging.getLogger("chat_routes")

router = APIRouter(
    prefix="/api",
    tags=["chat"],
)

# Welcome message for the chatbot
WELCOME_MESSAGE = "Hello! I'm PaperSeeker. Ask me about research papers, and I'll help you find relevant information."

# Helper function to get data manager and RAG engine
def get_dependencies():
    from ..app import data_manager
    return data_manager

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
    data_manager = get_dependencies()
    
    if not data_manager:
        return stream_text("System error: Data manager not initialized.")
    
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

@router.post("/stream")
async def stream_search(request: QueryRequest):
    """Endpoint that streams the response"""
    return StreamingResponse(
        stream_response(request.query),
        media_type="text/event-stream"
    )

@router.get("/welcome")
async def welcome_message():
    """Stream the welcome message"""
    return StreamingResponse(
        stream_text(WELCOME_MESSAGE),
        media_type="text/event-stream"
    )

@router.post("/search")
def api_search_papers(request: QueryRequest):
    """Simple search API for the chat interface"""
    data_manager = get_dependencies()
    
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    
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