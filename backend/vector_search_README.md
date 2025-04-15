# Vector Search Implementation for PaperSeeker

This implementation adds semantic vector search capabilities to PaperSeeker using Llama 3.2 (via Ollama) and the Milvus vector database.

## Prerequisites

1. **Ollama** with Llama 3.2 installed
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the Llama 3 model: `ollama pull llama3`

2. **Milvus** vector database
   - Install via Docker:
     ```bash
     docker run -d --name milvus_standalone \
       -p 19530:19530 \
       -p 9091:9091 \
       -v /path/to/your/milvus/data:/var/lib/milvus/data \
       -v /path/to/your/milvus/conf:/var/lib/milvus/conf \
       -v /path/to/your/milvus/logs:/var/lib/milvus/logs \
       milvusdb/milvus:v2.3.5
     ```

3. **Python Dependencies**
   - Install the required packages:
     ```bash
     pip install -r vector_requirements.txt
     ```

## Implementation Files

1. **generate_embeddings.py**
   - Script to generate embeddings from papers using Llama 3.2 and store them in Milvus
   - Usage: `python generate_embeddings.py`

2. **vector_search.py**
   - Contains the `VectorSearch` class for querying Milvus with vector similarity
   - Can be used standalone: `python vector_search.py "your query here"`

3. **app.py** (updated)
   - Integrates vector search into the FastAPI backend
   - Falls back to text-based search if vector search is unavailable

## Setup and Usage

1. **Step 1: Start required services**
   ```bash
   # Start Ollama (if not running)
   ollama serve
   
   # Start Milvus (if not running via Docker)
   docker start milvus_standalone
   ```

2. **Step 2: Generate embeddings and populate Milvus**
   ```bash
   python generate_embeddings.py
   ```

3. **Step 3: Start the API server**
   ```bash
   python app.py
   ```

4. **Step 4: Test the vector search**
   - Access the API at `http://localhost:8000/docs`
   - Try searching for papers with `/search` endpoint

## Vector Search Architecture

1. **Document Preparation**
   - Papers are parsed from JSON format
   - Title, abstract, and full text are combined with emphasis on title and abstract

2. **Embedding Generation**
   - Ollama API is used to generate embeddings with Llama 3.2
   - 4096-dimensional vectors represent each paper

3. **Vector Database**
   - Milvus stores paper data and embeddings
   - Papers are indexed using HNSW algorithm for efficient retrieval
   - Cosine similarity is used for ranking search results

4. **API Integration**
   - Vector search is the primary search method
   - Text-based search is used as fallback
   - Both streaming and regular API endpoints support vector search

## Troubleshooting

- **Ollama Connection Issues**
  - Ensure Ollama is running: `ollama serve`
  - Verify the Llama 3 model is installed: `ollama list`

- **Milvus Connection Issues**
  - Check if Milvus container is running: `docker ps`
  - Logs can be viewed with: `docker logs milvus_standalone`

- **Paper Parsing Issues**
  - Check logs for errors in paper parsing
  - Verify sample_papers.json file is correctly formatted 