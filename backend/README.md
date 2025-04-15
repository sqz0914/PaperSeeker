# PaperSeeker Backend

A FastAPI backend for semantically searching academic papers using Zilliz Cloud and Ollama.

## Overview

PaperSeeker uses vector search powered by Zilliz Cloud to find papers semantically similar to your query. The system:

1. Generates embeddings for papers using Ollama (with Llama 3.2)
2. Stores these embeddings in Zilliz Cloud for efficient similarity search
3. Provides a FastAPI interface for searching and retrieving papers

## Setup

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure you have [Ollama](https://ollama.ai/) installed and running with the Llama 3.2 model:

```bash
ollama pull llama3
```

### Zilliz Cloud Setup

1. Create a Zilliz Cloud account at [cloud.zilliz.com](https://cloud.zilliz.com/) if you don't have one
2. Create a new cluster (or use an existing one)
3. Copy the `.env.template` file to `.env`:
   ```bash
   cp .env.template .env
   ```
4. Update the `.env` file with your Zilliz Cloud cluster information:
   - `ZILLIZ_CLOUD_URI`: The URI of your Zilliz Cloud cluster (e.g., https://cluster-name.zillizcloud.com:19530)
   - `ZILLIZ_CLOUD_TOKEN`: Your Zilliz Cloud API key

### Data Preparation

Place your paper data in JSONL format (one JSON paper per line) in the `sample_papers.json` file. Each paper should follow this format:

```json
{
  "id": "unique_id",
  "source": "paper_source",
  "version": "version_info",
  "added": "timestamp",
  "created": "timestamp",
  "text": "full_paper_text",
  "metadata": {
    "year": 2023,
    "title": "Paper Title",
    "abstract": "Paper abstract text...",
    "sha1": "hash_value",
    "sources": ["source1", "source2"],
    "s2fieldsofstudy": ["field1", "field2"],
    "extfieldsofstudy": ["field3"],
    "external_ids": [
      {"source": "DOI", "id": "10.xxxx/yyyy"},
      {"source": "MAG", "id": "12345678"}
    ]
  }
}
```

### Generate Embeddings

Run the embedding generation script to process papers and create the vector database in Zilliz Cloud:

```bash
python generate_embeddings.py
```

This script:
- Processes the papers in batches to handle large files
- Generates embeddings using Ollama
- Stores papers and embeddings in Zilliz Cloud

## Running the API

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /`: API information
- `POST /search`: Search for papers using a query
- `GET /papers`: Get all papers (paginated)
- `GET /papers/{paper_id}`: Get a specific paper by ID
- `GET /api/topics`: Get all available research topics
- `POST /api/search`: Simple search API for chat interface
- `POST /api/stream`: Stream search results with typing effect
- `GET /api/welcome`: Welcome message for chat interface

## Implementation Details

### Zilliz Cloud

This implementation uses Zilliz Cloud, a fully managed vector database service based on Milvus, which:
- Requires no infrastructure management
- Provides high availability and scalability
- Offers excellent performance for vector similarity search
- Has a simple API for integrating into applications

### Batch Processing

Large paper collections are processed in batches to manage memory usage efficiently.

### Embedding Generation

Each paper's text is weighted (title appears multiple times) to improve search relevance, then processed with Llama 3.2 through Ollama API.

### Search Fallback

If vector search fails, the system falls back to text-based search for reliability. 