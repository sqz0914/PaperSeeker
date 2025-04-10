# PaperSeeker

A RAG-powered academic paper search engine with FastAPI backend.

## Features

- Search for academic papers using semantic similarity
- Retrieve papers based on text queries
- Augment search results with generative AI content
- Add new papers to the collection
- Modern FastAPI backend with easy-to-use JSON API

## Project Structure

```
PaperSeeker/
├── backend/
│   ├── app.py                     # FastAPI application
│   ├── models.py                  # Pydantic models
│   ├── run.py                     # Server startup script
│   ├── data/
│   │   ├── __init__.py            # Package initialization
│   │   ├── data_manager.py        # Data manager integration layer
│   │   ├── sample_papers.json     # Sample paper data
│   │   └── mock_data.json         # Mock data for testing
│   ├── rag_algorithm/             # RAG implementation directory
│   │   ├── __init__.py            # Package initialization
│   │   ├── ir_engine.py           # Traditional IR engine
│   │   └── rag_engine.py          # RAG engine implementation
├── client.py                      # Simple command-line client
├── run_paperseeker.py             # Project startup script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PaperSeeker.git
   cd PaperSeeker
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

There are two ways to run the application:

#### 1. Using the startup script:
   ```
   python run_paperseeker.py
   ```

#### 2. Starting the server manually:
   ```
   cd backend
   python run.py
   ```

The API will be available at `http://localhost:8000` with API documentation at `http://localhost:8000/docs`.

## API Endpoints

### Main API Endpoints

- `GET /`: Welcome message and API status
- `GET /search?query=<query>&top_k=<number>&augment=<boolean>`: RAG-powered paper search
- `GET /papers`: Get all papers in the RAG engine
- `POST /papers`: Add a new paper to the RAG engine

### Legacy/Frontend API Endpoints

- `POST /api/search`: Search papers using traditional methods
- `POST /api/stream`: Stream search results with typing effect
- `GET /api/welcome`: Get welcome message
- `GET /api/topics`: Get all available paper topics
- `GET /api/papers`: Get all papers (mock data)
- `POST /api/papers`: Add a paper (mock data)
- `PUT /api/papers/{query}`: Update a paper (mock data)
- `DELETE /api/papers/{query}`: Delete a paper (mock data)
- `GET /api/data-source`: Get information about data source

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Configuration
PORT=8000
HOST=0.0.0.0
RELOAD=true

# RAG Engine Configuration
USE_RAG=true
USE_MILVUS=false

# Optional: OpenAI API Key (for enhanced augmentation)
# OPENAI_API_KEY=your_api_key_here
```

## Using the Client

The project includes a simple command-line client for interacting with the API:

### Search for papers:
```bash
python client.py search "transformer models" 3
```

### Add a new paper:
```bash
python client.py add-paper
```

### List all papers:
```bash
python client.py list-papers
```

## License

MIT
 
