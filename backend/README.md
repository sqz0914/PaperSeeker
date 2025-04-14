# PaperSeeker Backend API

A simple API for searching and retrieving academic papers using FastAPI.

## Setup and Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
   ```
   pip install fastapi uvicorn
   ```
3. Run the API server:
   ```
   python app.py
   ```
   
The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the interactive API documentation.

### Available Endpoints

#### General Endpoints

- `GET /`: Basic information about the API
- `GET /papers`: Get all available papers
- `GET /papers/{paper_id}`: Get a specific paper by ID, parsed using the Paper model
- `POST /search`: Search for papers matching a query term

#### Frontend Support Endpoints

- `GET /api/topics`: Get all available paper topics/fields of study
- `POST /api/search`: Simple search API returning formatted responses for the chat interface
- `POST /api/stream`: Streaming search API with typing effect for interactive UI
- `GET /api/welcome`: Stream the welcome message

### Models

#### Paper Model

The API uses a detailed `Paper` model from `paper.py` that can parse raw paper data into structured objects, including:

- Basic metadata (id, title, authors, abstract, year)
- Source information
- External identifiers
- Fields of study
- Publication details

## Example Usage

### Search for Papers

```http
POST /search
Content-Type: application/json

{
  "query": "vaccine"
}
```

Response:
```json
{
  "papers": [
    {
      "id": "25228287",
      "source": "pes2o/s2ag",
      "version": "v3-fos-license",
      "text": "A fully synthetic four-component antitumor vaccine...",
      "metadata": {
        "title": "A fully synthetic four-component antitumor vaccine...",
        "abstract": "In a new concept of fully synthetic vaccines...",
        "year": 2014,
        ...
      }
    }
  ]
}
```

### Get a Specific Paper

```http
GET /papers/25228287
```

Response:
```json
{
  "id": "25228287",
  "title": "A fully synthetic four-component antitumor vaccine...",
  "abstract": "In a new concept of fully synthetic vaccines...",
  "year": "2014",
  "topic": "Medicine",
  ...
}
```

## Implementation Details

- Basic text-based search over titles, abstracts, and full paper text
- Structured paper models using Pydantic
- Streaming responses for interactive chat interface
- CORS support for frontend integration 