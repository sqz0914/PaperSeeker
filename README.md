# PaperSeeker

A chatbot application that helps users find and explore research papers.

## Project Structure

- `frontend/`: Next.js application with the chat interface
- `backend/`: FastAPI server providing paper search capabilities

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # On Windows
   .\.venv\Scripts\activate
   # On Unix/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```
   python run.py
   ```
   
   The server will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

   The frontend will be available at http://localhost:3000

## Features

- Chat interface for asking about research papers
- Search for papers on topics like deep learning, transformers, machine learning, etc.
- Citations provided for referenced papers

## Tech Stack

- **Frontend**: Next.js, React, TailwindCSS
- **Backend**: FastAPI, Python
 
