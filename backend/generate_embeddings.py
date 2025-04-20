#!/usr/bin/env python3
import json
import os
import sys
import logging
import numpy as np
import requests
import tiktoken
import re
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import MilvusClient
from pymilvus import DataType
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse

# Load environment variables
load_dotenv()

# Import the Paper model for properly parsing the papers
from models import Paper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.2"  # Llama 3.2 model name in Ollama

# Zilliz Cloud settings
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI", "")  # From .env file
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN", "")  # From .env file
COLLECTION_NAME = "paper_collection"
EMBEDDING_DIM = 3072  # Dimension of embeddings from llama3.2 model

def tokenize_with_tiktoken(text: str, max_tokens: int = 8192) -> str:
    """
    Tokenize and truncate text using tiktoken for better embedding quality.
    
    Args:
        text: Text to tokenize
        max_tokens: Maximum number of tokens to keep
        
    Returns:
        Processed text string ready for embedding
    """
    try:
        # Initialize the cl100k_base tokenizer (used by many newer models)
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize the text
        tokens = encoding.encode(text)
        
        # Truncate if needed
        if len(tokens) > max_tokens:
            logger.info(f"Truncating text from {len(tokens)} to {max_tokens} tokens")
            tokens = tokens[:max_tokens]
        
        # Convert back to text
        processed_text = encoding.decode(tokens)
        
        return processed_text
    except Exception as e:
        logger.warning(f"Error during tiktoken processing: {e}, falling back to simple truncation")
        # Simple fallback - just truncate characters
        return text[:32000]

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding using Ollama API with improved text processing.
    
    Args:
        text: Text to embed
        
    Returns:
        Vector embedding
    """
    try:
        # Process text with tiktoken
        processed_text = tokenize_with_tiktoken(text)
        
        # Log token counts
        logger.info(f"Embedding text with approximate length: {len(processed_text)} characters")
        
        # Send to Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "prompt": processed_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            
            # Log embedding dimension for debugging
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            
            return embedding
        else:
            logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def generate_chunked_embeddings(paper: Paper) -> Tuple[List[float], List[str]]:
    """
    Generate embeddings for a paper by splitting it into chunks and combining the results.
    
    Args:
        paper: The paper to generate embeddings for
        
    Returns:
        Tuple of (final embedding vector, list of chunks used)
    """
    # Prepare the formatted text content
    paper_text = paper.prepare_paper_embedding_content()
    
    # Initialize LangChain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a small chunk size to ensure we don't exceed token limits
        chunk_size=200,
        chunk_overlap=20,
        # Define separators in order of priority
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        # Length function uses token count instead of characters
        length_function=lambda text: len(tiktoken.get_encoding("cl100k_base").encode(text))
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(paper_text)
    
    logger.info(f"Split paper '{paper.title}' into {len(chunks)} chunks")
    
    # Generate embeddings for each chunk
    chunk_embeddings = []
    valid_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Generating embedding for chunk {i+1}/{len(chunks)}")
        embedding = generate_embedding(chunk)
        
        if embedding:
            chunk_embeddings.append(embedding)
            valid_chunks.append(chunk)
        else:
            logger.warning(f"Failed to generate embedding for chunk {i+1}")
    
    # If no valid embeddings were generated, return empty results
    if not chunk_embeddings:
        logger.error(f"Failed to generate any valid embeddings for paper '{paper.title}'")
        return [], []
    
    # Combine the embeddings by averaging
    stacked = np.array(chunk_embeddings)
    combined_embedding = np.mean(stacked, axis=0).tolist()
    
    logger.info(f"Successfully generated combined embedding (dim={len(combined_embedding)}) from {len(chunk_embeddings)} chunks")
    
    return combined_embedding, valid_chunks

def process_single_paper(paper_data: Dict[str, Any], milvus_client: MilvusClient) -> bool:
    """
    Process a single paper: parse, generate embedding, and store in Milvus.
    
    Args:
        paper_data: Raw dictionary data for a paper
        milvus_client: MilvusClient instance
        
    Returns:
        bool: Whether processing was successful
    """
    try:
        # Parse the paper
        try:
            paper = Paper.parse_data(paper_data)
            logger.info(f"Processing paper: {paper.title}")
        except Exception as e:
            logger.error(f"Error parsing paper: {e}")
            return False
        
        # Convert datetime objects to strings for JSON serialization
        raw_paper = paper.dict(exclude_none=True)
        if paper.added:
            raw_paper["added"] = paper.added.isoformat()
        if paper.created:
            raw_paper["created"] = paper.created.isoformat()
            
        # Generate embeddings
        embedding, chunks = generate_chunked_embeddings(paper)
        
        if not embedding:
            logger.warning(f"Skipping paper {paper.id} - failed to generate embedding")
            return False
            
        # Format paper data for insertion with all fields
        entry = {
            # Primary key and basic fields
            "id": str(paper.id),
            
            # Main content
            "title": paper.title or "",
            "abstract": paper.abstract[:2000] or "",
            
            # Additional metadata
            "year": str(paper.year) if paper.year else "",
            "citations": [] if not hasattr(paper, 'metadata') or not paper.metadata.external_ids else (
                [str(citation) for citation in paper.metadata.external_ids]
            ),
            
            # Vector embedding
            "embedding": embedding,
            
            # Complete JSON - Convert any problematic fields to strings
            "json_data": json.dumps(raw_paper)
        }
        
        # Insert into collection
        milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=[entry]  # Insert a single paper
        )
        logger.info(f"Successfully inserted paper '{paper.title}' into Milvus")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def setup_milvus_collection() -> MilvusClient:
    """
    Setup connection to Milvus and create collection if needed.
    
    Returns:
        MilvusClient instance
    """
    try:
        # Connect to Milvus
        milvus_client = MilvusClient(
            uri=ZILLIZ_CLOUD_URI,
            token=ZILLIZ_CLOUD_TOKEN,
        )
        logger.info(f"Connected to Zilliz Cloud at {ZILLIZ_CLOUD_URI}")
        
        # Check if collection exists and create if needed
        if milvus_client.has_collection(COLLECTION_NAME):
            logger.info(f"Collection {COLLECTION_NAME} already exists, dropping it for fresh setup")
            milvus_client.drop_collection(COLLECTION_NAME)
        
        # Create schema using the API style
        logger.info("Creating collection schema")
        schema = milvus_client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100, description="Paper ID")
        schema.add_field("title", DataType.VARCHAR, max_length=2000, description="Paper title")
        schema.add_field("abstract", DataType.VARCHAR, max_length=2000, description="Paper abstract")
        schema.add_field("year", DataType.VARCHAR, max_length=10, description="Publication year")
        schema.add_field("citations", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=100, description="Paper citations")
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM, description="Text embedding vector")
        schema.add_field("json_data", DataType.JSON, description="Complete paper data as JSON")
        
        # Prepare index parameters
        logger.info("Preparing index parameters")
        index_params = milvus_client.prepare_index_params()
        index_params.add_index("embedding", metric_type="COSINE", index_type="HNSW", params={"M": 8, "efConstruction": 64})
        
        # Create collection with schema and index parameters
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        
        return milvus_client
        
    except Exception as e:
        logger.error(f"Error setting up Milvus collection: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def process_papers_stream(data_path: str, limit: Optional[int] = None):
    """
    Process papers from a single file or all files in a directory.
    
    Args:
        data_path: Path to a file or directory containing papers
        limit: Optional limit on the number of papers to process per file
    """
    try:
        # Determine if the path is a directory or a file
        is_directory = os.path.isdir(data_path)
        
        # Set up Milvus collection once before processing
        milvus_client = setup_milvus_collection()
        
        if is_directory:
            logger.info(f"Processing all paper data files in directory: {data_path}")
            paper_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                          if f.endswith('.json') or f.endswith('.jsonl')]
            
            if not paper_files:
                logger.error(f"No JSON or JSONL files found in {data_path}")
                return
            
            logger.info(f"Found {len(paper_files)} paper data files to process")
            
            # Process each file
            for i, file_path in enumerate(paper_files, 1):
                logger.info(f"Processing file {i}/{len(paper_files)}: {file_path}")
                process_single_file(file_path, milvus_client, limit)
                logger.info(f"Completed file {i}/{len(paper_files)}")
            
            logger.info(f"All {len(paper_files)} paper data files have been processed")
        else:
            # Single file processing
            logger.info(f"Processing single paper data file: {data_path}")
            if not os.path.exists(data_path):
                logger.error(f"File not found: {data_path}")
                return
            
            process_single_file(data_path, milvus_client, limit)
        
        logger.info("Papers indexing process completed successfully")
        
        # Verify final collection size
        try:
            collection_stats = milvus_client.get_collection_stats(COLLECTION_NAME)
            paper_count = collection_stats["row_count"]
            logger.info(f"Milvus collection '{COLLECTION_NAME}' contains {paper_count} papers")
        except Exception as e:
            logger.error(f"Error checking collection stats: {e}")
    
    except Exception as e:
        logger.error(f"Error in processing papers: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def process_single_file(file_path: str, milvus_client: MilvusClient, limit: Optional[int] = None):
    """
    Process a single paper data file.
    
    Args:
        file_path: Path to the file containing papers
        milvus_client: MilvusClient instance
        limit: Optional limit on the number of papers to process
    """
    # Track statistics
    total_papers = 0
    successful_papers = 0
    failed_papers = 0
    
    try:
        # Detect file format and process papers one by one
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if it's a JSONL file (each line is a JSON object)
            first_line = f.readline().strip()
            # Reset file pointer to beginning
            f.seek(0)
            
            if first_line.startswith('{') and first_line.endswith('}'):
                # Process as JSONL (each line is a paper)
                logger.info(f"Processing {file_path} as JSONL format")
                
                for line_num, line in enumerate(f, 1):
                    if not line.strip():  # Skip empty lines
                        continue
                    
                    # Apply limit if specified
                    if limit and total_papers >= limit:
                        logger.info(f"Reached limit of {limit} papers, stopping")
                        break
                    
                    total_papers += 1
                    logger.info(f"Processing paper {total_papers} from line {line_num}")
                    
                    try:
                        paper_data = json.loads(line.strip())
                        success = process_single_paper(paper_data, milvus_client)
                        
                        if success:
                            successful_papers += 1
                            # Flush every 10 papers to ensure data is persisted
                            if successful_papers % 10 == 0:
                                logger.info("Flushing data to ensure persistence")
                                milvus_client.flush(COLLECTION_NAME)
                        else:
                            failed_papers += 1
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON line {line_num}: {e}")
                        failed_papers += 1
            else:
                # Try standard JSON format (array of objects)
                try:
                    logger.info(f"Processing {file_path} as JSON array format")
                    f.seek(0)  # Reset to beginning of file
                    papers_array = json.load(f)
                    
                    if isinstance(papers_array, list):
                        for i, paper_data in enumerate(papers_array, 1):
                            # Apply limit if specified
                            if limit and total_papers >= limit:
                                logger.info(f"Reached limit of {limit} papers, stopping")
                                break
                            
                            total_papers += 1
                            logger.info(f"Processing paper {total_papers} of {len(papers_array)}")
                            
                            success = process_single_paper(paper_data, milvus_client)
                            
                            if success:
                                successful_papers += 1
                                # Flush every 10 papers to ensure data is persisted
                                if successful_papers % 10 == 0:
                                    logger.info("Flushing data to ensure persistence")
                                    milvus_client.flush(COLLECTION_NAME)
                            else:
                                failed_papers += 1
                    else:
                        logger.error(f"Expected JSON array but got {type(papers_array)}")
                        
                except json.JSONDecodeError:
                    logger.error(f"Could not parse {file_path} as either JSONL or JSON array")
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    
    # Final flush to ensure all data is persisted for this file
    logger.info("Flushing data to ensure persistence")
    milvus_client.flush(COLLECTION_NAME)
    
    # Log summary statistics for this file
    logger.info(f"File processing complete: {file_path}")
    logger.info(f"Total papers processed: {total_papers}")
    logger.info(f"Successfully indexed: {successful_papers}")
    logger.info(f"Failed: {failed_papers}")

def main():
    """Parse command line arguments and start the paper processing."""
    parser = argparse.ArgumentParser(description="Generate embeddings for papers and index them in Milvus")
    parser.add_argument("data_path", help="Path to a folder containing paper data files or a single paper data file")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of papers to process per file", default=None)
    
    args = parser.parse_args()
    
    # Process the papers
    process_papers_stream(args.data_path, args.limit)

if __name__ == "__main__":
    main() 