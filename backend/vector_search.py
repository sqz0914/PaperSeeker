#!/usr/bin/env python3
import json
import logging
import numpy as np
import requests
import os
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3"  # Llama 3.2 model name in Ollama

# Zilliz Cloud settings
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI", "")  # From .env file
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN", "")  # From .env file
COLLECTION_NAME = "paper_collection"

class VectorSearch:
    """Class for semantic search using Zilliz Cloud and Ollama."""
    
    def __init__(self):
        self.client = None
        self.connect_to_zilliz()
        self.check_collection()
    
    def connect_to_zilliz(self):
        """Connect to Zilliz Cloud cluster."""
        try:
            # Check if required environment variables are set
            if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
                raise ValueError("ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set in .env file")
            
            logger.info(f"Connecting to Zilliz Cloud at {ZILLIZ_CLOUD_URI}")
            
            # Initialize client with Zilliz Cloud connection parameters
            self.client = MilvusClient(
                uri=ZILLIZ_CLOUD_URI,
                token=ZILLIZ_CLOUD_TOKEN,
                db_name="default"  # Specify the default database
            )
            
            # Test connection by listing collections
            collections = self.client.list_collections()
            logger.info(f"Connected to Zilliz Cloud. Available collections: {collections}")
        except Exception as e:
            logger.error(f"Failed to connect to Zilliz Cloud: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def check_collection(self):
        """Check if the paper collection exists in Zilliz Cloud."""
        try:
            # Check if collection exists
            if self.client.has_collection(COLLECTION_NAME):
                collection_stats = self.client.get_collection_stats(COLLECTION_NAME)
                paper_count = collection_stats["row_count"]
                logger.info(f"Found collection '{COLLECTION_NAME}' with {paper_count} papers")
            else:
                # Collection doesn't exist yet
                logger.warning(f"Collection '{COLLECTION_NAME}' does not exist. Please run generate_embeddings.py first.")
                raise ValueError(f"Collection '{COLLECTION_NAME}' not found")
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama API with Llama 3.2."""
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": MODEL_NAME, "prompt": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers similar to the query using vector similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of papers that match the query
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",  # Match the index metric type
                "params": {"ef": 64}      # Higher ef gives more accurate results
            }
            
            # List all fields we want to retrieve
            output_fields = ["id", "title", "abstract", "year", "citations", "json_data"]
            
            # Perform the search
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[query_embedding],  # List of query vectors
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            # Process results
            papers = []
            for hits in results:
                for hit in hits:
                    # Get the entity data
                    entity = hit["entity"]
                    
                    # The json_data field already contains the complete paper
                    paper_json = entity["json_data"]
                    
                    # Add the similarity score
                    paper_json['similarity_score'] = hit["score"]
                    
                    papers.append(paper_json)
            
            logger.info(f"Found {len(papers)} papers for query: '{query}'")
            return papers
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def close(self):
        """Clean up resources."""
        logger.info("Vector search resources released")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_search.py <query>")
        sys.exit(1)
    
    query = sys.argv[1]
    search = VectorSearch()
    
    try:
        results = search.search(query, limit=3)
        
        print(f"\nSearch results for '{query}':\n")
        for i, paper in enumerate(results):
            print(f"{i+1}. {paper.get('metadata', {}).get('title', 'No title')} ({paper.get('metadata', {}).get('year', 'N/A')})")
            print(f"   Score: {paper.get('similarity_score', 0):.4f}")
            print(f"   Abstract: {paper.get('metadata', {}).get('abstract', 'No abstract')[:200]}...")
            print()
    finally:
        search.close() 