#!/usr/bin/env python3
import json
import logging
import numpy as np
import requests
import os
import tiktoken
import re
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import MilvusClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models import Paper
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flag to track if NLTK resources are available
nltk_available = True

# Simple fallback tokenizer function in case NLTK fails
def simple_tokenize(text):
    """
    Enhanced simple tokenizer that splits on whitespace and punctuation,
    with special handling for common academic text patterns
    """
    if not text:
        return []
        
    # Convert to lowercase
    text = text.lower()
    
    # Handle special cases commonly found in academic papers
    # Replace hyphens in compound words with spaces
    text = re.sub(r'(\w)-(\w)', r'\1 \2', text)
    
    # Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    
    # Ensure spacing around parentheses and brackets
    text = re.sub(r'[\(\[\{\)\]\}]', ' ', text)
    
    # Handle common academic abbreviations
    text = re.sub(r'(fig\.|eq\.|ref\.|e\.g\.|i\.e\.|et al\.)', ' ', text)
    
    # Replace all punctuation with spaces - more comprehensive than the previous version
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    # Split and filter short words
    tokens = [token for token in text.split() if len(token) > 2]
    
    return tokens


# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.2"  # Llama 3.2 model name in Ollama

# Zilliz Cloud settings
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI", "")  # From .env file
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN", "")  # From .env file
COLLECTION_NAME = "paper_collection"

class VectorSearch:
    """Class for semantic search using Zilliz Cloud and Ollama."""
    
    def __init__(self):
        self.client = None
        self.embedding_dim = 3072  # Make sure this matches the dimension in generate_embeddings.py
        self.connect_to_zilliz()
        self.check_collection()
        # Initialize stopwords if NLTK is available, otherwise use a basic list
        if nltk_available:
            self.stop_words = set(stopwords.words('english'))
            print(self.stop_words)
        else:
            # Basic English stopwords as fallback
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                              'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these',
                              'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
                              'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in',
                              'on', 'by', 'with', 'at'}
    
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
    
    def tokenize_with_tiktoken(self, text: str, max_tokens: int = 8192) -> str:
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
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama API with improved text processing.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        try:
            # Process text with tiktoken
            processed_text = self.tokenize_with_tiktoken(text)
            
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
    
    def generate_chunked_embeddings(self, text: str) -> Tuple[List[float], List[str]]:
        """
        Generate embeddings by splitting text into chunks and combining the results.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            Tuple of (final embedding vector, list of chunks used)
        """
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
        chunks = text_splitter.split_text(text)
        
        logger.info(f"Split query into {len(chunks)} chunks")
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        valid_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating embedding for chunk {i+1}/{len(chunks)}")
            embedding = self.generate_embedding(chunk)
            
            if embedding:
                chunk_embeddings.append(embedding)
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Failed to generate embedding for chunk {i+1}")
        
        # If no valid embeddings were generated, return empty results
        if not chunk_embeddings:
            logger.error(f"Failed to generate any valid embeddings for query")
            return [], []
        
        # Combine the embeddings by averaging
        stacked = np.array(chunk_embeddings)
        combined_embedding = np.mean(stacked, axis=0).tolist()
        
        logger.info(f"Successfully generated combined embedding (dim={len(combined_embedding)}) from {len(chunk_embeddings)} chunks")
        
        return combined_embedding, valid_chunks
    
    def preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 by tokenizing, lowercasing, and removing stopwords.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []
        
        # Use enhanced simple tokenizer as fallback
        all_tokens = simple_tokenize(text)
        tokens = [token for token in all_tokens if token not in self.stop_words]
        
        # Filter very common words in academic papers that might not be in standard stopwords
        academic_stopwords = {'paper', 'study', 'research', 'method', 'data', 'result', 'results',
                            'analysis', 'approach', 'model', 'figure', 'table', 'using', 'based',
                            'proposed', 'show', 'shows', 'shown', 'used', 'use', 'uses', 'present',
                            'presents', 'presented', 'describe', 'describes', 'described'}
        
        tokens = [token for token in tokens if token not in academic_stopwords]
        return tokens
        
    def rerank_with_bm25(self, query: str, papers: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Rerank search results using BM25 algorithm.
        
        Args:
            query: Original search query
            papers: List of papers from vector search
            top_k: Number of top papers to return after reranking
            
        Returns:
            List of reranked papers (top_k)
        """
        if not papers:
            return []
            
        logger.info(f"Reranking {len(papers)} papers with BM25")
        
        try:
            # Preprocess query
            query_tokens = self.preprocess_text_for_bm25(query)
            if not query_tokens:
                logger.warning("No valid tokens in query after preprocessing, skipping BM25 reranking")
                return papers[:top_k] if len(papers) > top_k else papers
            
            # Create corpus for BM25
            corpus = []
            papers_list = []
            
            for paper in papers:
                try:
                    # Try to handle different paper formats
                    try:
                        # Get paper object to use with prepare_paper_embedding_content
                        if isinstance(paper, dict):
                            # If direct dictionary, parse into Paper object
                            paper_obj = Paper.parse_obj(paper)
                        else:
                            # If JSON string, parse into Paper object
                            paper_obj = Paper.parse_obj(json.loads(paper))
                        
                        # Get formatted text using the standard method
                        paper_text = paper_obj.prepare_paper_embedding_content()
                        
                    except Exception as e:
                        logger.warning(f"Error preparing paper text for BM25: {e} - using fallback method")
                        # Fallback to simple concatenation of title and abstract
                        title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
                        abstract = paper.get('metadata', {}).get('abstract', '') or paper.get('abstract', '')
                        paper_text = f"TITLE: {title}\n\nABSTRACT: {abstract}"
                    
                    # Preprocess the text
                    tokens = self.preprocess_text_for_bm25(paper_text)
                    
                    # Only include papers with enough tokens
                    if len(tokens) > 3:  # Need at least a few meaningful tokens
                        corpus.append(tokens)
                        papers_list.append(paper)
                    else:
                        logger.warning(f"Paper skipped for BM25 (insufficient tokens): {paper.get('id', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Error processing paper for BM25: {e}")
                    # Don't include this paper in BM25 ranking but keep it in results
            
            # Check if we have enough papers for reranking
            if len(corpus) < 2:
                logger.warning("Not enough valid papers for BM25 reranking, returning original results")
                return papers[:top_k] if len(papers) > top_k else papers
            
            # Initialize BM25
            bm25 = BM25Okapi(corpus)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Combine with papers
            scored_papers = list(zip(papers_list, bm25_scores))
            
            # Sort by BM25 score (descending)
            scored_papers.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            reranked_papers = [paper for paper, score in scored_papers[:top_k]]
            
            # Add BM25 scores to papers
            for i, (paper, score) in enumerate(scored_papers[:top_k]):
                reranked_papers[i]['bm25_score'] = score
                
            logger.info(f"BM25 reranking complete, returning top {len(reranked_papers)} papers")
            
            return reranked_papers
            
        except Exception as e:
            logger.error(f"Error during BM25 reranking: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fall back to original results if BM25 fails
            return papers[:top_k] if len(papers) > top_k else papers
    
    def search(self, query: str, limit: int = 5, rerank: bool = True, vector_candidates: int = 100, rerank_top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for papers similar to the query using vector similarity, with optional BM25 reranking.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return (final output)
            rerank: Whether to apply BM25 reranking
            vector_candidates: Number of initial vector search candidates to retrieve
            rerank_top_k: Number of results to keep after reranking
            
        Returns:
            List of papers that match the query
        """
        try:
            # For reranking, we need to get more candidates first
            initial_limit = vector_candidates if rerank else limit
            
            # Generate embedding for the query using the chunked approach
            query_embedding, chunks = self.generate_chunked_embeddings(query)
            
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
                search_params=search_params,
                limit=initial_limit,
                output_fields=output_fields
            )
            
            # Process results
            papers = []
            for hits in results:
                for hit in hits:
                    # Debug log to see the structure of hit
                    logger.debug(f"Hit keys: {hit.keys()}")
                    
                    # Get the entity data
                    entity = hit["entity"]
                    
                    # The json_data field already contains the complete paper
                    try:
                        paper_json = json.loads(entity["json_data"])
                    except:
                        paper_json = entity["json_data"]
                    
                    # Add the similarity score - use different possible key names
                    if "score" in hit:
                        paper_json['similarity_score'] = hit["score"]
                    elif "distance" in hit:
                        paper_json['similarity_score'] = hit["distance"]
                    elif "similarity" in hit:
                        paper_json['similarity_score'] = hit["similarity"]
                    else:
                        # If no score is found, log available keys and use a default
                        logger.warning(f"No score key found in hit. Available keys: {hit.keys()}")
                        paper_json['similarity_score'] = 0.0
                    
                    papers.append(paper_json)
            
            logger.info(f"Found {len(papers)} papers for query: '{query}' using vector search")
            
            # Apply BM25 reranking if requested and we have results
            if rerank and papers:
                papers = self.rerank_with_bm25(query, papers, top_k=rerank_top_k)
                logger.info(f"After BM25 reranking, keeping top {len(papers)} papers")
                
                # Limit to the user-requested number if needed
                if len(papers) > limit:
                    papers = papers[:limit]
            
            return papers
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def close(self):
        """Clean up resources."""
        logger.info("Vector search resources released")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python vector_search.py <query> <limit>")
        sys.exit(1)
    
    query = sys.argv[1]
    limit = int(sys.argv[2])
    search = VectorSearch()
    
    try:
        # Get results with default settings (100 vector candidates, reranked to top 20)
        results = search.search(query, limit=limit)
        
        print(f"\nSearch results for '{query}':\n")
        for i, paper in enumerate(results):
            title = paper.get('metadata', {}).get('title', 'No title') or paper.get('title', 'No title')
            year = paper.get('metadata', {}).get('year', 'N/A') or paper.get('year', 'N/A')
            bm25_score = paper.get('bm25_score', 'N/A')
            vector_score = paper.get('similarity_score', 0)
            
            print(f"{i+1}. {title} ({year})")
            print(f"   BM25 Score: {bm25_score}")
            print(f"   Vector Score: {vector_score:.4f}")
            abstract = paper.get('metadata', {}).get('abstract', 'No abstract') or paper.get('abstract', 'No abstract')
            print(f"   Abstract: {abstract[:200]}...")
            print()
    finally:
        search.close() 