import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

# Configure logger before imports that might use it
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence_transformers not available. Vector search will be disabled.")
    HAVE_SENTENCE_TRANSFORMERS = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for academic papers.
    
    This class handles:
    1. Loading papers
    2. Embedding papers
    3. Searching papers based on semantic similarity
    4. Augmenting search results with LLM-generated content
    """
    
    def __init__(self, data_file: str = "paper_data.json", use_milvus: bool = False):
        """
        Initialize the RAG engine.
        
        Args:
            data_file: Path to the papers data file
            use_milvus: Whether to use Milvus vector DB (currently not implemented)
        """
        # Adjust path to be relative to the data directory
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", data_file)
        self.use_milvus = use_milvus
        
        # Initialize the embedding model
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("Sentence transformers not available - vector search disabled")
        
        # Initialize storage for papers and embeddings
        self.papers = []
        self.embeddings = None
        self.index = {}  # Text search index
        
        # Sample papers path
        self.sample_papers_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data",
            "sample_papers.json"
        )
        
        # Try to load papers
        self._load_data()
        
        logger.info("RAG Engine initialized")
    
    def _load_data(self) -> None:
        """Load paper data from the data file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    first_char = f.read(1)
                    f.seek(0)
                    
                    if first_char == '[':
                        # It's a JSON array
                        self.papers = json.load(f)
                    else:
                        # Assume it's JSONL (one object per line)
                        self.papers = []
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                paper = json.loads(line)
                                self.papers.append(paper)
                
                # Process papers for format compatibility
                self._process_papers()
                
                # Generate embeddings and index
                self._generate_embeddings()
                self._build_text_index()
                logger.info(f"Loaded {len(self.papers)} papers from {self.data_file}")
            else:
                # If data file doesn't exist, try to load sample papers
                self.load_sample_papers()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # If we can't load the main data file, try to load sample papers
            self.load_sample_papers()
    
    def _process_papers(self) -> None:
        """Process loaded papers to ensure consistent format"""
        processed_papers = []
        for paper in self.papers:
            # Check if paper has metadata structure or direct properties
            if 'metadata' in paper:
                metadata = paper['metadata']
                processed_paper = {
                    'title': metadata.get('title', ''),
                    'abstract': metadata.get('abstract', ''),
                    'authors': [],
                    'year': metadata.get('year', ''),
                    'url': metadata.get('url', ''),
                    'topic': metadata.get('topic', 'Research'),
                    'id': paper.get('id', '')
                }
                
                # Extract authors
                authors = metadata.get('authors', [])
                for author in authors:
                    if isinstance(author, dict) and 'name' in author:
                        processed_paper['authors'].append(author['name'])
                    elif isinstance(author, str):
                        processed_paper['authors'].append(author)
                
                # Add the full text if available
                processed_paper['text'] = paper.get('text', '')
                
            else:
                # Assume direct properties
                processed_paper = {
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'authors': paper.get('authors', []),
                    'year': paper.get('year', ''),
                    'url': paper.get('url', ''),
                    'topic': paper.get('topic', 'Research'),
                    'id': paper.get('id', '')
                }
            
            processed_papers.append(processed_paper)
        
        self.papers = processed_papers
    
    def _build_text_index(self) -> None:
        """Build an inverted index for text search fallback"""
        self.index = {}
        
        for idx, paper in enumerate(self.papers):
            # Index title words
            if paper.get('title'):
                for word in paper['title'].lower().split():
                    word = word.strip('.,;:?!()[]{}"\'')
                    if word and len(word) >= 3:
                        if word not in self.index:
                            self.index[word] = []
                        self.index[word].append(idx)
            
            # Index abstract words
            if paper.get('abstract'):
                for word in paper['abstract'].lower().split():
                    word = word.strip('.,;:?!()[]{}"\'')
                    if word and len(word) >= 3:
                        if word not in self.index:
                            self.index[word] = []
                        self.index[word].append(idx)
        
        logger.info(f"Built text search index with {len(self.index)} terms")
    
    def load_sample_papers(self) -> None:
        """Load sample papers from the JSON file"""
        try:
            if not os.path.exists(self.sample_papers_path):
                logger.warning(f"Sample papers file not found: {self.sample_papers_path}")
                return
            
            with open(self.sample_papers_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            if not papers:
                logger.warning("No papers found in the sample file")
                return
                
            # Add papers and generate embeddings
            self.papers = papers
            self._generate_embeddings()
            self._build_text_index()
            logger.info(f"Loaded {len(papers)} sample papers")
        except Exception as e:
            logger.error(f"Error loading sample papers: {e}")
            raise
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all papers"""
        if not self.papers:
            logger.warning("No papers to generate embeddings for")
            return
        
        if not self.embedding_model:
            logger.warning("No embedding model available - skipping embedding generation")
            return
        
        try:
            # Create combined text representation for each paper
            texts = [
                f"{paper['title']} {paper['abstract']} {paper.get('topic', '')}"
                for paper in self.papers
            ]
            
            # Generate embeddings
            self.embeddings = self.embedding_model.encode(texts)
            logger.info(f"Generated embeddings for {len(texts)} papers")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.embeddings = None
    
    def search_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of paper dictionaries
        """
        if not self.papers:
            logger.warning("No papers available for search")
            return []
        
        # Use the unified search method
        return self.search(query, top_k, augment=False)
    
    def search_and_augment(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Search for papers and augment results with LLM-generated content
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (papers, augmented_content)
        """
        try:
            # First perform the search
            papers = self.search(query, top_k, augment=False)
            
            if not papers:
                return [], "No relevant papers found."
            
            # For now, return a simple summary without LLM augmentation
            # In a real implementation, you would call an LLM API here
            augmented_content = (
                f"Found {len(papers)} papers related to '{query}'. "
                f"The most relevant paper is '{papers[0]['title']}' "
                f"by {', '.join(papers[0]['authors'])} ({papers[0]['year']}). "
                f"To implement actual LLM augmentation, you'll need to "
                f"connect to an LLM API like OpenAI or use a local model."
            )
            
            return papers, augmented_content
        except Exception as e:
            logger.error(f"Error in search and augment: {e}")
            return [], f"Error during search: {str(e)}"
    
    def search(self, query: str, top_k: int = 3, augment: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Search for papers matching the query using vector search if available.
        Falls back to text-based search if vector search fails.
        
        Args:
            query: The search query string
            top_k: Number of results to return
            augment: Whether to augment result with additional context
            
        Returns:
            If augment=True: Dict with papers list and augmented content
            Otherwise: List of paper dictionaries
        """
        results = []
        
        # Try vector search first if model is available
        if self.embedding_model is not None and self.embeddings is not None and len(self.embeddings) > 0:
            try:
                # Encode the query
                query_embedding = self.embedding_model.encode(query)
                
                # Calculate cosine similarity
                similarities = np.dot(self.embeddings, query_embedding) / (
                    np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                # Get papers
                for idx in top_indices:
                    paper_copy = self.papers[idx].copy()  # Make a copy to avoid modifying original
                    paper_copy['score'] = float(similarities[idx])
                    results.append(paper_copy)
                
                logger.info(f"Found {len(results)} papers using vector search for query: {query}")
            except Exception as e:
                logger.error(f"Error in vector search: {e}")
        
        # Fall back to text-based search if no results or vector search failed
        if not results:
            try:
                # Simple tokenization by splitting on whitespace
                query_terms = query.lower().split()
                
                # Get matches for each term
                matches = {}
                for term in query_terms:
                    # Remove punctuation from term
                    term = term.strip('.,;:?!()[]{}"\'')
                    if not term or len(term) < 3:  # Skip short terms
                        continue
                    
                    # Get paper indices for term
                    term_matches = self.index.get(term, [])
                    
                    # Add to matches dict with count
                    for idx in term_matches:
                        if idx not in matches:
                            matches[idx] = 0
                        matches[idx] += 1
                
                # Sort by match count
                sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                # Get papers
                for idx, count in sorted_matches:
                    paper_copy = self.papers[idx].copy()  # Make a copy to avoid modifying original
                    paper_copy['score'] = count / len(query_terms)  # Normalize score
                    results.append(paper_copy)
                
                logger.info(f"Found {len(results)} papers using text search for query: {query}")
            except Exception as e:
                logger.error(f"Error in text search: {e}")
        
        # Return results or augmented result
        if not results:
            if augment:
                return {"papers": [], "augmented_content": "No relevant papers found."}
            return []
        
        # If we need augmentation
        if augment:
            # Generate augmented content
            augmented_content = self._generate_augmentation(query, results)
            return {
                "papers": results,
                "augmented_content": augmented_content
            }
        
        # Return raw results
        return results
    
    def _generate_augmentation(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """
        Generate augmented content from search results
        
        Args:
            query: The search query
            papers: List of paper results
            
        Returns:
            Augmented content as string
        """
        if not papers:
            return "No relevant papers found."
        
        top_paper = papers[0]
        
        # Format the response nicely
        response = f"Found {len(papers)} papers related to '{query}'. "
        response += f"The most relevant paper is '{top_paper['title']}'"
        
        if top_paper.get('year'):
            response += f" published in {top_paper['year']}"
        response += "."
        
        # Add authors if available
        if top_paper.get('authors'):
            authors = top_paper['authors']
            if len(authors) == 1:
                response += f" The paper was written by {authors[0]}."
            elif len(authors) == 2:
                response += f" The paper was written by {authors[0]} and {authors[1]}."
            else:
                response += f" The paper was written by {authors[0]} et al."
        
        if top_paper.get('abstract'):
            # Add a short excerpt from the abstract
            abstract_excerpt = top_paper['abstract'][:250] + "..." if len(top_paper['abstract']) > 250 else top_paper['abstract']
            response += f" The paper discusses: {abstract_excerpt}"
        
        # Add a brief summary of other papers
        if len(papers) > 1:
            response += f"\n\nOther relevant papers include:"
            for i, paper in enumerate(papers[1:3]):  # Only mention up to 2 more papers
                response += f"\n- '{paper['title']}'"
                if paper.get('year'):
                    response += f" ({paper['year']})"
                if paper.get('authors') and len(paper['authors']) > 0:
                    response += f" by {paper['authors'][0]}"
                    if len(paper['authors']) > 1:
                        response += " et al."
        
        return response
    
    def add_paper(self, paper: Dict[str, Any]) -> None:
        """
        Add a new paper to the collection
        
        Args:
            paper: Paper dictionary
        """
        try:
            # Add the paper
            self.papers.append(paper)
            
            # Regenerate embeddings and rebuild index
            self._generate_embeddings()
            self._build_text_index()
            
            # Save to data file for persistence
            self._save_data()
            
            logger.info(f"Added paper: {paper['title']}")
        except Exception as e:
            logger.error(f"Error adding paper: {e}")
            raise
    
    def _save_data(self) -> None:
        """Save paper data to the data file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.papers, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.papers)} papers to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def update_paper(self, paper_id: str, updated_paper: Dict[str, Any]) -> bool:
        """
        Update an existing paper by its ID
        
        Args:
            paper_id: ID of the paper to update
            updated_paper: Updated paper data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for i, paper in enumerate(self.papers):
                if str(paper.get('id', '')) == str(paper_id):
                    # Preserve the ID
                    updated_paper['id'] = paper_id
                    # Update the paper
                    self.papers[i] = updated_paper
                    # Regenerate embeddings and rebuild index
                    self._generate_embeddings()
                    self._build_text_index()
                    # Save to data file
                    self._save_data()
                    logger.info(f"Updated paper: {updated_paper['title']}")
                    return True
            
            logger.warning(f"Paper not found with ID: {paper_id}")
            return False
        except Exception as e:
            logger.error(f"Error updating paper: {e}")
            return False
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper by its ID
        
        Args:
            paper_id: ID of the paper to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for i, paper in enumerate(self.papers):
                if str(paper.get('id', '')) == str(paper_id):
                    # Remove the paper
                    del self.papers[i]
                    # Regenerate embeddings and rebuild index
                    self._generate_embeddings()
                    self._build_text_index()
                    # Save to data file
                    self._save_data()
                    logger.info(f"Deleted paper with ID: {paper_id}")
                    return True
            
            logger.warning(f"Paper not found with ID: {paper_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting paper: {e}")
            return False
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers"""
        return self.papers
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a paper by its ID."""
        for paper in self.papers:
            if str(paper.get('id', '')) == str(paper_id):
                return paper
        return None


# Example usage
if __name__ == "__main__":
    # Initialize the RAG engine
    engine = RAGEngine()
    
    # Load sample papers
    if not engine.papers:
        engine.load_sample_papers()
    
    # Search for papers
    query = "deep learning neural networks"
    papers = engine.search_papers(query)
    
    # Print top papers
    print(f"\nQuery: {query}")
    print(f"Found {len(papers)} papers")
    for i, paper in enumerate(papers[:3]):
        print(f"\n{i+1}. {paper['title']} by {', '.join(paper['authors'])} ({paper['year']})")
        print(f"   Abstract: {paper['abstract'][:150]}...")
    
    # Test the augmented search
    result = engine.search(query, top_k=3, augment=True)
    if result:
        print("\nAugmented search results:")
        print(f"Content: {result['augmented_content']}")
        print(f"Number of papers: {len(result['papers'])}") 