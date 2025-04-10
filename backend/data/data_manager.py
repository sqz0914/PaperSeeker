import json
import os
from typing import List, Dict, Any, Optional, Union
from ..rag.rag_engine import RAGEngine

class DataManager:
    """Class to manage paper data with RAG capabilities"""
    
    def __init__(self, 
                 mock_data_file: str = "mock_data.json", 
                 real_data_file: str = "paper_data.json",
                 use_milvus: bool = False):
        """Initialize with paths to data files"""
        self.mock_data_file = os.path.join(os.path.dirname(__file__), mock_data_file)
        self.real_data_file = os.path.join(os.path.dirname(__file__), real_data_file)
        self.mock_data = self.load_mock_data()
        
        # Flag to determine if we should use mock data or real data
        self.use_real_data = os.path.exists(self.real_data_file)
        
        # Initialize RAG engine for real paper data
        self.rag_engine = RAGEngine(data_file=real_data_file, use_milvus=use_milvus)
    
    def load_mock_data(self) -> List[Dict[str, Any]]:
        """Load mock paper data from JSON file"""
        try:
            with open(self.mock_data_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading mock data: {e}")
            return []
    
    def save_mock_data(self) -> bool:
        """Save mock paper data to JSON file"""
        try:
            with open(self.mock_data_file, "w") as f:
                json.dump(self.mock_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving mock data: {e}")
            return False
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers (mock data or real data depending on configuration)
        """
        if self.use_real_data:
            # For real data, use the RAG engine's papers list
            return self.rag_engine.papers if hasattr(self.rag_engine, 'papers') else []
        else:
            # For mock data
            return self.mock_data
    
    def get_all_topics(self) -> List[str]:
        """Get all available paper topics from mock data"""
        if not self.use_real_data:
            return [paper["query"] for paper in self.mock_data]
        else:
            # For real data, return some common research areas
            return [
                "machine learning", "deep learning", "natural language processing",
                "computer vision", "neural networks", "transformers", 
                "reinforcement learning", "quantum computing", "bioinformatics",
                "vaccines", "cancer research", "climate change", "renewable energy"
            ]
    
    def search_papers(self, query: str, top_k: int = 1, augment: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """
        Search for papers matching the query
        
        Args:
            query: The search query string
            top_k: Number of results to return
            augment: Whether to augment the result with additional context
            
        Returns:
            - For top_k=1 and augment=True: Dictionary with response and citation
            - Otherwise: List of paper dictionaries
            - None if no results found
        """
        # If real data is available and should be used
        if self.use_real_data:
            # Use RAG engine with vector embeddings
            return self.rag_engine.search(query, top_k=top_k, augment=augment)
        
        # Fall back to mock data if real data is not available
        query = query.lower()
        matches = []
        
        for paper in self.mock_data:
            paper_query = paper["query"].lower()
            if paper_query in query or query in paper_query:
                matches.append(paper)
                if len(matches) >= top_k:
                    break
        
        # If no matches found, return None
        if not matches:
            return None
        
        # If we need only one augmented result
        if top_k == 1 and augment:
            return matches[0]
        
        # Return all matches
        return matches
    
    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Add a new paper to the system
        
        For mock data: Adds to mock_data and saves to file
        For real data: Adds to RAG engine and updates the file
        """
        # Validate required fields
        if not all(key in paper for key in ["title", "abstract"]):
            return False
        
        if self.use_real_data:
            try:
                self.rag_engine.add_paper(paper)
                return True
            except Exception as e:
                print(f"Error adding paper to RAG engine: {e}")
                return False
        else:
            # For mock data, we're using a different structure
            if not all(key in paper for key in ["query", "response"]):
                return False
            
            # Check if query already exists
            if any(p["query"] == paper["query"] for p in self.mock_data):
                return False
            
            self.mock_data.append(paper)
            return self.save_mock_data()
    
    def update_paper(self, paper_id: str, new_data: Dict[str, Any]) -> bool:
        """Update an existing paper in mock data"""
        # Only implemented for mock data for now
        if self.use_real_data:
            return False
        
        for i, paper in enumerate(self.mock_data):
            if paper["query"] == paper_id:
                self.mock_data[i].update(new_data)
                return self.save_mock_data()
        return False
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper from mock data by query"""
        # Only implemented for mock data for now
        if self.use_real_data:
            return False
        
        for i, paper in enumerate(self.mock_data):
            if paper["query"] == paper_id:
                del self.mock_data[i]
                return self.save_mock_data()
        return False

# Example usage
if __name__ == "__main__":
    # Test the data manager
    manager = DataManager(use_milvus=False)
    
    # Check if real data is available
    print(f"Using real data: {manager.use_real_data}")
    
    # Search for papers
    query = "vaccine development for cancer"
    result = manager.search_papers(query, top_k=1, augment=True)
    
    if result:
        print(f"\nQuery: {query}")
        if isinstance(result, dict) and 'response' in result:
            print(f"Response: {result['response']}")
            if result.get('citation'):
                print(f"Citation: {result['citation']}")
        else:
            print(f"Found {len(result) if isinstance(result, list) else 1} matching papers")
    else:
        print(f"No papers found for query: {query}") 