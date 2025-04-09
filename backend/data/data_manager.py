import json
import os
from typing import List, Dict, Any, Optional

class DataManager:
    """Utility class to manage paper data"""
    
    def __init__(self, data_file: str = "paper_data.json"):
        """Initialize with path to data file"""
        self.data_file = os.path.join(os.path.dirname(__file__), data_file)
        self.data = self.load_data()
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load paper data from JSON file"""
        try:
            with open(self.data_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading data: {e}")
            return []
    
    def save_data(self) -> bool:
        """Save paper data to JSON file"""
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get all papers"""
        return self.data
    
    def get_all_topics(self) -> List[str]:
        """Get all available paper topics"""
        return [paper["query"] for paper in self.data]
    
    def search_papers(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for papers matching the query"""
        query = query.lower()
        for paper in self.data:
            if paper["query"] in query:
                return paper
        return None
    
    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """Add a new paper to the data"""
        # Validate required fields
        if not all(key in paper for key in ["query", "response"]):
            return False
        
        # Check if query already exists
        if any(p["query"] == paper["query"] for p in self.data):
            return False
        
        self.data.append(paper)
        return self.save_data()
    
    def update_paper(self, query: str, new_data: Dict[str, Any]) -> bool:
        """Update an existing paper"""
        for i, paper in enumerate(self.data):
            if paper["query"] == query:
                self.data[i].update(new_data)
                return self.save_data()
        return False
    
    def delete_paper(self, query: str) -> bool:
        """Delete a paper by query"""
        for i, paper in enumerate(self.data):
            if paper["query"] == query:
                del self.data[i]
                return self.save_data()
        return False

# Example usage
if __name__ == "__main__":
    # Test the data manager
    manager = DataManager()
    
    # Print all available topics
    print("Available topics:", manager.get_all_topics())
    
    # Example of adding a new paper
    new_paper = {
        "query": "graph neural networks",
        "response": "Graph Neural Networks (GNNs) are powerful models for processing graph-structured data. A key paper is 'The Graph Neural Network Model' by Scarselli et al.",
        "citation": "Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2008). The graph neural network model. IEEE Transactions on Neural Networks, 20(1), 61-80."
    }
    
    if manager.add_paper(new_paper):
        print(f"Added new paper on '{new_paper['query']}'")
    else:
        print("Failed to add new paper (might already exist)")
    
    # Print updated topics
    print("Updated topics:", manager.get_all_topics()) 