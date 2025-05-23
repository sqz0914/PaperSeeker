#!/usr/bin/env python3
import json
import logging
import requests
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from models import Paper
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # Llama 3.2 model

class LLMProcessor:
    """
    Class for processing search results using LLM to rerank and generate informative responses
    """
    
    def __init__(self, model_name: str = MODEL_NAME, ollama_url: str = OLLAMA_API_URL):
        """Initialize the LLM processor with model settings"""
        self.model_name = model_name
        self.ollama_url = ollama_url
        logger.info(f"LLM Processor initialized with model: {model_name}")
    
    def _prepare_paper_for_llm(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format relevant information from a paper for LLM processing
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Dictionary with formatted paper information
        """
        try:
            # Extract paper text
            try:
                paper_obj = Paper.parse_obj(paper)
                paper_text = paper_obj.prepare_paper_embedding_content()
            except Exception as e:
                logger.warning(f"Error creating Paper object: {e}, using fallback method")
                # Fallback method - extract fields manually
                paper_text = self._fallback_paper_text(paper)
            
            # Get scores
            bm25_score = paper.get('bm25_score', 0)
            similarity_score = paper.get('similarity_score', 0)
            
            # Get metadata
            title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
            abstract = paper.get('metadata', {}).get('abstract', '') or paper.get('abstract', '')
            year = paper.get('metadata', {}).get('year', '') or paper.get('year', '')
            authors = self._get_authors(paper)
            citation = self._get_citation(paper)
            
            return {
                "title": title,
                "year": year,
                "authors": authors,
                "abstract": abstract,
                "full_text": paper_text,
                "bm25_score": bm25_score,
                "similarity_score": similarity_score,
                "citation": citation
            }
        except Exception as e:
            logger.error(f"Error preparing paper for LLM: {e}")
            # Return minimal information to avoid breaking the pipeline
            return {
                "title": paper.get('title', 'Unknown title'),
                "abstract": paper.get('abstract', ''),
                "full_text": "",
                "bm25_score": paper.get('bm25_score', 0),
                "similarity_score": paper.get('similarity_score', 0),
                "citation": ""
            }
    
    def _fallback_paper_text(self, paper: Dict[str, Any]) -> str:
        """Create a formatted text representation when Paper object can't be created"""
        sections = []
        
        # Extract fields with fallbacks
        title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
        if title:
            sections.append(f"TITLE: {title}")
        
        abstract = paper.get('metadata', {}).get('abstract', '') or paper.get('abstract', '')
        if abstract:
            sections.append(f"ABSTRACT: {abstract}")
        
        year = paper.get('metadata', {}).get('year', '') or paper.get('year', '')
        if year:
            sections.append(f"YEAR: {year}")
        
        authors = self._get_authors(paper)
        if authors:
            sections.append(f"AUTHORS: {authors}")
        
        text = paper.get('metadata', {}).get('text', '') or paper.get('text', '')
        if text:
            sections.append(f"CONTENT: {text[:5000]}")  # Limit content length
        
        return "\n\n".join(sections)
    
    def _get_authors(self, paper: Dict[str, Any]) -> str:
        """Extract author information from a paper dict with fallbacks"""
        if paper.get('metadata', {}).get('authors'):
            authors = paper['metadata']['authors']
            if isinstance(authors, list):
                # Check if authors are objects or strings
                if authors and isinstance(authors[0], dict):
                    return ", ".join([a.get('name', '') for a in authors])
                else:
                    return ", ".join(authors)
            else:
                return str(authors)
        elif paper.get('authors'):
            authors = paper['authors']
            if isinstance(authors, list):
                if authors and isinstance(authors[0], dict):
                    return ", ".join([a.get('name', '') for a in authors])
                else:
                    return ", ".join(authors)
            else:
                return str(authors)
        return ""
    
    def _get_citation(self, paper: Dict[str, Any]) -> str:
        """Extract citation information from a paper dict with fallbacks"""
        # Try to get external IDs
        if paper.get('metadata', {}).get('external_ids'):
            ext_ids = []
            for ext_id in paper['metadata']['external_ids']:
                if isinstance(ext_id, dict) and ext_id.get('source') and ext_id.get('id'):
                    ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                elif isinstance(ext_id, str):
                    ext_ids.append(ext_id)
            if ext_ids:
                return ", ".join(ext_ids)
        
        # Try direct external_ids
        elif paper.get('external_ids'):
            ext_ids = []
            for ext_id in paper['external_ids']:
                if isinstance(ext_id, dict) and ext_id.get('source') and ext_id.get('id'):
                    ext_ids.append(f"{ext_id['source']}: {ext_id['id']}")
                elif isinstance(ext_id, str):
                    ext_ids.append(ext_id)
            if ext_ids:
                return ", ".join(ext_ids)
        
        # Use ID as fallback
        return f"ID: {paper.get('id', 'unknown')}"
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the Ollama API with a given prompt and return the response
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for more factual responses
                        "num_predict": 2048  # Limit output to avoid excessive responses
                    }
                },
                timeout=60  # 60 second timeout for longer processing
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                return "Error generating response from language model."
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error connecting to language model service."
    
    def _prepare_single_paper_prompt(self, query: str, paper: Dict[str, Any]) -> str:
        """
        Create a prompt for LLM to generate a summary for a single paper
        
        Args:
            query: User's search query
            paper: Paper data
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a research assistant with expertise in understanding and explaining academic research papers.

USER QUERY: "{query}"

TASK:
Generate a concise summary of the following paper that explains how it relates to the query: "{query}".
Focus only on the aspects of the paper that are directly relevant to the query.

PAPER:
- Title: {paper['title']}
- Year: {paper['year']}
- Authors: {paper['authors']}
- Abstract: {paper['abstract']}
- Content: {paper['full_text']}...

RESPONSE FORMAT:
Provide a concise summary (around 100-150 words) explaining what this paper contributes to the topic of "{query}".
Focus on the key findings, methodologies, and conclusions that are most relevant to the query.
Your response should be direct and informative without any preamble.
"""
        return prompt
        
    def _prepare_conclusion_prompt(self, query: str, papers_with_summaries: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for LLM to generate a conclusion based on paper summaries
        
        Args:
            query: User's search query
            papers_with_summaries: List of papers with their summaries
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a research assistant with expertise in understanding and explaining academic research papers.

USER QUERY: "{query}"

TASK:
Generate a brief introduction and conclusion based on the following paper summaries.
These papers were selected because they are relevant to the query: "{query}".

PAPER SUMMARIES:
"""
        
        for i, paper in enumerate(papers_with_summaries, 1):
            prompt += f"""
[PAPER {i}]
- Title: {paper['title']}
- Year: {paper['year']}
- Authors: {paper['authors']}
- Summary: {paper['summary']}

"""

        prompt += f"""
RESPONSE FORMAT:
Provide a response in the following JSON format:

```json
{{
  "introduction": "A brief introduction that frames the query '{query}' and what these papers collectively address (2-3 sentences)",
  "conclusion": "A thoughtful conclusion that synthesizes the main insights from these papers about '{query}' (3-5 sentences)"
}}
```

Your response MUST be in valid JSON format with exactly these two fields. Make sure the content is directly relevant to the query.
"""
        return prompt
    
    def _prepare_relevance_prompt(self, query: str, paper: Dict[str, Any]) -> str:
        """
        Create a prompt for LLM to determine if a paper is relevant to the query
        
        Args:
            query: User's search query
            paper: Paper data
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a research assistant with expertise in understanding academic research papers.

USER QUERY: "{query}"

TASK:
Determine whether the following paper might be relevant or useful to the query: "{query}".
Be inclusive rather than exclusive - papers that are partially related could still provide useful insights.
Consider both direct relevance and indirect connections that might still be valuable to the user.

PAPER:
- Title: {paper['title']}
- Year: {paper['year']}
- Authors: {paper['authors']}
- Abstract: {paper['abstract']}
- Content Preview: {paper['full_text']}

RESPONSE FORMAT:
Provide a JSON response with two fields:
1. "relevant": true or false based on whether this paper is relevant to the query. Be lenient - if the paper might contain useful information even peripherally related to the query, consider it relevant.
2. "reason": A brief explanation of why the paper is or is not relevant (2-3 sentences)

Your response should be in this exact format:
```json
{{"relevant": true/false, "reason": "Your explanation here"}}
```
"""
        return prompt
        
    def _check_paper_relevance(self, query: str, paper: Dict[str, Any], paper_index: int) -> Dict[str, Any]:
        """
        Check if a paper is relevant to the query
        
        Args:
            query: User's search query
            paper: Paper data
            paper_index: Original index of the paper in the search results
            
        Returns:
            Dictionary with paper, relevance result and original index
        """
        # Create a prompt for relevance check
        relevance_prompt = self._prepare_relevance_prompt(query, paper)
        
        # Call LLM to check relevance
        relevance_response = self.call_llm(relevance_prompt)
        
        # Parse the JSON response
        try:
            # Extract JSON content
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', relevance_response)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_match = re.search(r'(\{[\s\S]*\})', relevance_response)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    logger.warning(f"No JSON content found in relevance check for paper: {paper['title']}")
                    return {
                        "paper": paper,
                        "is_relevant": False,
                        "reason": "Failed to parse relevance check",
                        "original_index": paper_index
                    }
            
            relevance_result = json.loads(json_content)
            is_relevant = relevance_result.get('relevant', False)
            reason = relevance_result.get('reason', '')
            
            logger.info(f"Paper '{paper['title']}' relevance: {is_relevant} - {reason}")
            
            return {
                "paper": paper,
                "is_relevant": is_relevant,
                "reason": reason,
                "original_index": paper_index
            }
                
        except Exception as e:
            logger.error(f"Error parsing relevance check JSON for paper '{paper['title']}': {e}")
            return {
                "paper": paper,
                "is_relevant": False,
                "reason": f"Error: {str(e)}",
                "original_index": paper_index
            }
        
    def _generate_paper_summary(self, query: str, paper: Dict[str, Any], paper_index: int) -> Dict[str, Any]:
        """
        Generate a summary for a specific paper
        
        Args:
            query: User's search query
            paper: Paper data
            paper_index: Original index of the paper
            
        Returns:
            Dictionary with paper and its summary
        """
        try:
            logger.info(f"Generating summary for paper: {paper['title']}")
            
            # Create a prompt for this specific paper
            paper_prompt = self._prepare_single_paper_prompt(query, paper)
            
            # Call LLM for paper summary
            paper_summary = self.call_llm(paper_prompt)
            
            # Create a copy with summary
            paper_with_summary = paper.copy()
            paper_with_summary['summary'] = paper_summary.strip()
            paper_with_summary['original_index'] = paper_index
            
            return paper_with_summary
        except Exception as e:
            logger.error(f"Error generating summary for paper '{paper['title']}': {e}")
            # Return paper with error message as summary
            paper_with_summary = paper.copy()
            paper_with_summary['summary'] = f"Error generating summary: {str(e)}"
            paper_with_summary['original_index'] = paper_index
            return paper_with_summary

    def filter_and_generate(self, query: str, papers: List[Dict[str, Any]], 
                          max_papers_for_filter: int = 20,
                          max_papers_for_response: int = 5,
                          max_workers: int = 20) -> Dict[str, Any]:
        """
        Filter papers for relevance, generate summaries for relevant papers, and create a comprehensive response
        
        Args:
            query: User's search query
            papers: List of papers from vector search/BM25
            max_papers_for_filter: Maximum number of papers to check for relevance
            max_papers_for_response: Maximum number of papers to include in final response
            max_workers: Maximum number of concurrent workers for parallel processing
            
        Returns:
            Dictionary containing introduction, papers with summaries, and conclusion
        """
        if not papers:
            return {
                "introduction": f"I couldn't find any research papers related to '{query}'.",
                "papers": [],
                "conclusion": "Please try a different search term or check if the papers database is properly loaded."
            }
        
        # Limit papers to check for relevance
        papers_to_check = papers[:max_papers_for_filter]
        
        # Prepare papers for LLM processing
        prepared_papers = [self._prepare_paper_for_llm(paper) for paper in papers_to_check]
        
        # Create a mapping of paper title to original paper
        title_to_paper = {}
        for i, paper in enumerate(papers_to_check):
            # Get title from either metadata or direct field
            title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
            if title:
                title_to_paper[title] = paper
        
        # Filter papers for relevance in parallel
        relevant_papers_results = []
        
        logger.info(f"Starting parallel relevance checks for {len(prepared_papers)} papers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all relevance check tasks
            future_to_paper = {
                executor.submit(self._check_paper_relevance, query, paper, i): (i, paper) 
                for i, paper in enumerate(prepared_papers)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_paper):
                result = future.result()
                if result["is_relevant"]:
                    paper = result["paper"]
                    paper['relevance_reason'] = result["reason"]
                    relevant_papers_results.append(result)
                
        # Sort by original index to respect the initial ranking from the search
        relevant_papers_results.sort(key=lambda x: x["original_index"])
        
        # Extract the paper objects from the results
        relevant_papers = [result["paper"] for result in relevant_papers_results]
        
        # If we have no relevant papers, return a message
        if not relevant_papers:
            return {
                "introduction": f"I couldn't find any research papers directly relevant to '{query}'.",
                "papers": [],
                "conclusion": "Please try a different search term or broaden your query."
            }
            
        logger.info(f"Found {len(relevant_papers)} relevant papers for query: '{query}'")
        
        # Limit to max_papers_for_response
        relevant_papers = relevant_papers[:max_papers_for_response]
        
        # Process each relevant paper to get summaries in parallel
        papers_with_summaries = []
        
        logger.info(f"Starting parallel summary generation for {len(relevant_papers)} papers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all summary generation tasks
            future_to_summary = {
                executor.submit(self._generate_paper_summary, query, paper, i): (i, paper) 
                for i, paper in enumerate(relevant_papers)
            }
            
            # Process completed tasks as they finish
            summary_results = []
            for future in as_completed(future_to_summary):
                paper_with_summary = future.result()
                
                # Add original paper data
                if paper_with_summary['title'] in title_to_paper:
                    paper_with_summary['paper_data'] = title_to_paper[paper_with_summary['title']]
                else:
                    # Try to find a close match
                    for paper_title, paper_obj in title_to_paper.items():
                        if (paper_with_summary['title'].lower() in paper_title.lower() or 
                            paper_title.lower() in paper_with_summary['title'].lower()):
                            paper_with_summary['paper_data'] = paper_obj
                            break
                
                summary_results.append(paper_with_summary)
        
        # Sort by original index to maintain the order
        summary_results.sort(key=lambda x: x.get('original_index', 0))
        papers_with_summaries = summary_results
        
        # Generate introduction and conclusion based on all paper summaries
        conclusion_prompt = self._prepare_conclusion_prompt(query, papers_with_summaries)
        
        logger.info("Generating introduction and conclusion")
        intro_conclusion_response = self.call_llm(conclusion_prompt)
        
        # Try to parse the JSON response
        try:
            # First, try to find JSON content within the response
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', intro_conclusion_response)
            if json_match:
                json_content = json_match.group(1)
            else:
                # If no JSON code block found, try to find anything that looks like JSON
                json_match = re.search(r'(\{[\s\S]*\})', intro_conclusion_response)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    logger.warning("No JSON content found in LLM conclusion response")
                    json_content = '{}'
            
            intro_conclusion = json.loads(json_content)
        except Exception as e:
            logger.error(f"Error parsing conclusion JSON: {e}")
            intro_conclusion = {}
        
        # Format the final response
        response = {
            "introduction": intro_conclusion.get("introduction", f"Here are some papers related to '{query}'."),
            "papers": [],
            "conclusion": intro_conclusion.get("conclusion", f"These papers provide insights into the topic of '{query}'.")
        }
        
        # Add papers with their summaries
        for paper in papers_with_summaries:
            paper_entry = {
                "title": paper["title"],
                "summary": paper["summary"],
                "paper_data": paper.get("paper_data", {})
            }
            response["papers"].append(paper_entry)
        
        return response


# Example usage
if __name__ == "__main__":
    import argparse
    from vector_search import VectorSearch
    
    parser = argparse.ArgumentParser(description='Test LLM paper filtering and response generation')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--limit', type=int, default=20, help='Number of papers to retrieve initially')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top papers for final response')
    
    args = parser.parse_args()
    
    # Initialize vector search
    vector_search = VectorSearch()
    
    # Search for papers
    papers = vector_search.search(args.query, limit=args.limit)
    
    print(f"Retrieved {len(papers)} papers for query: '{args.query}'")
    
    # Initialize LLM processor
    llm_processor = LLMProcessor()
    
    # Filter and generate response
    response = llm_processor.filter_and_generate(
        query=args.query,
        papers=papers,
        max_papers_for_filter=args.limit,
        max_papers_for_response=args.top_k
    )
    
    print("\n=== GENERATED RESPONSE ===\n")
    print(response) 