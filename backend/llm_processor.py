#!/usr/bin/env python3
import json
import logging
import requests
import os
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from models import Paper

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
            paper: Paper object or dictionary
            
        Returns:
            Dictionary with formatted paper information
        """
        try:
            # Try to convert to Paper object if it's a dict
            if isinstance(paper, dict):
                try:
                    paper_obj = Paper.parse_obj(paper)
                    paper_text = paper_obj.prepare_paper_embedding_content()
                except Exception as e:
                    logger.warning(f"Error creating Paper object: {e}, using fallback method")
                    # Fallback method - extract fields manually
                    paper_text = self._fallback_paper_text(paper)
            else:
                paper_text = paper.prepare_paper_embedding_content()
            
            # Get scores
            bm25_score = paper.get('bm25_score', 0)
            similarity_score = paper.get('similarity_score', 0)
            
            # Get metadata
            if isinstance(paper, dict):
                title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
                abstract = paper.get('metadata', {}).get('abstract', '') or paper.get('abstract', '')
                year = paper.get('metadata', {}).get('year', '') or paper.get('year', '')
                authors = self._get_authors(paper)
                citation = self._get_citation(paper)
            else:
                title = paper.title
                abstract = paper.abstract if paper.abstract else ''
                year = paper.year if paper.year else ''
                authors = ', '.join(paper.authors) if hasattr(paper, 'authors') and paper.authors else ''
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
    
    def _prepare_reranking_prompt(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for LLM to rerank the papers based on relevance to the query
        
        Args:
            query: User's search query
            papers: List of preprocessed papers
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a research assistant with expertise in understanding and ranking academic papers.

USER QUERY: "{query}"

TASK:
1. First, carefully understand what the user is asking for. They want papers relevant to: {query}
2. Examine each of the following academic papers.
3. Rank the papers from most to least relevant to the user's query, based on:
   - Direct relevance to the query topic
   - Usefulness of the paper's content for addressing the query
   - Focus on the specific subject mentioned in the query
4. Select ONLY papers that are truly relevant to the query "{query}". If a paper is not relevant, do not include it.
5. Base your ranking on semantic relevance to the query, not just keyword matching.

PAPERS:
"""
        
        for i, paper in enumerate(papers, 1):
            prompt += f"""
[PAPER {i}]
- Title: {paper['title']}
- Year: {paper['year']}
- Authors: {paper['authors']}
- Abstract: {paper['abstract'][:300]}...
- Vector Similarity Score: {paper['similarity_score']}
- BM25 Score: {paper['bm25_score']}
- Content Preview: {paper['full_text'][:500]}...

"""

        prompt += f"""
IMPORTANT: Before responding, double check that your selected papers actually relate to "{query}". Papers about unrelated topics should NOT be included even if they have high similarity scores.

RESPONSE FORMAT:
Please provide your top 5 papers in order of relevance, numbered from 1 to 5. For each paper, include:
1. Paper number from the input list [PAPER X]
2. A 2-3 sentence explanation of why this paper is relevant to the query: "{query}"

Then list ONLY the paper numbers of your top 5 choices in a single line like this: "SELECTED: 3, 7, 12, 2, 9"
"""
        
        return prompt
    
    def _prepare_response_generation_prompt(self, query: str, top_papers: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for LLM to generate a comprehensive response based on top papers
        
        Args:
            query: User's search query
            top_papers: List of top ranked papers
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a research assistant with expertise in understanding and explaining academic research papers.

USER QUERY: "{query}"

TASK:
Generate a comprehensive answer to the user's query using the following academic papers as sources.
Ensure that each paper is truly relevant to the query "{query}" - if any paper seems unrelated, please ignore it.

PAPERS:
"""
        
        for i, paper in enumerate(top_papers, 1):
            prompt += f"""
[PAPER {i}]
- Title: {paper['title']}
- Year: {paper['year']}
- Authors: {paper['authors']}
- Abstract: {paper['abstract']}
- Content: {paper['full_text'][:1000]}...
- Citation: {paper['citation']}

"""

        prompt += f"""
IMPORTANT: Your response must be directly relevant to the query: "{query}". If the papers are not relevant, acknowledge this issue in your response.

RESPONSE FORMAT:
You must provide your response in a valid JSON format like this:

```json
{{
  "introduction": "Brief introduction addressing the query about {query} (1-3 sentences)",
  "papers": [
    {{
      "title": "Exact title of the paper as provided in the input",
      "summary": "A near 100-word summary of what this paper contributes to the topic"
    }},
    ...more papers...
  ],
  "conclusion": "Brief conclusion summarizing what these papers collectively tell us about the query (1-3 sentences)"
}}
```

Your response MUST be a valid JSON object with exactly these three fields: "introduction", "papers" (an array of objects with "title" and "summary"), and "conclusion".
The title field MUST exactly match the original paper title from the input.
Do not include any explanatory text outside the JSON structure.
"""
        
        return prompt
    
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
    
    async def call_llm_async(self, prompt: str) -> str:
        """
        Asynchronously call the Ollama API with a given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        # Use a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, prompt)
    
    def extract_top_papers_indices(self, reranking_result: str) -> List[int]:
        """
        Extract the indices of top papers from the LLM reranking result
        
        Args:
            reranking_result: The LLM's response to the reranking prompt
            
        Returns:
            List of paper indices (0-based)
        """
        try:
            # Look for the SELECTED: line
            if "SELECTED:" in reranking_result:
                selected_line = [line for line in reranking_result.split('\n') 
                               if "SELECTED:" in line][0]
                # Extract numbers after "SELECTED:"
                selected_part = selected_line.split("SELECTED:")[1].strip()
                # Parse the comma-separated list of paper numbers
                paper_numbers = [int(num.strip()) for num in selected_part.split(',')]
                # Convert to 0-based indices (papers were numbered from 1)
                return [num - 1 for num in paper_numbers]
            else:
                # Fallback: try to find numbered list like "1. [PAPER X]"
                paper_indices = []
                lines = reranking_result.split('\n')
                for line in lines:
                    if re.search(r'\d+\.\s*\[PAPER\s+(\d+)\]', line):
                        match = re.search(r'\[PAPER\s+(\d+)\]', line)
                        if match:
                            paper_idx = int(match.group(1)) - 1  # Convert to 0-based
                            paper_indices.append(paper_idx)
                
                # If we found at least one, return them (max 5)
                if paper_indices:
                    return paper_indices[:5]
                
                # Last resort: just look for paper numbers
                all_matches = re.findall(r'\[PAPER\s+(\d+)\]', reranking_result)
                if all_matches:
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_indices = []
                    for idx in [int(m) - 1 for m in all_matches]:
                        if idx not in seen:
                            seen.add(idx)
                            unique_indices.append(idx)
                    return unique_indices[:5]
                
                # If all parsing fails, return the first 5 papers
                logger.warning("Could not parse paper indices from LLM result, using first 5 papers")
                return list(range(min(5, len(reranking_result))))
        except Exception as e:
            logger.error(f"Error extracting top paper indices: {e}")
            # Return the first 5 papers as a fallback
            return list(range(5))
    
    def rerank_and_generate(self, query: str, papers: List[Dict[str, Any]], 
                          max_papers_for_rerank: int = 20,
                          max_papers_for_response: int = 5) -> Dict[str, Any]:
        """
        Rerank papers using LLM and generate a comprehensive response
        
        Args:
            query: User's search query
            papers: List of papers from vector search/BM25
            max_papers_for_rerank: Maximum number of papers to include in reranking
            max_papers_for_response: Maximum number of papers to include in final response
            
        Returns:
            Dictionary containing introduction, papers with summaries, and conclusion
        """
        if not papers:
            return {
                "introduction": f"I couldn't find any research papers related to '{query}'.",
                "papers": [],
                "conclusion": "Please try a different search term or check if the papers database is properly loaded."
            }
        
        # Limit papers to rerank
        papers_to_rerank = papers[:max_papers_for_rerank]
        
        # Prepare papers for LLM processing
        prepared_papers = [self._prepare_paper_for_llm(paper) for paper in papers_to_rerank]
        
        # Create reranking prompt
        reranking_prompt = self._prepare_reranking_prompt(query, prepared_papers)
        
        # Call LLM for reranking
        logger.info(f"Sending reranking prompt to LLM (length: {len(reranking_prompt)})")
        reranking_result = self.call_llm(reranking_prompt)
        
        # Extract top papers based on LLM's ranking
        top_indices = self.extract_top_papers_indices(reranking_result)
        
        # Ensure we don't exceed list bounds
        top_indices = [idx for idx in top_indices if idx < len(prepared_papers)]
        
        # Limit to max_papers_for_response
        top_indices = top_indices[:max_papers_for_response]
        
        # Get the top papers
        top_papers = [prepared_papers[idx] for idx in top_indices if idx < len(prepared_papers)]
        
        # If we couldn't parse any indices or the list is empty, use the first few papers
        if not top_papers:
            logger.warning("No valid paper indices found, using top papers by vector/BM25 score")
            top_papers = prepared_papers[:max_papers_for_response]
        
        # Create a mapping of paper title to original paper
        title_to_paper = {}
        for i, paper in enumerate(papers_to_rerank):
            # Get title from either metadata or direct field
            title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
            if title:
                title_to_paper[title] = paper
        
        # Prepare response generation prompt
        response_prompt = self._prepare_response_generation_prompt(query, top_papers)
        
        # Call LLM for response generation
        logger.info(f"Sending response generation prompt to LLM (length: {len(response_prompt)})")
        llm_response = self.call_llm(response_prompt)
        
        # Parse the JSON response
        parsed_response = self.parse_llm_response(llm_response)
        
        # Attach original paper data to each paper in the response
        for paper_summary in parsed_response["papers"]:
            title = paper_summary.get("title", "")
            if title in title_to_paper:
                # Attach the original paper data
                paper_summary["paper_data"] = title_to_paper[title]
            else:
                # Try to find a close match
                for paper_title, paper_obj in title_to_paper.items():
                    # Check if the title is contained within or vice versa
                    if (title.lower() in paper_title.lower() or 
                        paper_title.lower() in title.lower()):
                        paper_summary["paper_data"] = paper_obj
                        break
        
        return parsed_response
    
    async def rerank_and_generate_async(self, query: str, papers: List[Dict[str, Any]], 
                                    max_papers_for_rerank: int = 20,
                                    max_papers_for_response: int = 5) -> Dict[str, Any]:
        """
        Asynchronous version of rerank_and_generate
        
        Args:
            query: User's search query
            papers: List of papers from vector search/BM25
            max_papers_for_rerank: Maximum number of papers to include in reranking
            max_papers_for_response: Maximum number of papers to include in final response
            
        Returns:
            Dictionary containing introduction, papers with summaries, and conclusion
        """
        if not papers:
            return {
                "introduction": f"I couldn't find any research papers related to '{query}'.",
                "papers": [],
                "conclusion": "Please try a different search term or check if the papers database is properly loaded."
            }
            
        # Limit papers to rerank
        papers_to_rerank = papers[:max_papers_for_rerank]
        
        # Prepare papers for LLM processing
        prepared_papers = [self._prepare_paper_for_llm(paper) for paper in papers_to_rerank]
        
        # Create reranking prompt
        reranking_prompt = self._prepare_reranking_prompt(query, prepared_papers)
        
        # Call LLM for reranking
        logger.info(f"Sending reranking prompt to LLM (length: {len(reranking_prompt)})")
        reranking_result = await self.call_llm_async(reranking_prompt)
        
        # Extract top papers based on LLM's ranking
        top_indices = self.extract_top_papers_indices(reranking_result)
        
        # Ensure we don't exceed list bounds
        top_indices = [idx for idx in top_indices if idx < len(prepared_papers)]
        
        # Limit to max_papers_for_response
        top_indices = top_indices[:max_papers_for_response]
        print(top_indices)
        
        # Get the top papers
        top_papers = [prepared_papers[idx] for idx in top_indices if idx < len(prepared_papers)]
        
        # If we couldn't parse any indices or the list is empty, use the first few papers
        if not top_papers:
            logger.warning("No valid paper indices found, using top papers by vector/BM25 score")
            top_papers = prepared_papers[:max_papers_for_response]
        
        # Create a mapping of paper title to original paper
        title_to_paper = {}
        for i, paper in enumerate(papers_to_rerank):
            # Get title from either metadata or direct field
            title = paper.get('metadata', {}).get('title', '') or paper.get('title', '')
            if title:
                title_to_paper[title] = paper
        
        # Prepare response generation prompt
        response_prompt = self._prepare_response_generation_prompt(query, top_papers)
        
        # Call LLM for response generation
        logger.info(f"Sending response generation prompt to LLM (length: {len(response_prompt)})")
        llm_response = await self.call_llm_async(response_prompt)
        
        # Parse the JSON response
        parsed_response = self.parse_llm_response(llm_response)
        
        # Attach original paper data to each paper in the response
        for paper_summary in parsed_response["papers"]:
            title = paper_summary.get("title", "")
            if title in title_to_paper:
                # Attach the original paper data
                paper_summary["paper_data"] = title_to_paper[title]
            else:
                # Try to find a close match
                for paper_title, paper_obj in title_to_paper.items():
                    # Check if the title is contained within or vice versa
                    if (title.lower() in paper_title.lower() or 
                        paper_title.lower() in title.lower()):
                        paper_summary["paper_data"] = paper_obj
                        break
        
        return parsed_response

    def parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response from JSON format to a structured dictionary
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            Structured dictionary containing introduction, papers, and conclusion
        """
        # Define default response structure
        default_response = {
            "introduction": "I've analyzed some relevant papers for your query.",
            "papers": [],
            "conclusion": "These papers provide insights into your topic of interest."
        }
        
        try:
            # First, try to find JSON content within the response
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', llm_response)
            if json_match:
                json_content = json_match.group(1)
            else:
                # If no JSON code block found, try to find anything that looks like JSON
                json_match = re.search(r'(\{[\s\S]*\})', llm_response)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    logger.warning("No JSON content found in LLM response")
                    return default_response
            
            # Parse the JSON content
            parsed_response = json.loads(json_content)
            
            # Validate the structure
            if not isinstance(parsed_response, dict):
                logger.warning("JSON response is not a dictionary")
                return default_response
            
            # Check for required fields
            if "introduction" not in parsed_response:
                logger.warning("JSON response missing 'introduction' field")
                parsed_response["introduction"] = default_response["introduction"]
            
            if "papers" not in parsed_response or not isinstance(parsed_response["papers"], list):
                logger.warning("JSON response missing or invalid 'papers' field")
                parsed_response["papers"] = default_response["papers"]
            
            if "conclusion" not in parsed_response:
                logger.warning("JSON response missing 'conclusion' field")
                parsed_response["conclusion"] = default_response["conclusion"]
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM response: {e}")
            return default_response
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return default_response


# Example usage
if __name__ == "__main__":
    import argparse
    from vector_search import VectorSearch
    
    parser = argparse.ArgumentParser(description='Test LLM paper reranking and response generation')
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
    
    # Rerank and generate response
    response = llm_processor.rerank_and_generate(
        query=args.query,
        papers=papers,
        max_papers_for_rerank=args.limit,
        max_papers_for_response=args.top_k
    )
    
    print("\n=== GENERATED RESPONSE ===\n")
    print(response) 