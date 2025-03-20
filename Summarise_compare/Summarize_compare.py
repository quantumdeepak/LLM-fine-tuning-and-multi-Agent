import os
import fitz  # PyMuPDF
import json
import requests
import time
from typing import List, Dict, Any
import re
from collections import defaultdict

#######################################################################
# CONFIGURATION - EDIT THESE VALUES
#######################################################################

# Input/Output Settings
PAPERS_DIRECTORY = "/mnt/DATA/Glucoma/LLM/Summarise_compare/pdf"           # Directory containing PDF files
OUTPUT_JSON_FILE = "/mnt/DATA/Glucoma/LLM/Summarise_compare/output/paper_analysis_results.json"  # Main output JSON file
COMPARISON_OUTPUT_FILE = "/mnt/DATA/Glucoma/LLM/Summarise_compare/output/paper_comparison.txt"   # Text file for comparison results

# Processing Limits
MAX_PAPERS = None                     # Maximum number of papers to process (None for all)
MAX_CONTEXT_TOKENS = 128000           # Maximum tokens to process (128K context window)

# Ollama API Settings
OLLAMA_API_URL = "http://localhost:11434/api"  # Ollama API URL
OLLAMA_MODEL = "gemma3:12b"           # Model to use for analysis

# Summary Generation Settings
SUMMARY_TEMPERATURE = 0.1             # Lower = more deterministic summaries
SUMMARY_TOP_P = 0.9                   # Nucleus sampling parameter for summaries
PAPER_CONTENT_PREVIEW_LENGTH = 10000  # How many characters of paper to include in prompt

# Comparison Settings
COMPARISON_TEMPERATURE = 0.2          # Temperature for comparison generation
COMPARISON_TOP_P = 0.95               # Nucleus sampling for comparison
SUMMARY_PREVIEW_LENGTH = 1000         # How many characters of each summary to include in comparison

# API Request Settings
API_REQUEST_DELAY = 1                 # Delay between API calls (seconds)

#######################################################################
# CODE - NO NEED TO MODIFY BELOW THIS LINE
#######################################################################

class PaperAnalyzer:
    def __init__(self):
        """Initialize the paper analyzer with configuration parameters."""
        self.ollama_api_url = OLLAMA_API_URL
        self.model = OLLAMA_MODEL
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.current_token_count = 0
        self.papers = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def estimate_token_count(self, text: str) -> int:
        """Roughly estimate token count based on whitespace and punctuation."""
        # Simple estimation: ~4 characters per token on average
        return len(text) // 4
    
    def load_papers_from_directory(self, directory: str, max_papers: int = None) -> List[Dict[str, Any]]:
        """Load papers from a directory up to context window limit or max_papers."""
        self.papers = []
        self.current_token_count = 0
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        if max_papers:
            pdf_files = pdf_files[:max_papers]
            
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            paper_text = self.extract_text_from_pdf(pdf_path)
            
            # Skip empty papers
            if not paper_text.strip():
                print(f"Skipping {pdf_file} - failed to extract text or empty file")
                continue
                
            paper_tokens = self.estimate_token_count(paper_text)
            
            # Check if adding this paper would exceed the context window
            if self.current_token_count + paper_tokens > self.max_context_tokens:
                print(f"Reached context window limit of {self.max_context_tokens} tokens. Stopping.")
                break
                
            paper_info = {
                "filename": pdf_file,
                "text": paper_text,
                "tokens": paper_tokens,
                "metadata": self.extract_metadata(paper_text)
            }
            
            self.papers.append(paper_info)
            self.current_token_count += paper_tokens
            print(f"Loaded {pdf_file} ({paper_tokens} tokens). Total: {self.current_token_count}/{self.max_context_tokens}")
            
        return self.papers
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract basic metadata from paper text."""
        # This is a simple implementation; real-world usage might need more sophisticated extraction
        metadata = {}
        
        # Extract title (assume it's in the first few lines)
        first_lines = text.split("\n")[:10]
        title_candidates = [line for line in first_lines if len(line) > 20 and len(line) < 200]
        if title_candidates:
            metadata["title"] = title_candidates[0].strip()
        else:
            metadata["title"] = "Unknown Title"
        
        # Try to extract abstract
        abstract_match = re.search(r"abstract(.*?)(?:introduction|keywords)", text.lower(), re.DOTALL)
        if abstract_match:
            metadata["abstract"] = abstract_match.group(1).strip()
        else:
            # Fallback to the first substantial paragraph
            paragraphs = [p for p in text.split("\n\n") if len(p) > 150]
            if paragraphs:
                metadata["abstract"] = paragraphs[0].strip()
            else:
                metadata["abstract"] = "Abstract not found"
                
        return metadata
    
    def generate_summary(self, paper_index: int) -> Dict[str, str]:
        """Generate a summary for a specific paper using Ollama."""
        if not self.papers or paper_index >= len(self.papers):
            return {"error": "Invalid paper index"}
            
        paper = self.papers[paper_index]
        
        # Create a prompt for summarization
        prompt = f"""Please provide a detailed summary of the following academic paper:
Title: {paper['metadata'].get('title', 'Unknown')}

Here's the content of the paper:
{paper['text'][:PAPER_CONTENT_PREVIEW_LENGTH]}...

Generate a comprehensive summary that includes:
1. The main research question or objective
2. The methodology used
3. Key findings and results
4. Main conclusions and implications
5. Any limitations mentioned
"""

        try:
            # Call Ollama API for summarization
            response = requests.post(
                f"{self.ollama_api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": SUMMARY_TEMPERATURE,
                        "top_p": SUMMARY_TOP_P
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "Failed to generate summary")
                
                # Update the paper with its summary
                self.papers[paper_index]["summary"] = summary
                return {"filename": paper["filename"], "summary": summary}
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Error generating summary: {e}"}
    
    def generate_all_summaries(self) -> None:
        """Generate summaries for all loaded papers."""
        for i in range(len(self.papers)):
            print(f"Generating summary for paper {i+1}/{len(self.papers)}: {self.papers[i]['filename']}")
            result = self.generate_summary(i)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Summary generated successfully")
            # Add a delay to avoid overwhelming the API
            time.sleep(API_REQUEST_DELAY)
    
    def compare_papers(self) -> Dict[str, Any]:
        """Compare all papers and identify relationships and differences."""
        if len(self.papers) < 2:
            return {"error": "Need at least 2 papers to compare"}
            
        # Extract summaries for comparison
        paper_summaries = []
        for paper in self.papers:
            if "summary" not in paper:
                print(f"Warning: No summary for {paper['filename']}. Run generate_all_summaries first.")
                return {"error": "Missing summaries"}
                
            paper_summaries.append({
                "filename": paper["filename"],
                "title": paper["metadata"].get("title", "Unknown"),
                "summary": paper["summary"]
            })
            
        # Create comparison prompt
        papers_text = ""
        for i, paper in enumerate(paper_summaries):
            papers_text += f"\nPaper {i+1}: {paper['title']}\nSummary: {paper['summary'][:SUMMARY_PREVIEW_LENGTH]}...\n"
            
        comparison_prompt = f"""Compare the following {len(paper_summaries)} academic papers:
{papers_text}

Please provide a detailed comparison that:
1. Identifies common themes, approaches, or findings across the papers
2. Highlights key differences in methodology or conclusions
3. Analyzes how the papers relate to each other (do they build on similar work, contradict each other, etc.)
4. Suggests potential areas for future research based on gaps or limitations identified
"""

        try:
            # Call Ollama API for comparison
            response = requests.post(
                f"{self.ollama_api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": comparison_prompt,
                    "stream": False,
                    "options": {
                        "temperature": COMPARISON_TEMPERATURE,
                        "top_p": COMPARISON_TOP_P
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                comparison = result.get("response", "Failed to generate comparison")
                return {
                    "papers_compared": len(paper_summaries),
                    "comparison_result": comparison
                }
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Error generating comparison: {e}"}
    
    def save_results(self, output_file: str) -> None:
        """Save all results to a JSON file."""
        results = {
            "meta": {
                "total_papers": len(self.papers),
                "total_tokens": self.current_token_count,
                "model_used": self.model
            },
            "papers": []
        }
        
        for paper in self.papers:
            paper_data = {
                "filename": paper["filename"],
                "metadata": paper["metadata"],
                "token_count": paper["tokens"]
            }
            
            if "summary" in paper:
                paper_data["summary"] = paper["summary"]
                
            results["papers"].append(paper_data)
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
            
    def save_comparison(self, comparison_result: Dict[str, Any], output_file: str) -> None:
        """Save comparison results to a separate text file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("PAPER COMPARISON RESULTS\n")
                f.write("========================\n\n")
                f.write(f"Number of papers compared: {comparison_result.get('papers_compared', 0)}\n\n")
                f.write(comparison_result.get('comparison_result', 'No comparison available'))
            print(f"Comparison results saved to {output_file}")
        except Exception as e:
            print(f"Error saving comparison results: {e}")

# Main execution
def run_analysis():
    print("\nPaper Analysis System")
    print("====================")
    print(f"Reading papers from: {PAPERS_DIRECTORY}")
    print(f"Using model: {OLLAMA_MODEL}")
    print(f"Max context window: {MAX_CONTEXT_TOKENS} tokens")
    print(f"Max papers: {MAX_PAPERS if MAX_PAPERS else 'All'}")
    print("====================\n")
    
    analyzer = PaperAnalyzer()
    
    # Load papers
    print("Loading papers...")
    papers = analyzer.load_papers_from_directory(PAPERS_DIRECTORY, MAX_PAPERS)
    print(f"Loaded {len(papers)} papers")
    
    if not papers:
        print("No papers were loaded. Exiting.")
        return
        
    # Generate summaries
    print("Generating summaries for all papers...")
    analyzer.generate_all_summaries()
    
    # Compare papers if there are at least 2
    comparison_result = None
    if len(papers) >= 2:
        print("Comparing papers...")
        comparison_result = analyzer.compare_papers()
        if "error" in comparison_result:
            print(f"Error during comparison: {comparison_result['error']}")
        else:
            print("Comparison completed")
            print("\n--- PAPER COMPARISON PREVIEW ---")
            preview = comparison_result["comparison_result"][:500] + "..." if len(comparison_result["comparison_result"]) > 500 else comparison_result["comparison_result"]
            print(preview)
            print("--- END OF PREVIEW ---\n")
            
            # Save comparison to a separate file
            analyzer.save_comparison(comparison_result, COMPARISON_OUTPUT_FILE)
    
    # Save all results to JSON
    analyzer.save_results(OUTPUT_JSON_FILE)
    print(f"Analysis completed. Results saved to {OUTPUT_JSON_FILE}")

# Run the analysis when this script is executed
if __name__ == "__main__":
    run_analysis()