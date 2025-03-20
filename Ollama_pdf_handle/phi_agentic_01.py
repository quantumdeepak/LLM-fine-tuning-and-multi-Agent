import os
import sys
from typing import List, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import pickle
import numpy as np
import torch
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import phi components
try:
    from phi.agent import Agent
    from phi.tools.arxiv_toolkit import ArxivToolkit as ArxivTools
    from phi.tools.duckduckgo import DuckDuckGo
    from phi.tools.github import GithubTools
    from phi.tools.tool import Tool
    from phi.tools.googlesearch import GoogleSearch
    from phi.model.ollama import Ollama
    from phi.assistant import Assistant
    import ollama
    from transformers import AutoTokenizer, AutoModel
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install the required packages:")
    print("pip install phidata ollama torch transformers selenium webdriver-manager beautifulsoup4 python-dotenv")
    sys.exit(1)

# Configuration
CONFIG = {
    "text_directory": "/mnt/DATA/Glucoma/LLM/Ollama_pdf_handle/Text_pdf",            # Directory with text files
    "vector_db_path": "vector_db",                # Directory to store vector database
    "ollama_model": "gemma3:12b",                 # Ollama model to use
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # HF model for embeddings
    "embedding_dimension": 384,                   # Embedding dimension for the model
    "top_k": 5,                                   # Number of documents to retrieve per query
    "search_depth": 5,                            # Number of pages to extract content from
}

# Get available Ollama models
def get_available_models():
    try:
        # Using the ollama command line interface through subprocess instead of the API
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        models = []
        if len(lines) > 1:  # Skip header row
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
        
        return models
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        # Fall back to direct API call
        try:
            response = ollama.list()
            if 'models' in response and response['models']:
                return [model.get('name', '').split(':')[0] + ':' + model.get('name', '').split(':')[1] 
                        for model in response['models'] if model.get('name')]
        except Exception as e:
            logger.error(f"Error with fallback to Ollama API: {e}")
        return []

# Create directory if it doesn't exist
def ensure_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

# ----- Embedding and Vector DB Functions -----

# Mean Pooling for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize embedding model
def initialize_embedding_model():
    logger.info(f"Loading embedding model: {CONFIG['embedding_model']}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['embedding_model'])
        model = AutoModel.from_pretrained(CONFIG['embedding_model'])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if device.type == 'cuda':
            logger.info(f"Embedding model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Embedding model loaded on CPU")
        
        def get_embeddings(texts):
            if isinstance(texts, str):
                texts = [texts]
                
            # Process in batches
            all_embeddings = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded_input = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
            
            if len(all_embeddings) == 1 and len(texts) == 1:
                return all_embeddings[0][0]
            return np.vstack(all_embeddings)
        
        return get_embeddings
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        return None

# Simple vector database implementation
class SimpleVectorDB:
    def __init__(self, embedding_dim):
        self.embeddings = None
        self.documents = []
        self.metadata = []
        self.embedding_dim = embedding_dim
    
    def add_documents(self, documents, metadata, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def similarity_search(self, query_embedding, top_k=5):
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        dot_product = np.dot(self.embeddings, query_embedding)
        indices = np.argsort(dot_product)[-top_k:][::-1]
        similarities = dot_product[indices]
        
        results = []
        for i, idx in enumerate(indices):
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(similarities[i])
            })
        
        return results
    
    def save(self, filepath):
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents,
            "metadata": self.metadata,
            "embedding_dim": self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return cls(CONFIG["embedding_dimension"])
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(data["embedding_dim"])
        db.embeddings = data["embeddings"]
        db.documents = data["documents"]
        db.metadata = data["metadata"]
        return db

# Helper functions for content extraction
def initialize_webdriver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.set_page_load_timeout(10)
        
        return driver
    except Exception as e:
        logger.error(f"Error initializing WebDriver: {e}")
        return None

def extract_content_from_url(driver, url):
    try:
        driver.get(url)
        time.sleep(2)  # Allow page to load
        
        # Get page title
        title = driver.title
        
        # Use BeautifulSoup for parsing
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit text length
        text = text[:20000]
        
        return {
            "title": title,
            "url": url,
            "content": text
        }
    except Exception as e:
        return {
            "title": "Error",
            "url": url,
            "content": f"Failed to extract content: {str(e)}"
        }

def get_content_from_urls(urls, max_pages=5):
    if not urls:
        return []
    
    driver = initialize_webdriver()
    if not driver:
        return []
    
    try:
        results = []
        for i, url in enumerate(urls[:max_pages]):
            content = extract_content_from_url(driver, url)
            results.append(content)
        return results
    finally:
        driver.quit()

# Custom LLM class to directly use Ollama
class OllamaLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        logger.info(f"Initialized OllamaLLM with model: {model_name}")
        
    def generate(self, prompt):
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.5,
                    "num_predict": 2048
                }
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response with {self.model_name}: {e}")
            return f"Error: {str(e)}"

# ----- Main Multi-Agent System -----

class ResearchAgentPipeline:
    def __init__(self):
        # Ensure directories exist
        ensure_directory(CONFIG["text_directory"])
        ensure_directory(CONFIG["vector_db_path"])
        
        # Initialize embedding model
        self.embedding_function = initialize_embedding_model()
        if not self.embedding_function:
            logger.error("Failed to initialize embedding model")
            sys.exit(1)
        
        # Load vector database
        vector_db_path = os.path.join(CONFIG["vector_db_path"], "vector_db.pkl")
        self.vector_db = SimpleVectorDB.load(vector_db_path)
        
        # Initialize direct Ollama connection
        self.ollama = OllamaLLM(CONFIG["ollama_model"])
        logger.info(f"Initialized direct Ollama connection with model: {CONFIG['ollama_model']}")
        
        # For phi agents, create an Ollama instance
        # Note: We'll bypass this when possible
        self.ollama_model = Ollama(model=CONFIG["ollama_model"])
        
        # Set up agents
        self.setup_agents()
    
    def run_query(self, query):
        """Run a query directly without using phi's Agent"""
        try:
            response = ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[{"role": "user", "content": query}],
                options={
                    "temperature": 0.5,
                    "num_predict": 2048
                }
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error running query: {e}")
            return f"Error: {str(e)}"
    
    def setup_agents(self):
        # Define RAG function
        def rag_function(query):
            # Generate query embedding
            query_embedding = self.embedding_function(query)
            
            # Retrieve similar documents
            results = self.vector_db.similarity_search(
                query_embedding=query_embedding, 
                top_k=CONFIG["top_k"]
            )
            
            if not results:
                return "No relevant documents found in the knowledge base."
            
            # Format context from retrieved documents
            context_parts = []
            for i, result in enumerate(results):
                source = result["metadata"].get("source", "Unknown")
                score = result["score"]
                similarity_pct = f"{score * 100:.1f}% relevance"
                context_parts.append(f"Document {i+1}: {source} ({similarity_pct})\n\n{result['document']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant specialized in deep learning and computer vision research (CVPR). 
    You have access to the following documents:

    {context}

    Based on the information above, please answer the following question:
    {query}

    If the information needed to answer the question isn't in the documents, please say so, but try to be helpful.
    """
            
            # Get response from Ollama
            try:
                response = ollama.chat(
                    model=CONFIG["ollama_model"],
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.3,
                        "num_predict": 4096
                    }
                )
                
                answer = response["message"]["content"]
                
                # Include sources
                sources = []
                for result in results:
                    source = result["metadata"].get("source", "Unknown")
                    score = result["score"]
                    sources.append(f"- {source} (relevance: {score:.2f})")
                
                source_text = "\n\nSources:\n" + "\n".join(sources)
                
                return answer + source_text
            except Exception as e:
                logger.error(f"Error getting response from Ollama: {e}")
                return f"Error: {str(e)}"
        
        # Define content extraction function
        def content_extraction_function(query, urls=None):
            if not urls:
                # Try Google Search
                try:
                    google_search = GoogleSearch()
                    results = google_search.search(query, num_results=10)
                    urls = [result["link"] for result in results if "link" in result]
                except Exception as e:
                    # Fallback to DuckDuckGo
                    try:
                        duck = DuckDuckGo()
                        results = duck.search(query, num_results=10)
                        urls = [result["link"] for result in results if "link" in result]
                    except Exception as e:
                        return {"error": f"Search failed: {str(e)}"}
            
            if not urls:
                return {"error": "No URLs found"}
                
            # Extract content using Selenium
            content_results = get_content_from_urls(urls, max_pages=CONFIG["search_depth"])
            
            # Summarize results
            summaries = []
            for result in content_results:
                # Create a prompt for summarization
                summary_prompt = f"""Summarize the following content related to deep learning and computer vision research:

    Title: {result['title']}
    URL: {result['url']}

    Content:
    {result['content'][:5000]}

    Focus on key findings, methodologies, and technical details. Be concise but thorough.
    """
                
                # Get summary using Ollama
                try:
                    summary_response = ollama.chat(
                        model=CONFIG["ollama_model"],
                        messages=[{"role": "user", "content": summary_prompt}],
                        options={
                            "temperature": 0.2,
                            "num_predict": 1000
                        }
                    )
                    
                    summaries.append({
                        "title": result['title'],
                        "url": result['url'],
                        "summary": summary_response["message"]["content"]
                    })
                except Exception as e:
                    logger.error(f"Error summarizing content: {e}")
                    summaries.append({
                        "title": result['title'],
                        "url": result['url'],
                        "summary": f"Error summarizing content: {str(e)}"
                    })
            
            return {
                "query": query,
                "results": summaries
            }
        
        logger.info("Setting up agents...")
        
        # Initialize direct functions for each agent type
        self.knowledge_base_search = rag_function
        self.content_extraction = content_extraction_function
        
        # Initialize agents with error handling
        try:
            self.rag_agent = Agent(
                name="Knowledge Base",
                model=self.ollama_model,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "rag_tool",
                            "description": "Get answer from local knowledge base about deep learning and computer vision",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The question to answer using the local knowledge base",
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ],
                instructions=[
                    "You provide information from the knowledge base for deep learning and computer vision research",
                    "Always include sources when possible",
                    "Be concise and technical in your responses"
                ],
                show_tool_calls=True,
                markdown=True,
                functions={"rag_tool": rag_function}
            )
            logger.info("RAG Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            self.rag_agent = None
    
    def process_query(self, query: str) -> str:
        """Process a research query through our pipeline using direct Ollama calls instead of agents"""
        results = {}
        
        print("🔍 Starting research pipeline for query:", query)
        
        # Step 1: Check local knowledge base
        print("\n📚 Checking local knowledge base...")
        try:
            if self.rag_agent:
                # Try to use the agent first
                try:
                    rag_response = self.rag_agent.run(query)
                    results["knowledge_base"] = rag_response
                except Exception as e:
                    logger.warning(f"Agent-based RAG failed: {e}, using direct function")
                    rag_response = self.knowledge_base_search(query)
                    results["knowledge_base"] = rag_response
            else:
                # Use direct function
                rag_response = self.knowledge_base_search(query)
                results["knowledge_base"] = rag_response
            
            print("✅ Knowledge base search completed")
        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}")
            results["knowledge_base"] = f"Error accessing knowledge base: {str(e)}"
        
        # Step 2: Search ArXiv papers with direct query
        print("\n📑 Searching academic papers...")
        try:
            arxiv_prompt = f"""As a research assistant, find and summarize recent academic papers about "{query}" in the field of deep learning and computer vision. 
Focus on:
1. Paper titles, authors, and publication dates
2. Key methodologies and approaches
3. Main findings and contributions
4. Potential applications of the research

If you don't have specific information, provide general information about how this topic is typically researched in academic literature."""

            arxiv_response = self.run_query(arxiv_prompt)
            results["academic_papers"] = arxiv_response
            print("✅ Academic paper search completed")
        except Exception as e:
            logger.error(f"Error in academic paper search: {e}")
            results["academic_papers"] = f"Error searching academic papers: {str(e)}"
        
        # Step 3: Perform a general web search
        print("\n🌐 Searching the web...")
        try:
            web_prompt = f"""As a research assistant, find and summarize the latest information available about "{query}" in the field of deep learning and computer vision.
Focus on:
1. Recent developments and breakthroughs
2. Leading companies and research groups in this area
3. Best practices and current challenges
4. Future trends and directions

Present this as a comprehensive overview of what someone would find if they searched the web for this topic."""

            web_response = self.run_query(web_prompt)
            results["web_search"] = web_response
            print("✅ Web search completed")
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            results["web_search"] = f"Error searching the web: {str(e)}"
        
        # Step 4: Extract content with direct function
        print("\n🔎 Extracting and analyzing web content...")
        try:
            content_prompt = f"""As a technical analyst, provide a detailed technical explanation of "{query}" in the context of deep learning and computer vision.
Include:
1. Technical definition and fundamental concepts
2. Mathematical formulations if applicable
3. Common implementations and algorithms
4. Evaluation metrics and benchmarks
5. Technical challenges and limitations

Present this as a comprehensive technical guide that would be useful for someone implementing this in their research or projects."""

            content_response = self.run_query(content_prompt)
            results["content_analysis"] = content_response
            print("✅ Content extraction completed")
        except Exception as e:
            logger.error(f"Error in content extraction: {e}")
            results["content_analysis"] = f"Error extracting web content: {str(e)}"
        
        # Step 5: Find code implementations
        print("\n💻 Finding code implementations...")
        try:
            code_prompt = f"""As a developer, describe how "{query}" can be implemented in code for deep learning and computer vision applications.
Include:
1. Common libraries and frameworks used (PyTorch, TensorFlow, etc.)
2. Key algorithms and how they're typically coded
3. Code structure and important functions
4. Tips for efficient implementation
5. Example pseudocode or high-level implementation approaches

Present this as a practical guide for developers looking to implement this technique."""

            code_response = self.run_query(code_prompt)
            results["code_implementations"] = code_response
            print("✅ Code implementation search completed")
        except Exception as e:
            logger.error(f"Error in code implementation search: {e}")
            results["code_implementations"] = f"Error searching code implementations: {str(e)}"
        
        # Step 6: Synthesize all information
        print("\n🧠 Synthesizing information...")
        
        synthesis_prompt = f"""I've researched "{query}" in the field of deep learning and computer vision using multiple sources. Please synthesize this information into a comprehensive report:

1. Knowledge Base Results:
{results.get('knowledge_base', 'Not available')}

2. Academic Papers:
{results.get('academic_papers', 'Not available')}

3. Web Search Results:
{results.get('web_search', 'Not available')}

4. Content Analysis:
{results.get('content_analysis', 'Not available')}

5. Code Implementations:
{results.get('code_implementations', 'Not available')}

Synthesize this information into a well-organized comprehensive report with the following sections:
- Introduction and Definition
- Theoretical Background
- State-of-the-Art Approaches
- Practical Applications
- Implementation Details
- Current Challenges and Future Directions
- Conclusion

Remove any duplicated information and present a coherent, comprehensive analysis. Cite sources where appropriate.
"""
        
        try:
            final_response = self.run_query(synthesis_prompt)
            print("✅ Information synthesis completed")
        except Exception as e:
            logger.error(f"Error in information synthesis: {e}")
            final_response = f"Error synthesizing information: {str(e)}\n\nRaw results:\n{results}"
        
        print("\n✅ Research pipeline complete!")
        return final_response

# Main function
def main():
    # Create the research pipeline
    try:
        # First check what models are available using the command line
        available_models = get_available_models()
        print("Available Ollama models:")
        for model in available_models:
            print(f"- {model}")
        
        if available_models and CONFIG["ollama_model"] not in available_models:
            print(f"\nSpecified model '{CONFIG['ollama_model']}' not found.")
            print("Please choose from available models:")
            for i, model in enumerate(available_models):
                print(f"{i+1}. {model}")
            
            choice = input("\nEnter model number (or press Enter to use the first available): ")
            if choice.strip():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        CONFIG["ollama_model"] = available_models[idx]
                    else:
                        CONFIG["ollama_model"] = available_models[0]
                except ValueError:
                    CONFIG["ollama_model"] = available_models[0]
            else:
                CONFIG["ollama_model"] = available_models[0]
            
            print(f"\nUsing model: {CONFIG['ollama_model']}")
        
        # Test the model directly before proceeding
        try:
            print(f"Testing direct connection to {CONFIG['ollama_model']}...")
            test_response = ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[{"role": "user", "content": "Hello! Please respond with one word."}],
                options={"temperature": 0.1}
            )
            print(f"Test successful - Response received: {test_response['message']['content'][:20]}...")
        except Exception as e:
            print(f"Error testing Ollama model: {e}")
            print("Please check your Ollama installation and verify the model is available.")
            sys.exit(1)
        
        pipeline = ResearchAgentPipeline()
        
        print("=" * 50)
        print("Deep Learning Research Multi-Agent Pipeline")
        print("=" * 50)
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your research query: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            response = pipeline.process_query(query)
            print("\n" + "=" * 50)
            print("FINAL RESEARCH REPORT:")
            print("=" * 50)
            print(response)
            print("\n" + "=" * 50)
    except Exception as e:
        logger.error(f"Critical error in pipeline: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()