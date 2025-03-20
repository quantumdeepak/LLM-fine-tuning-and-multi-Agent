import os
import sys
import time
import json
import pickle
import traceback
import signal
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import threading
import queue
import logging
import urllib.parse
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_system.log')
    ]
)
logger = logging.getLogger("MultiAgentSystem")

# Required dependencies - Adding safer imports with fallbacks
try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    import ollama
    import requests
    from bs4 import BeautifulSoup
    
    # Handle lxml.html.clean issue
    try:
        from lxml_html_clean import clean_html
        HAVE_LXML_HTML_CLEAN = True
    except ImportError:
        try:
            from lxml.html.clean import clean_html
            HAVE_LXML_HTML_CLEAN = True
        except ImportError:
            HAVE_LXML_HTML_CLEAN = False
            logger.warning("lxml_html_clean not found. HTML cleaning will be limited.")
    
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    
    # Use try-except for webdriver_manager which might not be installed
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        HAVE_WEBDRIVER_MANAGER = True
    except ImportError:
        HAVE_WEBDRIVER_MANAGER = False
        logger.warning("webdriver_manager not found. Will use system Chrome driver.")
    
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from tqdm import tqdm
    import nltk
    
    # Import newspaper with fallbacks for cleaner errors
    try:
        from newspaper import Article, ArticleException
    except ImportError as e:
        if "lxml.html.clean" in str(e):
            logger.error("newspaper3k requires lxml_html_clean. Please install with: pip install lxml_html_clean")
            raise ImportError("newspaper3k requires lxml_html_clean. Please install with: pip install lxml_html_clean")
        else:
            raise
    
    # DuckDuckGo search
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo_search not found. DuckDuckGo search will be disabled.")
        DDGS = None
    
    # Use simple HTML to text converter if html2text is not available
    try:
        import html2text
        HAVE_HTML2TEXT = True
    except ImportError:
        HAVE_HTML2TEXT = False
        logger.warning("html2text not found. Using a basic HTML stripper.")
        
        # Simple HTML to text function as fallback
        def strip_html_tags(html):
            """Remove html tags from a string"""
            if not html:
                return ""
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")

except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    print(f"Missing dependency: {e}")
    print("Please install the required packages:")
    print("pip install numpy torch transformers ollama requests bs4 selenium tqdm nltk newspaper3k duckduckgo-search webdriver-manager lxml_html_clean")
    print("You may need to download nltk data: python -c 'import nltk; nltk.download(\"punkt\")'")
    sys.exit(1)

# Try to download nltk data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Handle graceful exit
def signal_handler(sig, frame):
    print("\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Default configuration
DEFAULT_CONFIG = {
    # File and directory paths
    "text_directory": "/mnt/DATA/Glucoma/LLM/Ollama_pdf_handle/Text_pdf",  # Using your specified directory
    "vector_db_path": "data/vector_db",   # Directory to store vector database
    "cache_directory": "data/cache",      # Directory to store web page and search caches
    
    # Model settings
    "ollama_model": "gemma3:12b",         # Ollama model to use
    "cuda_device": "0",                   # GPU device to use (set to "" for CPU)
    "temperature": 0.7,                   # Temperature for text generation
    "max_tokens": 2048,                   # Maximum tokens in response
    
    # Embedding settings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # HF model for embeddings
    "embedding_dimension": 384,           # Embedding dimension for the model
    "rebuild_db": False,                  # Whether to rebuild the vector DB on startup
    
    # Chunking parameters
    "chunk_size": 1000,                   # Size of text chunks for embedding
    "chunk_overlap": 200,                 # Overlap between chunks
    "min_chunk_size": 50,                 # Minimum chunk size to keep
    
    # Retrieval parameters
    "top_k": 5,                           # Number of documents to retrieve per query
    
    # Web search parameters
    "search_results_count": 5,            # Number of search results to fetch
    "search_timeout": 30,                 # Search timeout in seconds
    "scrape_timeout": 60,                 # Web scraping timeout per page in seconds
    "search_engine": "google",            # Options: "google", "duckduckgo", "both"
    "google_search_url": "https://www.google.com/search?q=",  # Base URL for Google search scraping
    
    # Processing parameters
    "batch_size": 16,                     # Batch size for embedding generation
    "thread_count": 3,                    # Number of threads for parallel processing
    
    # Agent settings
    "enable_rag_agent": True,             # Enable RAG-based agent
    "enable_search_agent": True,          # Enable search-based agent
    "enable_google_agent": True,          # Enable Google search-based agent
    "enable_scrape_agent": True,          # Enable web scraping agent
    "cache_results": True,                # Cache search and scraping results
    "cache_expiry": 86400,                # Cache expiry time in seconds (24 hours)
    
    # UI settings
    "api_port": 8000,                     # Port for API server
    "enable_debug": False,                # Enable debug mode
}

# Load configuration
def load_config():
    config_file = "config.json"
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Configuration file {config_file} not found, using defaults")
        # Create a default config file for the user
        try:
            with open(config_file, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            logger.info(f"Created default configuration file: {config_file}")
        except Exception as e:
            logger.error(f"Failed to create default configuration file: {e}")
    
    return config

CONFIG = load_config()

# Create directory if it doesn't exist
def ensure_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

# Configure GPU
def configure_gpu():
    if CONFIG["cuda_device"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_device"]
        
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("No CUDA GPUs found despite configuration, falling back to CPU")
            return False
    else:
        logger.info("Using CPU as configured")
        return False

# Mean Pooling for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize embedding model
def initialize_embedding_model():
    logger.info(f"Loading embedding model: {CONFIG['embedding_model']}")
    
    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['embedding_model'])
        model = AutoModel.from_pretrained(CONFIG['embedding_model'])
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() and CONFIG["cuda_device"] else "cpu")
        model = model.to(device)
        
        if device.type == 'cuda':
            logger.info(f"Embedding model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Embedding model loaded on CPU")
        
        # Create embedding function
        def get_embeddings(texts):
            if isinstance(texts, str):
                texts = [texts]
                
            # Process in batches
            all_embeddings = []
            batch_size = CONFIG['batch_size']
            
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
                
                # Move to device
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                # Compute embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                # Pool embeddings
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
            
            # Combine batches
            if len(all_embeddings) == 1 and len(texts) == 1:
                return all_embeddings[0][0]  # Return single embedding
            return np.vstack(all_embeddings)
        
        return get_embeddings
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        traceback.print_exc()
        sys.exit(1)

# Chunk text for embedding
def chunk_text(text):
    if not text:
        return []
    
    # Split by paragraphs
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding paragraph exceeds chunk size and we have content, 
        # save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CONFIG["chunk_size"] and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap for context
            current_chunk = current_chunk[-CONFIG["chunk_overlap"]:] if CONFIG["chunk_overlap"] > 0 else ""
        
        # Add paragraph
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle very large paragraphs that exceed chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= CONFIG["chunk_size"]:
            final_chunks.append(chunk)
        else:
            # Split by sentences using nltk
            try:
                sentences = nltk.sent_tokenize(chunk)
                new_chunk = ""
                
                for sentence in sentences:
                    if len(new_chunk) + len(sentence) > CONFIG["chunk_size"] and new_chunk:
                        final_chunks.append(new_chunk.strip())
                        # Keep overlap
                        new_chunk = new_chunk[-CONFIG["chunk_overlap"]:] if CONFIG["chunk_overlap"] > 0 else ""
                    
                    if new_chunk:
                        new_chunk += " " + sentence
                    else:
                        new_chunk = sentence
                
                if new_chunk.strip():
                    final_chunks.append(new_chunk.strip())
            except:
                # Fallback for nltk failure: split by characters
                for i in range(0, len(chunk), CONFIG["chunk_size"] - CONFIG["chunk_overlap"]):
                    final_chunks.append(chunk[i:i + CONFIG["chunk_size"]].strip())
    
    # Filter out chunks that are too small
    return [chunk for chunk in final_chunks if len(chunk) >= CONFIG["min_chunk_size"]]

# Load and process text files
def load_and_process_files():
    import glob
    
    text_files = glob.glob(os.path.join(CONFIG["text_directory"], "**/*.txt"), recursive=True)
    logger.info(f"Found {len(text_files)} text files in {CONFIG['text_directory']}")
    
    all_chunks = []
    all_metadata = []
    
    for file_idx, file_path in enumerate(tqdm(text_files, desc="Processing files")):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Chunk the text
            chunks = chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Add metadata
                rel_path = os.path.relpath(file_path, CONFIG["text_directory"])
                metadata = {
                    "source": rel_path,
                    "file_path": file_path,
                    "chunk_idx": chunk_idx,
                    "file_idx": file_idx,
                    "id": f"doc_{file_idx}_chunk_{chunk_idx}"
                }
                
                all_chunks.append(chunk)
                all_metadata.append(metadata)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return all_chunks, all_metadata

# Simple vector database implementation
class SimpleVectorDB:
    def __init__(self, embedding_dim):
        self.embeddings = None  # Will be numpy array
        self.documents = []     # Original text chunks
        self.metadata = []      # Metadata for each chunk
        self.embedding_dim = embedding_dim
    
    def add_documents(self, documents, metadata, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def similarity_search(self, query_embedding, top_k=5):
        if not len(self.documents):
            return []
            
        # Compute cosine similarity
        dot_product = np.dot(self.embeddings, query_embedding)
        
        # Get top k results
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
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(data["embedding_dim"])
        db.embeddings = data["embeddings"]
        db.documents = data["documents"]
        db.metadata = data["metadata"]
        return db

# Build or load vector database
def setup_vector_db(embedding_function):
    ensure_directory(CONFIG["vector_db_path"])
    db_file = os.path.join(CONFIG["vector_db_path"], "vector_db.pkl")
    
    if os.path.exists(db_file) and not CONFIG["rebuild_db"]:
        logger.info(f"Loading existing vector database from {db_file}")
        try:
            return SimpleVectorDB.load(db_file)
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            logger.info("Rebuilding database...")
            CONFIG["rebuild_db"] = True
    
    logger.info("Building new vector database...")
    
    # Create new database
    vector_db = SimpleVectorDB(CONFIG["embedding_dimension"])
    
    # Load and process documents
    chunks, metadata = load_and_process_files()
    
    if not chunks:
        logger.warning("No documents found or processed. The RAG agent will use only the LLM's knowledge.")
        return vector_db
    
    logger.info(f"Generated {len(chunks)} chunks from documents")
    
    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    
    batch_size = CONFIG["batch_size"]
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_metadata = metadata[i:end_idx]
        
        # Generate embeddings
        batch_embeddings = embedding_function(batch_chunks)
        
        # Add to database
        vector_db.add_documents(batch_chunks, batch_metadata, batch_embeddings)
    
    # Save database
    vector_db.save(db_file)
    logger.info(f"Vector database saved to {db_file}")
    
    return vector_db

# Cache implementation for search results and scraped content
class ResultCache:
    def __init__(self, cache_dir, expiry=86400):
        self.cache_dir = ensure_directory(cache_dir)
        self.expiry = expiry  # Cache expiry in seconds
    
    def get_cache_path(self, key, prefix=""):
        # Create a filename from the key
        import hashlib
        hash_obj = hashlib.md5(key.encode('utf-8'))
        filename = f"{prefix}_{hash_obj.hexdigest()}.json"
        return os.path.join(self.cache_dir, filename)
    
    def get(self, key, prefix=""):
        cache_path = self.get_cache_path(key, prefix)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_path) > self.expiry:
                os.remove(cache_path)
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            return cached_data
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, key, data, prefix=""):
        cache_path = self.get_cache_path(key, prefix)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False

# DuckDuckGo Search agent implementation
class DuckDuckGoSearchAgent:
    def __init__(self, cache=None):
        self.cache = cache
        if DDGS is None:
            logger.warning("DuckDuckGo search is disabled due to missing duckduckgo_search module")
    
    def search(self, query):
        # Check if DuckDuckGo search is available
        if DDGS is None:
            logger.warning("DuckDuckGo search requested but module is not available")
            return []
            
        # Check cache first
        if self.cache and CONFIG["cache_results"]:
            cached_results = self.cache.get(query, "ddg_search")
            if cached_results:
                logger.info(f"Using cached DuckDuckGo search results for: {query}")
                return cached_results
        
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query, 
                    max_results=CONFIG["search_results_count"]
                ))
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "href": result.get("href", ""),
                        "body": result.get("body", ""),
                        "source": "duckduckgo"
                    })
            
            # Cache results
            if self.cache and CONFIG["cache_results"]:
                self.cache.set(query, results, "ddg_search")
            
            return results
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}")
            traceback.print_exc()
            return []
    
    def format_search_results(self, results):
        if not results:
            return "No DuckDuckGo search results found."
        
        formatted = "### DuckDuckGo Search Results\n\n"
        for i, result in enumerate(results):
            formatted += f"**{i+1}. [{result['title']}]({result['href']})**\n"
            formatted += f"{result['body']}\n\n"
        
        return formatted
    
    def generate_response(self, query, results=None):
        if DDGS is None:
            return {
                "query": query,
                "response": "DuckDuckGo search is not available. Please install the duckduckgo-search package.",
                "source": "duckduckgo_search_agent"
            }
            
        if results is None:
            results = self.search(query)
        
        search_context = self.format_search_results(results)
        
        prompt = f"""You are a helpful search assistant. 
You have access to DuckDuckGo search results for the query: "{query}".

Here are the search results:

{search_context}

Based on these search results, provide a helpful and informative response to the query.
Ensure you cite your sources from the search results, using [1], [2], etc. to reference them.
If the search results don't contain enough information to fully answer the query, mention this and provide the best response you can with the available information.
"""
        
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        answer = response["message"]["content"]
        
        return {
            "query": query,
            "search_results": results,
            "response": answer,
            "source": "duckduckgo_search_agent"
        }

# Google Search agent implementation
class GoogleSearchAgent:
    def __init__(self, cache=None):
        self.cache = cache
        self.chrome_options = None
        self.init_selenium()
    
    def init_selenium(self):
        # Configure Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option("useAutomationExtension", False)
    
    def search(self, query):
        # Check cache first
        if self.cache and CONFIG["cache_results"]:
            cached_results = self.cache.get(query, "google_search")
            if cached_results:
                logger.info(f"Using cached Google search results for: {query}")
                return cached_results
        
        try:
            logger.info(f"Searching Google for: {query}")
            
            # Prepare the search URL
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"{CONFIG['google_search_url']}{encoded_query}"
            
            # Use Selenium to perform the search
            driver = None
            try:
                # Initialize the driver, with fallback to system driver
                if HAVE_WEBDRIVER_MANAGER:
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=self.chrome_options)
                else:
                    # Fallback to using system Chrome driver
                    driver = webdriver.Chrome(options=self.chrome_options)
                    
                driver.set_page_load_timeout(CONFIG["search_timeout"])
                
                # Navigate to the search URL
                driver.get(search_url)
                
                # Wait for search results to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "search"))
                )
                
                # Extract search results
                results = []
                search_results = driver.find_elements(By.CSS_SELECTOR, "div.g")
                
                for i, result in enumerate(search_results):
                    if i >= CONFIG["search_results_count"]:
                        break
                    
                    try:
                        title_element = result.find_element(By.CSS_SELECTOR, "h3")
                        title = title_element.text
                        
                        link_element = result.find_element(By.CSS_SELECTOR, "a")
                        href = link_element.get_attribute("href")
                        
                        # Get snippet
                        snippet_element = result.find_element(By.CSS_SELECTOR, "div.VwiC3b")
                        snippet = snippet_element.text
                        
                        if title and href:
                            results.append({
                                "title": title,
                                "href": href,
                                "body": snippet,
                                "source": "google"
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting search result: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"Error during Google search with Selenium: {e}")
                # Fallback to simple requests-based scraping if Selenium fails
                results = self._fallback_search(query)
            
            finally:
                if driver:
                    driver.quit()
            
            # Cache results
            if results and self.cache and CONFIG["cache_results"]:
                self.cache.set(query, results, "google_search")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during Google search: {e}")
            traceback.print_exc()
            return []
    
    def _fallback_search(self, query):
        """Fallback method using requests and BeautifulSoup if Selenium fails"""
        try:
            logger.info("Using fallback Google search method")
            
            # Prepare the search URL
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"{CONFIG['google_search_url']}{encoded_query}"
            
            # Set custom headers to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            # Make the request
            response = requests.get(search_url, headers=headers, timeout=CONFIG["search_timeout"])
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            search_results = soup.select("div.g")
            
            for i, result in enumerate(search_results):
                if i >= CONFIG["search_results_count"]:
                    break
                
                try:
                    title_element = result.select_one("h3")
                    if not title_element:
                        continue
                    
                    title = title_element.get_text()
                    
                    link_element = result.select_one("a")
                    if not link_element:
                        continue
                        
                    href = link_element.get("href")
                    if href and href.startswith("/url?"):
                        match = re.search(r"q=([^&]+)", href)
                        if match:
                            href = match.group(1)
                    
                    # Get snippet
                    snippet_element = result.select_one("div.VwiC3b")
                    snippet = snippet_element.get_text() if snippet_element else ""
                    
                    if title and href and href.startswith(("http://", "https://")):
                        results.append({
                            "title": title,
                            "href": href,
                            "body": snippet,
                            "source": "google"
                        })
                except Exception as e:
                    logger.warning(f"Error extracting search result: {e}")
                    continue
            
            return results
        
        except Exception as e:
            logger.error(f"Error during fallback Google search: {e}")
            return []
    
    def format_search_results(self, results):
        if not results:
            return "No Google search results found."
        
        formatted = "### Google Search Results\n\n"
        for i, result in enumerate(results):
            formatted += f"**{i+1}. [{result['title']}]({result['href']})**\n"
            formatted += f"{result['body']}\n\n"
        
        return formatted
    
    def generate_response(self, query, results=None):
        if results is None:
            results = self.search(query)
        
        search_context = self.format_search_results(results)
        
        prompt = f"""You are a helpful search assistant. 
You have access to Google search results for the query: "{query}".

Here are the search results:

{search_context}

Based on these search results, provide a helpful and informative response to the query.
Ensure you cite your sources from the search results, using [1], [2], etc. to reference them.
If the search results don't contain enough information to fully answer the query, mention this and provide the best response you can with the available information.
"""
        
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        answer = response["message"]["content"]
        
        return {
            "query": query,
            "search_results": results,
            "response": answer,
            "source": "google_search_agent"
        }

# Web scraping agent implementation
class WebScrapingAgent:
    def __init__(self, cache=None):
        self.cache = cache
        self.chrome_options = None
        
        # Initialize HTML to text converter
        if HAVE_HTML2TEXT:
            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = False
            self.html_converter.ignore_images = True
            self.html_converter.ignore_tables = False
            self.html_converter.body_width = 0  # No wrapping
        else:
            # Use the fallback function
            self.html_converter = None
            
        self.init_selenium()
    
    def init_selenium(self):
        # Configure Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option("useAutomationExtension", False)
    
    def _html_to_text(self, html):
        """Convert HTML to text, using html2text if available, otherwise fallback"""
        if not html:
            return ""
            
        if HAVE_HTML2TEXT and self.html_converter:
            return self.html_converter.handle(html)
        else:
            return strip_html_tags(html)
    
    def _extract_with_selenium(self, url):
        driver = None
        try:
            # Initialize the driver with fallback to system driver
            if HAVE_WEBDRIVER_MANAGER:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=self.chrome_options)
            else:
                # Fallback to using system Chrome driver
                driver = webdriver.Chrome(options=self.chrome_options)
                
            driver.set_page_load_timeout(CONFIG["scrape_timeout"])
            
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page content
            content = driver.page_source
            
            # Find the main content area
            main_content = driver.find_elements(By.XPATH, "//main|//article|//div[@role='main']|//div[@id='content']")
            if main_content:
                text_content = []
                for element in main_content:
                    text_content.append(element.text)
                text = "\n\n".join(text_content)
            else:
                # If no main content found, use body
                body = driver.find_element(By.TAG_NAME, "body")
                text = body.text
            
            # Also convert HTML to markdown as a backup
            markdown_content = self._html_to_text(content)
            
            # If text content is too short, use markdown content
            if len(text) < 100 and len(markdown_content) > 100:
                text = markdown_content
            
            title = driver.title
            
            result = {
                "url": url,
                "title": title,
                "content": text,
                "html": content,
                "extraction_method": "selenium"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Selenium extraction error for {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _extract_with_newspaper(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            result = {
                "url": url,
                "title": article.title,
                "content": article.text,
                "html": article.html,
                "extraction_method": "newspaper"
            }
            
            return result
        except Exception as e:
            logger.error(f"Newspaper extraction error for {url}: {e}")
            return None
    
    def _extract_with_requests(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            response = requests.get(url, headers=headers, timeout=CONFIG["scrape_timeout"])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.text if soup.title else url
            
            # Get content
            content_elements = soup.select("main, article, div[role='main'], div#content, div.content")
            if content_elements:
                content = "\n\n".join([elem.get_text(separator="\n") for elem in content_elements])
            else:
                # Fallback to body
                content = soup.body.get_text(separator="\n") if soup.body else ""
            
            # Also convert HTML to markdown as a backup
            markdown_content = self._html_to_text(response.text)
            
            # If content is too short, use markdown content
            if len(content) < 100 and len(markdown_content) > 100:
                content = markdown_content
            
            result = {
                "url": url,
                "title": title,
                "content": content,
                "html": response.text,
                "extraction_method": "requests"
            }
            
            return result
        except Exception as e:
            logger.error(f"Requests extraction error for {url}: {e}")
            return None
    
    def scrape_url(self, url):
        # Check cache first
        if self.cache and CONFIG["cache_results"]:
            cached_content = self.cache.get(url, "scrape")
            if cached_content:
                logger.info(f"Using cached content for: {url}")
                return cached_content
        
        logger.info(f"Scraping: {url}")
        
        # Try newspaper first (it's faster)
        result = self._extract_with_newspaper(url)
        
        # If newspaper fails or content is too short, try Selenium
        if not result or not result.get("content") or len(result.get("content", "").strip()) < 100:
            result = self._extract_with_selenium(url)
        
        # If Selenium fails or content is too short, try requests
        if not result or not result.get("content") or len(result.get("content", "").strip()) < 100:
            result = self._extract_with_requests(url)
        
        # If all methods fail, return None
        if not result or not result.get("content") or len(result.get("content", "").strip()) < 50:
            logger.warning(f"Failed to extract content from {url}")
            return None
        
        # Truncate content if it's too long
        if len(result["content"]) > 20000:
            result["content"] = result["content"][:20000] + "... [truncated due to length]"
        
        # Remove HTML to save space
        if "html" in result:
            del result["html"]
        
        # Cache results
        if result and self.cache and CONFIG["cache_results"]:
            self.cache.set(url, result, "scrape")
        
        return result
    
    def scrape_search_results(self, search_results):
        scraped_contents = []
        
        for result in search_results:
            url = result.get("href", "")
            if not url or not url.startswith(("http://", "https://")):
                continue
                
            content = self.scrape_url(url)
            if content:
                # Add search result metadata
                content["search_title"] = result.get("title", "")
                content["search_snippet"] = result.get("body", "")
                scraped_contents.append(content)
        
        return scraped_contents
    
    def summarize_content(self, scraped_contents, query):
        if not scraped_contents:
            return "No content was successfully scraped from the provided URLs."
        
        # Create summaries of each scraped content
        summaries = []
        
        for i, content in enumerate(scraped_contents):
            # Prepare context for the model
            text = content.get("content", "")
            title = content.get("title", content.get("search_title", ""))
            url = content.get("url", "")
            
            # Truncate if too long
            if len(text) > 10000:
                text = text[:10000] + "... [truncated due to length]"
            
            summaries.append({
                "index": i + 1,
                "title": title,
                "url": url,
                "summary": text
            })
        
        return summaries
    
    def generate_response(self, query, search_results=None, search_agent=None):
        if search_results is None and search_agent is not None:
            search_results = search_agent.search(query)
        
        if not search_results:
            return {
                "query": query,
                "response": "No search results available to scrape content from. Unable to provide a web scraping response.",
                "source": "scraping_agent"
            }
        
        # Scrape content from search results
        scraped_contents = self.scrape_search_results(search_results)
        
        if not scraped_contents:
            return {
                "query": query,
                "response": "Failed to scrape any content from the search results. Unable to provide a web scraping response.",
                "source": "scraping_agent"
            }
        
        # Summarize content
        summaries = self.summarize_content(scraped_contents, query)
        
        # Create prompt for LLM
        context_parts = []
        for summary in summaries:
            context_parts.append(f"Page {summary['index']}: {summary['title']} ({summary['url']})\n\n{summary['summary']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful web scraping assistant.
You have access to scraped content from multiple web pages related to the query: "{query}".

Here is the scraped content:

{context}

Based on this scraped content, provide a detailed and informative response to the query.
Ensure you cite your sources using [1], [2], etc. to reference the different pages.
If the scraped content doesn't contain enough information to fully answer the query, mention this and provide the best response you can with the available information.
"""
        
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        answer = response["message"]["content"]
        
        return {
            "query": query,
            "scraped_pages": [{"title": s["title"], "url": s["url"]} for s in summaries],
            "response": answer,
            "source": "scraping_agent"
        }

# RAG agent implementation
class RAGAgent:
    def __init__(self, vector_db, embedding_function):
        self.vector_db = vector_db
        self.embedding_function = embedding_function
    
    def retrieve(self, query):
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Retrieve similar documents
        results = self.vector_db.similarity_search(
            query_embedding=query_embedding, 
            top_k=CONFIG["top_k"]
        )
        
        return results
    
    def create_rag_prompt(self, query, results):
        # Format context from retrieved documents
        if not results:
            return f"You are a helpful AI assistant. Please answer the following question based on your knowledge: {query}"
            
        context_parts = []
        
        for i, result in enumerate(results):
            source = result["metadata"].get("source", "Unknown")
            score = result["score"]
            similarity_pct = f"{score * 100:.1f}% relevance"
            
            context_parts.append(f"Document {i+1}: {source} ({similarity_pct})\n\n{result['document']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant with access to the following documents:

{context}

Based on the information in these documents, please answer the following question:
{query}

If the information needed to answer the question isn't in the documents, please say so, but try to be helpful with what you know from the documents.
"""
        return prompt
    
    def generate_response(self, query):
        # Retrieve documents
        results = self.retrieve(query)
        
        # Create RAG prompt
        prompt = self.create_rag_prompt(query, results)
        
        # Get response from Ollama
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        answer = response["message"]["content"]
        
        return {
            "query": query,
            "documents": [{"source": r["metadata"].get("source", "Unknown"), "score": r["score"]} for r in results],
            "response": answer,
            "source": "rag_agent"
        }

# Coordinator to manage all agents
class AgentCoordinator:
    def __init__(self):
        # Initialize cache
        self.cache = ResultCache(
            ensure_directory(CONFIG["cache_directory"]), 
            expiry=CONFIG["cache_expiry"]
        )
        
        # Configure GPU
        configure_gpu()
        
        # Initialize embedding model
        self.embedding_function = initialize_embedding_model()
        
        # Setup vector database for RAG
        self.vector_db = setup_vector_db(self.embedding_function)
        
        # Initialize agents
        self.rag_agent = RAGAgent(self.vector_db, self.embedding_function) if CONFIG["enable_rag_agent"] else None
        self.ddg_search_agent = DuckDuckGoSearchAgent(self.cache) if CONFIG["enable_search_agent"] else None
        self.google_search_agent = GoogleSearchAgent(self.cache) if CONFIG["enable_google_agent"] else None
        self.scrape_agent = WebScrapingAgent(self.cache) if CONFIG["enable_scrape_agent"] else None
        
        # For parallel execution
        self.thread_pool = []
        self.results_queue = queue.Queue()
    
    def _run_agent_and_queue_result(self, agent_name, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            self.results_queue.put(result)
            logger.info(f"{agent_name} completed processing")
        except Exception as e:
            logger.error(f"Error in {agent_name}: {e}")
            traceback.print_exc()
            self.results_queue.put({
                "source": agent_name.lower().replace(" ", "_"),
                "response": f"Error processing query: {str(e)}",
                "error": str(e)
            })
    
    def _process_query_parallel(self, query):
        # Clear any previous results
        while not self.results_queue.empty():
            self.results_queue.get()
        
        # Clear any existing threads
        self.thread_pool = []
        
        # Create threads for each agent
        if self.rag_agent:
            rag_thread = threading.Thread(
                target=self._run_agent_and_queue_result,
                args=("RAG Agent", self.rag_agent.generate_response, query)
            )
            self.thread_pool.append(rag_thread)
        
        # Run search agents
        search_results = {}
        
        if self.ddg_search_agent:
            ddg_thread = threading.Thread(
                target=self._run_agent_and_queue_result,
                args=("DuckDuckGo Search Agent", self.ddg_search_agent.generate_response, query)
            )
            self.thread_pool.append(ddg_thread)
            
            # Get search results for scraper (don't wait for response generation)
            search_results["ddg"] = self.ddg_search_agent.search(query)
        
        if self.google_search_agent:
            google_thread = threading.Thread(
                target=self._run_agent_and_queue_result,
                args=("Google Search Agent", self.google_search_agent.generate_response, query)
            )
            self.thread_pool.append(google_thread)
            
            # Get search results for scraper (don't wait for response generation)
            search_results["google"] = self.google_search_agent.search(query)
        
        # Combine search results for scraper
        combined_results = []
        if "ddg" in search_results:
            combined_results.extend(search_results["ddg"])
        if "google" in search_results:
            combined_results.extend(search_results["google"])
        
        # Only run scraper if we have search results
        if self.scrape_agent and combined_results:
            scrape_thread = threading.Thread(
                target=self._run_agent_and_queue_result,
                args=("Scrape Agent", self.scrape_agent.generate_response, query, combined_results, None)
            )
            self.thread_pool.append(scrape_thread)
        
        # Start all threads
        for thread in self.thread_pool:
            thread.start()
        
        # Wait for all threads to complete
        for thread in self.thread_pool:
            thread.join()
    
    def _process_query_sequential(self, query):
        # Run RAG agent
        if self.rag_agent:
            try:
                result = self.rag_agent.generate_response(query)
                self.results_queue.put(result)
                logger.info("RAG Agent completed processing")
            except Exception as e:
                logger.error(f"Error in RAG Agent: {e}")
                self.results_queue.put({
                    "source": "rag_agent",
                    "response": f"Error processing query: {str(e)}",
                    "error": str(e)
                })
        
        # Run DuckDuckGo Search agent
        ddg_search_results = None
        if self.ddg_search_agent:
            try:
                ddg_search_results = self.ddg_search_agent.search(query)
                result = self.ddg_search_agent.generate_response(query, ddg_search_results)
                self.results_queue.put(result)
                logger.info("DuckDuckGo Search Agent completed processing")
            except Exception as e:
                logger.error(f"Error in DuckDuckGo Search Agent: {e}")
                self.results_queue.put({
                    "source": "duckduckgo_search_agent",
                    "response": f"Error processing query: {str(e)}",
                    "error": str(e)
                })
        
        # Run Google Search agent
        google_search_results = None
        if self.google_search_agent:
            try:
                google_search_results = self.google_search_agent.search(query)
                result = self.google_search_agent.generate_response(query, google_search_results)
                self.results_queue.put(result)
                logger.info("Google Search Agent completed processing")
            except Exception as e:
                logger.error(f"Error in Google Search Agent: {e}")
                self.results_queue.put({
                    "source": "google_search_agent",
                    "response": f"Error processing query: {str(e)}",
                    "error": str(e)
                })
        
        # Combine search results for scraper
        combined_results = []
        if ddg_search_results:
            combined_results.extend(ddg_search_results)
        if google_search_results:
            combined_results.extend(google_search_results)
        
        # Run Scrape agent
        if self.scrape_agent and combined_results:
            try:
                result = self.scrape_agent.generate_response(query, combined_results)
                self.results_queue.put(result)
                logger.info("Scrape Agent completed processing")
            except Exception as e:
                logger.error(f"Error in Scrape Agent: {e}")
                self.results_queue.put({
                    "source": "scraping_agent",
                    "response": f"Error processing query: {str(e)}",
                    "error": str(e)
                })
    
    def _generate_coordinator_response(self, query, results):
        # Format all agent responses
        agent_responses = []
        
        if "rag_agent" in results:
            rag_result = results["rag_agent"]
            agent_responses.append(f"Local knowledge base analysis (RAG):\n{rag_result['response']}")
        
        if "duckduckgo_search_agent" in results:
            ddg_result = results["duckduckgo_search_agent"]
            agent_responses.append(f"DuckDuckGo search results:\n{ddg_result['response']}")
        
        if "google_search_agent" in results:
            google_result = results["google_search_agent"]
            agent_responses.append(f"Google search results:\n{google_result['response']}")
        
        if "scraping_agent" in results:
            scrape_result = results["scraping_agent"]
            agent_responses.append(f"Detailed web content analysis:\n{scrape_result['response']}")
        
        agent_context = "\n\n====================\n\n".join(agent_responses)
        
        # Create prompt for coordinator
        prompt = f"""You are a coordinator agent responsible for synthesizing information from multiple AI agents who researched the query: "{query}".

Below are the responses from each agent:

{agent_context}

Your task is to:
1. Synthesize a comprehensive response that integrates information from all agents
2. Highlight areas of agreement and resolve any conflicts or contradictions
3. Provide a final, coherent answer to the original query
4. Indicate which sources provided the most relevant information

Here's your synthesized response:
"""
        
        # Get response from Ollama
        response = ollama.chat(
            model=CONFIG["ollama_model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"] * 2  # Allow longer responses for synthesis
            }
        )
        
        return response["message"]["content"]
    
    def process_query(self, query):
        """Process a query and return the results"""
        logger.info(f"Processing query: {query}")
        
        # Run agents in parallel or sequential based on thread count
        if CONFIG["thread_count"] > 1:
            self._process_query_parallel(query)
        else:
            self._process_query_sequential(query)
        
        # Collect results from queue
        results = {}
        while not self.results_queue.empty():
            result = self.results_queue.get()
            results[result["source"]] = result
        
        # Generate coordinator response
        coordinator_response = self._generate_coordinator_response(query, results)
        
        return {
            "query": query,
            "coordinator_response": coordinator_response,
            "agent_responses": results
        }
    
    def interactive_chat(self):
        print("\nMulti-Agent Interactive Chat with", CONFIG["ollama_model"])
        print("Type 'quit', 'exit', 'bye', 'end', or 'terminate' to end the chat")
        print("Type 'config' to show current configuration")
        print("Type 'edit config' to modify configuration settings")
        print("Type 'rebuild' to rebuild the vector database")
        print("="*50)
        
        while True:
            query = input("\nYou: ")
            query = query.strip()
            
            if query.lower() in ["quit", "exit", "bye", "end", "terminate"]:
                print("Ending chat session. Goodbye!")
                break
                
            elif query.lower() == "config":
                print("\nCurrent Configuration:")
                for key, value in CONFIG.items():
                    print(f"{key}: {value}")
                continue
                
            elif query.lower() == "edit config":
                self._edit_config()
                continue
                
            elif query.lower() == "rebuild":
                CONFIG["rebuild_db"] = True
                self.vector_db = setup_vector_db(self.embedding_function)
                continue
            
            # Show "thinking" animation
            import itertools
            import sys
            spinner = itertools.cycle(['-', '/', '|', '\\'])
            print("Processing your query ", end="")
            self.stop_spinner = False
            spinner_thread = threading.Thread(target=self._display_spinner)
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
                # Process query with all agents
                start_time = time.time()
                
                # Get results
                results = self.process_query(query)
                coordinator_response = results["coordinator_response"]
                agent_responses = results["agent_responses"]
                
                # Stop spinner
                self.stop_spinner = True
                spinner_thread.join(timeout=1)
                
                processing_time = time.time() - start_time
                
                # Print the coordinator's response
                print(f"\nResponse (processed in {processing_time:.2f}s):")
                print("-"*50)
                print(coordinator_response)
                print("-"*50)
                
                # Ask if user wants to see individual agent responses
                show_details = input("\nWould you like to see individual agent responses? (y/n): ")
                if show_details.lower() in ["y", "yes"]:
                    for source, response in agent_responses.items():
                        print(f"\n{source.replace('_', ' ').title()} Response:")
                        print("-"*50)
                        print(response["response"])
                        print("-"*50)
                
            except Exception as e:
                # Stop spinner
                self.stop_spinner = True
                spinner_thread.join(timeout=1)
                
                print(f"\nError generating response: {e}")
                traceback.print_exc()
            
            print("\n" + "="*50)
    
    def _display_spinner(self):
        import itertools
        import time
        import sys
        
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        while not getattr(self, "stop_spinner", False):
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')
    
    def _edit_config(self):
        """Interactive configuration editor"""
        print("\nConfiguration Editor:")
        print("Enter the key you want to modify, or 'save' to finish editing")
        
        while True:
            # Show current config
            print("\nCurrent Configuration:")
            for i, (key, value) in enumerate(CONFIG.items(), 1):
                print(f"{i}. {key}: {value}")
            
            choice = input("\nEnter item number or name to edit (or 'save' to finish): ")
            
            if choice.lower() == 'save':
                # Save configuration to file
                try:
                    with open("config.json", 'w') as f:
                        json.dump(CONFIG, f, indent=2)
                    print("Configuration saved to config.json")
                except Exception as e:
                    print(f"Error saving configuration: {e}")
                break
            
            # Handle numeric input
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(CONFIG):
                    key = list(CONFIG.keys())[idx]
                else:
                    print("Invalid number. Please try again.")
                    continue
            else:
                key = choice
            
            # Check if key exists
            if key not in CONFIG:
                print(f"Key '{key}' not found in configuration")
                continue
            
            # Get current value type
            current_value = CONFIG[key]
            value_type = type(current_value)
            
            # Prompt for new value
            print(f"Current value of '{key}' is: {current_value} (type: {value_type.__name__})")
            new_value_str = input(f"Enter new value for '{key}': ")
            
            # Convert input to appropriate type
            try:
                if value_type == bool:
                    new_value = new_value_str.lower() in ["true", "yes", "y", "1"]
                elif value_type == int:
                    new_value = int(new_value_str)
                elif value_type == float:
                    new_value = float(new_value_str)
                else:
                    # String or other types
                    new_value = new_value_str
                
                # Update config
                CONFIG[key] = new_value
                print(f"Updated '{key}' to: {new_value}")
                
                # Special handling for certain settings
                if key == "rebuild_db" and new_value:
                    print("Note: Vector database will be rebuilt on next query")
                
            except ValueError:
                print(f"Error: Could not convert '{new_value_str}' to {value_type.__name__}")

# Helper function for API server
def create_coordinator():
    """Create and initialize the agent coordinator"""
    coordinator = AgentCoordinator()
    return coordinator

# Simple API server for web UI integration
def start_api_server(host="0.0.0.0", port=8000):
    """Start a FastAPI server for web UI integration"""
    try:
        from fastapi import FastAPI, HTTPException, Body
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        from pydantic import BaseModel
        
        class QueryRequest(BaseModel):
            query: str
            show_details: bool = False
        
        app = FastAPI(title="Multi-Agent System API")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize coordinator
        coordinator = create_coordinator()
        
        @app.get("/")
        def read_root():
            return {"status": "online", "message": "Multi-Agent System API is running"}
        
        @app.post("/query")
        def process_query(request: QueryRequest = Body(...)):
            query = request.query
            show_details = request.show_details
            
            try:
                # Process query
                results = coordinator.process_query(query)
                
                response = {
                    "query": query,
                    "coordinator_response": results["coordinator_response"],
                }
                
                if show_details:
                    response["agent_responses"] = results["agent_responses"]
                
                return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/config")
        def get_config():
            return CONFIG
        
        @app.post("/config")
        def update_config(new_config: dict = Body(...)):
            global CONFIG
            try:
                # Update only keys that exist in current config
                for key, value in new_config.items():
                    if key in CONFIG:
                        CONFIG[key] = value
                
                # Save to file
                with open("config.json", 'w') as f:
                    json.dump(CONFIG, f, indent=2)
                
                return {"status": "success", "config": CONFIG}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/rebuild_db")
        def rebuild_database():
            try:
                CONFIG["rebuild_db"] = True
                coordinator.vector_db = setup_vector_db(coordinator.embedding_function)
                return {"status": "success", "message": "Vector database rebuilt successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info(f"Starting API server on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("Error: FastAPI or uvicorn not installed. Please install with: pip install fastapi uvicorn")
        print("Error: FastAPI or uvicorn not installed. Please install with:")
        print("pip install fastapi uvicorn")
        return

# Main application - allows CLI or API server
def main():
    # Create required directories
    ensure_directory(CONFIG["text_directory"])
    ensure_directory(CONFIG["vector_db_path"])
    ensure_directory(CONFIG["cache_directory"])
    
    # Check if API server mode is requested
    if "--api" in sys.argv:
        try:
            port_index = sys.argv.index("--port")
            port = int(sys.argv[port_index + 1])
        except (ValueError, IndexError):
            port = CONFIG["api_port"]
        
        start_api_server(port=port)
    else:
        # Create and start the agent coordinator
        coordinator = AgentCoordinator()
        
        # Start interactive chat
        coordinator.interactive_chat()

if __name__ == "__main__":
    main()