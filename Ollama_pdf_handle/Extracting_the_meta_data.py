import os
import json
import fitz  # PyMuPDF
import requests
import re
import hashlib
import time
from tqdm import tqdm
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_extraction.log"),
        logging.StreamHandler()
    ]
)

# Configuration parameters
CONFIG = {
    "pdf_directory": "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2015",  # Directory containing PDF files
    "text_directory": "path/to/extracted_text",  # Directory to store extracted text files
    "output_file": "cvpr_papers_database.json",  # Output JSON file
    "gpu_devices": ["2", "3"],  # GPU devices to use
    "extraction_timeout": 60,  # Timeout for API calls in seconds (increased)
    "max_retries": 5,  # Maximum number of retries
    "retry_backoff_base": 5,  # Base time for exponential backoff
    "max_workers_pdf": 8,  # Number of parallel workers for PDF extraction
    "max_workers_llm": 4,  # Number of parallel workers for LLM processing
    "use_direct_api": True,  # Whether to use direct API calls instead of ollama package
    "ollama_host": "http://127.0.0.1:11434",  # Ollama API host
    "models": ["gemma3:12b"],  # Models to use (will round-robin between them)
    "skip_existing_text_files": True,  # Skip processing PDFs that already have text files
    "skip_existing_database_entries": True,  # Skip processing papers already in database
    "max_api_failures": 10,  # Maximum consecutive API failures before pause
    "failure_pause_time": 300  # Seconds to pause after max_api_failures
}

# Set GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CONFIG["gpu_devices"])

def find_pdf_files(directory):
    """Find all PDF files in the given directory and its subdirectories."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def generate_paper_id(file_path):
    """Generate a unique ID for a paper based on its file path."""
    return hashlib.md5(file_path.encode()).hexdigest()

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file with column-aware layout."""
    try:
        doc = fitz.open(file_path)
        text_by_page = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            # Sort blocks by y-coordinate (top to bottom) first, then by x-coordinate (left to right)
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            page_text = "\n".join(block[4] for block in sorted_blocks)
            text_by_page.append(page_text)
        
        # Combine all pages with separators
        full_text = "\n\n".join(text_by_page)
        doc.close()
        return full_text
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return ""

def save_text_to_file(text, text_file_path):
    """Save extracted text to a file."""
    try:
        os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logging.error(f"Error saving text to {text_file_path}: {e}")
        return False

def get_text_file_path(pdf_path, text_dir):
    """Generate the path for the text file corresponding to a PDF."""
    paper_id = generate_paper_id(pdf_path)
    # Create a two-level directory structure to avoid too many files in one directory
    subdir = paper_id[:2]
    return os.path.join(text_dir, subdir, f"{paper_id}.txt")

def process_pdf_to_text(pdf_path, text_dir):
    """Extract text from a PDF and save it to a file."""
    text_file_path = get_text_file_path(pdf_path, text_dir)
    
    # Skip if the text file already exists
    if CONFIG["skip_existing_text_files"] and os.path.exists(text_file_path):
        return text_file_path
    
    text = extract_text_from_pdf(pdf_path)
    if text and save_text_to_file(text, text_file_path):
        return text_file_path
    return None

def direct_llm_query(prompt, model=None, consecutive_failures=0):
    """
    Query LLM directly using the Ollama API with improved error handling.
    """
    if model is None:
        model = random.choice(CONFIG["models"])
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an academic assistant that extracts information from research papers."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.1
        }
    }
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            # Calculate backoff time with exponential increase
            retry_delay = CONFIG["retry_backoff_base"] * (2 ** attempt)
            
            # If we've had too many consecutive failures, pause processing
            if consecutive_failures >= CONFIG["max_api_failures"]:
                logging.warning(f"Too many consecutive API failures ({consecutive_failures}). Pausing for {CONFIG['failure_pause_time']} seconds...")
                time.sleep(CONFIG["failure_pause_time"])
                consecutive_failures = 0
            
            response = requests.post(
                f"{CONFIG['ollama_host']}/api/chat",
                headers=headers,
                json=data,
                timeout=CONFIG["extraction_timeout"]
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"], 0  # Reset consecutive failures
            else:
                logging.warning(f"API returned status code {response.status_code} on attempt {attempt+1}. Response: {response.text[:200]}")
                time.sleep(retry_delay)
                consecutive_failures += 1
                
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout querying LLM (attempt {attempt+1}/{CONFIG['max_retries']})")
            time.sleep(retry_delay)
            consecutive_failures += 1
        except Exception as e:
            logging.warning(f"Error querying LLM (attempt {attempt+1}/{CONFIG['max_retries']}): {e}")
            time.sleep(retry_delay)
            consecutive_failures += 1
    
    return None, consecutive_failures  # Return None if all attempts fail, and the current failure count

def extract_paper_metadata(text, file_path, consecutive_failures=0):
    """Extract basic paper metadata (title and authors)."""
    header_text = text[:2000]
    
    prompt = f"""
Extract ONLY the title and authors from this academic paper.
Paper content starts here:

{header_text}

Format your response as follows (keep the --- markers):
---TITLE---
[Full paper title here]
---AUTHORS---
[List authors here, one per line]
"""
    
    response, consecutive_failures = direct_llm_query(prompt, consecutive_failures=consecutive_failures)
    metadata = {"title": "Unknown Title", "authors": []}
    
    if response:
        # Parse response using the markers
        title_match = re.search(r'---TITLE---\s*(.*?)(?:---AUTHORS---|$)', response, re.DOTALL)
        authors_match = re.search(r'---AUTHORS---\s*(.*?)(?:---|$)', response, re.DOTALL)
        
        if title_match:
            title = title_match.group(1).strip()
            if title and title != "[Full paper title here]":
                metadata["title"] = title
                
        if authors_match:
            authors_text = authors_match.group(1).strip()
            authors = [a.strip() for a in authors_text.split('\n') if a.strip() and a.strip() != "[List authors here, one per line]"]
            if authors:
                metadata["authors"] = authors
    
    # Add file path and ID
    metadata["id"] = generate_paper_id(file_path)
    metadata["file_path"] = file_path
    
    return metadata, consecutive_failures

def extract_abstract_and_topics(text, paper_info, consecutive_failures=0):
    """Extract abstract and topics."""
    abstract_section = text[:5000]
    
    prompt = f"""
Extract the abstract and main topics from this paper.

Paper content (first section):
{abstract_section}

Format your response as follows (keep the --- markers):
---ABSTRACT---
[Full abstract text here]
---TOPICS---
[List 3-5 main research topics/keywords, one per line]
"""
    
    response, consecutive_failures = direct_llm_query(prompt, consecutive_failures=consecutive_failures)
    
    if response:
        # Parse response using the markers
        abstract_match = re.search(r'---ABSTRACT---\s*(.*?)(?:---TOPICS---|$)', response, re.DOTALL)
        topics_match = re.search(r'---TOPICS---\s*(.*?)(?:---|$)', response, re.DOTALL)
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            if abstract and abstract != "[Full abstract text here]":
                paper_info["abstract"] = abstract
        
        if topics_match:
            topics_text = topics_match.group(1).strip()
            topics = [t.strip() for t in topics_text.split('\n') if t.strip() and t.strip() != "[List 3-5 main research topics/keywords, one per line]"]
            if topics:
                paper_info["topics"] = topics
    
    # Ensure abstract field exists
    if "abstract" not in paper_info:
        paper_info["abstract"] = "Abstract not found"
    
    # Ensure topics field exists
    if "topics" not in paper_info:
        paper_info["topics"] = []
    
    return paper_info, consecutive_failures

def extract_references(text, paper_info, consecutive_failures=0):
    """Extract references."""
    # Look for the references section (typically at the end)
    references_section = None
    
    # Try to find the references section
    ref_patterns = [
        r'(?i)(?:references|bibliography)\s*\n([\s\S]+)',
        r'(?i)(?:references|bibliography)\s*([\s\S]+)'
    ]
    
    for pattern in ref_patterns:
        ref_match = re.search(pattern, text[-20000:])  # Look in the last 20000 characters
        if ref_match:
            references_section = ref_match.group(1)
            break
    
    if not references_section:
        paper_info["references"] = []
        return paper_info, consecutive_failures
    
    # Limit the size to avoid overwhelming the model
    references_section = references_section[:10000]
    
    prompt = f"""
Extract up to 10 most important references from this paper's references section.

References section:
{references_section}

Format your response as follows (keep the --- markers):
---REFERENCES---
1. [First reference: include author names, title, publication venue and year]
2. [Second reference]
... and so on
"""
    
    response, consecutive_failures = direct_llm_query(prompt, consecutive_failures=consecutive_failures)
    references = []
    
    if response:
        refs_match = re.search(r'---REFERENCES---\s*([\s\S]+)', response, re.DOTALL)
        if refs_match:
            refs_text = refs_match.group(1).strip()
            ref_lines = [r.strip() for r in refs_text.split('\n') if r.strip() and not r.strip().startswith("[")]
            
            for ref in ref_lines:
                # Remove numbering if present
                cleaned_ref = re.sub(r'^\d+\.\s*', '', ref).strip()
                if cleaned_ref:
                    references.append({"citation": cleaned_ref})
    
    paper_info["references"] = references
    return paper_info, consecutive_failures

def extract_affiliations(text, paper_info, consecutive_failures=0):
    """Extract author affiliations and details."""
    # Focus on the beginning of the paper where affiliations are typically found
    header_text = text[:3000]
    
    # Create a prompt that includes the authors we've already identified
    authors_list = "\n".join(paper_info.get("authors", []))
    
    prompt = f"""
For the following authors of the paper "{paper_info.get('title', 'Unknown Title')}", extract their affiliations and email addresses if available.

Authors:
{authors_list}

Paper header section:
{header_text}

Format your response as follows (keep the --- markers):
---AUTHOR_DETAILS---
Author: [Author Name 1]
Affiliation: [University/Organization]
Email: [Email if available]

Author: [Author Name 2]
Affiliation: [University/Organization]
Email: [Email if available]
... and so on for each author
"""
    
    response, consecutive_failures = direct_llm_query(prompt, consecutive_failures=consecutive_failures)
    author_details = []
    
    if response:
        details_match = re.search(r'---AUTHOR_DETAILS---\s*([\s\S]+)', response, re.DOTALL)
        if details_match:
            details_text = details_match.group(1).strip()
            author_blocks = re.split(r'\n\s*\n', details_text)
            
            for block in author_blocks:
                if not block.strip():
                    continue
                    
                author_info = {}
                
                author_match = re.search(r'Author:\s*(.*?)(?:\n|$)', block)
                if author_match:
                    author_info["name"] = author_match.group(1).strip()
                
                affiliation_match = re.search(r'Affiliation:\s*(.*?)(?:\n|$)', block)
                if affiliation_match:
                    author_info["affiliation"] = affiliation_match.group(1).strip()
                    
                email_match = re.search(r'Email:\s*(.*?)(?:\n|$)', block)
                if email_match:
                    email = email_match.group(1).strip()
                    if email and email.lower() != "[email if available]":
                        author_info["email"] = email
                
                if author_info.get("name"):
                    author_details.append(author_info)
    
    paper_info["author_details"] = author_details if author_details else [{"name": author, "affiliation": ""} for author in paper_info.get("authors", [])]
    return paper_info, consecutive_failures

def load_existing_database(output_file):
    """Load an existing database if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading existing database: {e}")
            # Try loading from backup if main file is corrupted
            backup_file = output_file + ".backup"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    pass
    return []

def save_paper_to_database(paper_info, output_file):
    """Save or update a single paper in the database."""
    try:
        # Load the current database
        database = load_existing_database(output_file)
        
        # Find if this paper already exists in the database
        paper_id = paper_info["id"]
        existing_index = next((i for i, p in enumerate(database) if p.get("id") == paper_id), None)
        
        # Update or append
        if existing_index is not None:
            database[existing_index] = paper_info
        else:
            database.append(paper_info)
        
        # Save the database
        # First save to a temporary file, then rename for atomicity
        temp_file = output_file + f".temp.{random.randint(1000, 9999)}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2)
        
        # Create a backup of the current file if it exists
        if os.path.exists(output_file):
            backup_file = output_file + ".backup"
            try:
                with open(output_file, 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dest:
                        dest.write(src.read())
            except:
                pass
        
        # Rename the temp file to the actual output file
        os.rename(temp_file, output_file)
        
        logging.info(f"Successfully saved/updated paper {paper_id} in database")
        return True
    
    except Exception as e:
        logging.error(f"Error saving paper to database: {e}")
        return False

def extract_text_from_pdfs(pdf_files, text_directory):
    """First phase: Extract text from all PDFs and save to files."""
    logging.info(f"Starting text extraction from {len(pdf_files)} PDF files")
    
    # Create the text directory if it doesn't exist
    os.makedirs(text_directory, exist_ok=True)
    
    successful = 0
    text_file_paths = []
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers_pdf"]) as executor:
        future_to_file = {
            executor.submit(process_pdf_to_text, pdf_path, text_directory): pdf_path 
            for pdf_path in pdf_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(pdf_files), desc="Extracting text from PDFs"):
            pdf_path = future_to_file[future]
            try:
                text_file_path = future.result()
                if text_file_path:
                    text_file_paths.append((pdf_path, text_file_path))
                    successful += 1
            except Exception as e:
                logging.error(f"Error extracting text from {pdf_path}: {e}")
    
    logging.info(f"Successfully extracted text from {successful} out of {len(pdf_files)} PDFs")
    return text_file_paths

def process_text_files(text_file_paths, output_file):
    """Second phase: Process the extracted text files with LLM."""
    logging.info(f"Starting LLM processing of {len(text_file_paths)} text files")
    
    # Filter out already processed papers if requested
    if CONFIG["skip_existing_database_entries"]:
        existing_database = load_existing_database(output_file)
        processed_ids = {paper.get("id") for paper in existing_database if paper.get("id")}
        
        new_text_file_paths = []
        for pdf_path, text_file_path in text_file_paths:
            paper_id = generate_paper_id(pdf_path)
            if paper_id not in processed_ids:
                new_text_file_paths.append((pdf_path, text_file_path))
        
        logging.info(f"Found {len(new_text_file_paths)} new papers to process (skipping {len(text_file_paths) - len(new_text_file_paths)} already processed)")
        text_file_paths = new_text_file_paths
    
    # Function to process a single text file
    def process_text_file(item):
        pdf_path, text_file_path = item
        consecutive_failures = 0
        
        try:
            logging.info(f"Processing paper: {pdf_path}")
            
            # Read the text file
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text:
                logging.warning(f"Empty text file for {pdf_path}")
                return None
            
            # Step 1: Extract basic metadata
            paper_info, consecutive_failures = extract_paper_metadata(text, pdf_path, consecutive_failures)
            
            # Step 2: Extract abstract and topics
            paper_info, consecutive_failures = extract_abstract_and_topics(text, paper_info, consecutive_failures)
            
            # Step 3: Extract references
            paper_info, consecutive_failures = extract_references(text, paper_info, consecutive_failures)
            
            # Step 4: Extract author affiliations
            paper_info, consecutive_failures = extract_affiliations(text, paper_info, consecutive_failures)
            
            return paper_info
            
        except Exception as e:
            logging.error(f"Error processing text file for {pdf_path}: {e}")
            logging.debug(traceback.format_exc())
            
            # Return basic information in case of failure
            return {
                "id": generate_paper_id(pdf_path),
                "file_path": pdf_path,
                "title": os.path.basename(pdf_path).replace("_", " ").replace(".pdf", ""),
                "authors": [],
                "abstract": f"Error processing: {str(e)}",
                "topics": [],
                "references": [],
                "error": str(e)
            }
    
    # Process text files in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers_llm"]) as executor:
        future_to_file = {
            executor.submit(process_text_file, item): item[0]  # pdf_path as key
            for item in text_file_paths
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(text_file_paths), desc="Processing papers with LLM"):
            pdf_path = future_to_file[future]
            try:
                paper_info = future.result()
                if paper_info:
                    # Save each paper to the database immediately after processing
                    if save_paper_to_database(paper_info, output_file):
                        successful += 1
            except Exception as e:
                logging.error(f"Error processing result for {pdf_path}: {e}")
    
    logging.info(f"Successfully processed {successful} out of {len(text_file_paths)} papers with LLM")

def main():
    """Main function to orchestrate the two-phase workflow."""
    logging.info("Starting two-phase paper extraction process")
    
    # Make sure all necessary directories exist
    os.makedirs(CONFIG["pdf_directory"], exist_ok=True)
    os.makedirs(CONFIG["text_directory"], exist_ok=True)
    
    # Phase 1: Find all PDF files and extract text
    pdf_files = find_pdf_files(CONFIG["pdf_directory"])
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Extract text from PDFs
    text_file_paths = extract_text_from_pdfs(pdf_files, CONFIG["text_directory"])
    
    # Phase 2: Process the text files with LLM
    process_text_files(text_file_paths, CONFIG["output_file"])
    
    logging.info("Paper extraction process completed")

if __name__ == "__main__":
    main()