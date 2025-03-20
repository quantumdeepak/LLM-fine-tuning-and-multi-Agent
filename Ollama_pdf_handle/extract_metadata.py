import os
import json
import requests
import re
import time
from tqdm import tqdm
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metadata_extraction.log"),
        logging.StreamHandler()
    ]
)

# Configuration parameters
CONFIG = {
    "text_directory": "/mnt/DATA/Glucoma/LLM/Ollama_pdf_handle/Text_pdf",  # Directory containing extracted text files
    "output_file": "cvpr_papers_database.json",  # Output JSON file
    "gpu_device": "2",  # Only use GPU 2
    "extraction_timeout": 60,  # Timeout for API calls in seconds
    "max_retries": 5,  # Maximum number of retries
    "retry_backoff_base": 5,  # Base time for exponential backoff
    "max_workers": 4,  # Number of parallel workers for LLM processing
    "ollama_host": "http://127.0.0.1:11434",  # Ollama API host
    "models": ["gemma3:12b"],  # Models to use
    "skip_existing_database_entries": True,  # Skip processing papers already in database
    "max_api_failures": 10,  # Maximum consecutive API failures before pause
    "failure_pause_time": 300,  # Seconds to pause after max_api_failures
    "processing_steps": ["metadata", "abstract", "references", "affiliations"]  # Processing steps
}

# Set GPU device to only use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_device"]

def find_text_files(directory):
    """Find all text files in the given directory and its subdirectories."""
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                text_files.append(os.path.join(root, file))
    return text_files

def get_pdf_path_from_text_path(text_path):
    """Derive the original PDF path from the text file path."""
    rel_path = os.path.relpath(text_path, CONFIG["text_directory"])
    pdf_path = os.path.join(os.path.dirname(CONFIG["text_directory"]), "cvpr_papers", 
                           os.path.splitext(rel_path)[0] + '.pdf')
    return pdf_path

def direct_llm_query(prompt, model=None, consecutive_failures=0):
    """Query LLM directly using the Ollama API with improved error handling."""
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
    """Step 1: Extract basic paper metadata (title and authors)."""
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
    
    # Add file path
    metadata["file_path"] = file_path
    # Generate a unique ID based on file name
    metadata["id"] = os.path.splitext(os.path.basename(file_path))[0]
    
    return metadata, consecutive_failures

def extract_abstract_and_topics(text, paper_info, consecutive_failures=0):
    """Step 2: Extract abstract and topics."""
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
    """Step 3: Extract references."""
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
    """Step 4: Extract author affiliations and details."""
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

def process_text_file_stepwise(text_file_path, steps):
    """Process a single text file in a stepwise manner."""
    pdf_path = get_pdf_path_from_text_path(text_file_path)
    consecutive_failures = 0
    paper_info = {"id": os.path.splitext(os.path.basename(text_file_path))[0]}
    
    try:
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text:
            logging.warning(f"Empty text file: {text_file_path}")
            return None
        
        # Process each step sequentially
        for step in steps:
            logging.info(f"Processing step '{step}' for {text_file_path}")
            
            if step == "metadata":
                paper_info, consecutive_failures = extract_paper_metadata(text, pdf_path, consecutive_failures)
                logging.info(f"Extracted metadata for {text_file_path}: {paper_info['title']}")
                
            elif step == "abstract":
                paper_info, consecutive_failures = extract_abstract_and_topics(text, paper_info, consecutive_failures)
                logging.info(f"Extracted abstract and topics for {text_file_path}")
                
            elif step == "references":
                paper_info, consecutive_failures = extract_references(text, paper_info, consecutive_failures)
                logging.info(f"Extracted {len(paper_info.get('references', []))} references for {text_file_path}")
                
            elif step == "affiliations":
                paper_info, consecutive_failures = extract_affiliations(text, paper_info, consecutive_failures)
                logging.info(f"Extracted affiliations for {text_file_path}")
            
            # Save after each step to ensure partial progress is preserved
            save_paper_to_database(paper_info, CONFIG["output_file"])
        
        return paper_info
        
    except Exception as e:
        logging.error(f"Error processing {text_file_path}: {e}")
        logging.debug(traceback.format_exc())
        
        # Create a minimal paper info object for error case
        error_info = {
            "id": os.path.splitext(os.path.basename(text_file_path))[0],
            "file_path": pdf_path,
            "title": os.path.basename(text_file_path).replace("_", " ").replace(".txt", ""),
            "authors": [],
            "abstract": f"Error processing: {str(e)}",
            "topics": [],
            "references": [],
            "error": str(e)
        }
        
        # Still save the error info
        save_paper_to_database(error_info, CONFIG["output_file"])
        return error_info

def process_text_files(text_files, steps):
    """Process multiple text files with stepwise extraction."""
    logging.info(f"Starting stepwise LLM processing of {len(text_files)} text files using GPU {CONFIG['gpu_device']}")
    
    # Filter out already processed papers if requested
    if CONFIG["skip_existing_database_entries"]:
        existing_database = load_existing_database(CONFIG["output_file"])
        processed_ids = {paper.get("id") for paper in existing_database 
                         if paper.get("id") and all(step_key in paper for step_key in 
                                                   ["title", "abstract", "references", "author_details"])}
        
        new_text_files = []
        for text_file in text_files:
            file_id = os.path.splitext(os.path.basename(text_file))[0]
            if file_id not in processed_ids:
                new_text_files.append(text_file)
        
        skipped = len(text_files) - len(new_text_files)
        logging.info(f"Found {len(new_text_files)} new papers to process (skipping {skipped} already processed)")
        text_files = new_text_files
    
    # Process text files in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_file = {
            executor.submit(process_text_file_stepwise, text_file, steps): text_file
            for text_file in text_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(text_files), desc="Processing papers with LLM"):
            text_file = future_to_file[future]
            try:
                paper_info = future.result()
                if paper_info and "error" not in paper_info:
                    successful += 1
            except Exception as e:
                logging.error(f"Error in thread for {text_file}: {e}")
    
    logging.info(f"Successfully processed {successful} out of {len(text_files)} papers with LLM")

def main():
    """Main function to orchestrate the metadata extraction process."""
    logging.info("Starting metadata extraction process")
    
    # Make sure the text directory exists
    if not os.path.exists(CONFIG["text_directory"]):
        logging.error(f"Text directory does not exist: {CONFIG['text_directory']}")
        return
    
    # Find all text files in the text directory
    text_files = find_text_files(CONFIG["text_directory"])
    
    if not text_files:
        logging.error("No text files found. Run the PDF extraction script first.")
        return
    
    logging.info(f"Found {len(text_files)} text files to process")
    
    # Process the text files in a stepwise manner
    process_text_files(text_files, CONFIG["processing_steps"])
    
    logging.info("Metadata extraction process completed")

if __name__ == "__main__":
    main()