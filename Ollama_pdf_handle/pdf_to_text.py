### Code to extract the text from PDF files using PyMuPDF (fitz) library
### The extracted text is saved to text files in a separate directory

import os
import fitz  # PyMuPDF
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_extraction.log"),
        logging.StreamHandler()
    ]
)

# Configuration parameters
CONFIG = {
    "pdf_directory": "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2015",  # Directory containing PDF files
    "text_directory": "/mnt/DATA/Glucoma/LLM/Ollama_pdf_handle/Text_pdf",  # Directory to store extracted text files
    "max_workers": 8,  # Number of parallel workers for PDF extraction
    "skip_existing_text_files": True,  # Skip processing PDFs that already have text files
}

def find_pdf_files(directory):
    """Find all PDF files in the given directory and its subdirectories."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file with column-aware layout, ignoring images."""
    try:
        doc = fitz.open(file_path)
        text_by_page = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Using get_text with "blocks" parameter to get text blocks
            blocks = page.get_text("blocks")
            # Filter blocks to exclude images (block[6] == 1 indicates image block in PyMuPDF)
            text_blocks = [block for block in blocks if block[6] == 0]
            # Sort blocks by y-coordinate (top to bottom) first, then by x-coordinate (left to right)
            sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
            page_text = "\n".join(block[4] for block in sorted_blocks)
            text_by_page.append(page_text)
        
        # Combine all pages with separators
        full_text = "\n\n".join(text_by_page)
        doc.close()
        return full_text
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return ""

def get_text_file_path(pdf_path, text_dir):
    """Generate the path for the text file corresponding to a PDF with the same name."""
    # Keep the same directory structure as the PDF files
    rel_path = os.path.relpath(pdf_path, CONFIG["pdf_directory"])
    # Replace .pdf extension with .txt
    text_path = os.path.join(text_dir, os.path.splitext(rel_path)[0] + '.txt')
    return text_path

def save_text_to_file(text, text_file_path):
    """Save extracted text to a file."""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logging.error(f"Error saving text to {text_file_path}: {e}")
        return False

def process_pdf_to_text(pdf_path, text_dir):
    """Extract text from a PDF and save it to a file."""
    try:
        text_file_path = get_text_file_path(pdf_path, text_dir)
        
        # Skip if the text file already exists
        if CONFIG["skip_existing_text_files"] and os.path.exists(text_file_path):
            logging.debug(f"Skipping {pdf_path} - text file already exists")
            return text_file_path, True  # Return path and a flag indicating it was skipped
        
        logging.debug(f"Extracting text from {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            logging.warning(f"No text extracted from {pdf_path}")
            return None, False
        
        if save_text_to_file(text, text_file_path):
            return text_file_path, False  # Return path and a flag indicating it wasn't skipped
        
        return None, False
    
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        logging.debug(traceback.format_exc())
        return None, False

def extract_text_from_pdfs(pdf_files, text_directory):
    """Extract text from all PDFs and save to files."""
    logging.info(f"Starting text extraction from {len(pdf_files)} PDF files")
    
    # Create the text directory if it doesn't exist
    os.makedirs(text_directory, exist_ok=True)
    
    # Track statistics
    successful = 0
    skipped = 0
    failed = 0
    text_file_paths = []
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_file = {
            executor.submit(process_pdf_to_text, pdf_path, text_directory): pdf_path 
            for pdf_path in pdf_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(pdf_files), desc="Extracting text from PDFs"):
            pdf_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    text_file_path, was_skipped = result
                    
                    if was_skipped:
                        skipped += 1
                    else:
                        successful += 1
                        
                    if text_file_path:
                        text_file_paths.append((pdf_path, text_file_path))
                else:
                    failed += 1
                    
            except Exception as e:
                logging.error(f"Error extracting text from {pdf_path}: {e}")
                failed += 1
    
    logging.info(f"Text extraction complete: {successful} successful, {skipped} skipped, {failed} failed")
    
    # Save the mapping of PDF files to text files for the next phase
    mapping_file = os.path.join(text_directory, "pdf_to_text_mapping.txt")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        for pdf_path, text_path in text_file_paths:
            f.write(f"{pdf_path}|{text_path}\n")
    
    logging.info(f"Saved PDF to text mapping to {mapping_file}")
    return text_file_paths

def main():
    """Main function to orchestrate the PDF to text extraction process."""
    logging.info("Starting PDF to text extraction process")
    
    # Make sure necessary directories exist
    os.makedirs(CONFIG["pdf_directory"], exist_ok=True)
    os.makedirs(CONFIG["text_directory"], exist_ok=True)
    
    # Find all PDF files
    pdf_files = find_pdf_files(CONFIG["pdf_directory"])
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Extract text from PDFs
    extract_text_from_pdfs(pdf_files, CONFIG["text_directory"])
    
    logging.info("PDF to text extraction process completed")

if __name__ == "__main__":
    main()