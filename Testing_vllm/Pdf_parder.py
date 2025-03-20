import os
import re
import sys
from tqdm import tqdm
import json
import fitz  # PyMuPDF
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration variables
INPUT_DIR = "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR"  # Directory containing PDF files
OUTPUT_FILE = "cvpr_papers.json"  # Output JSON file path
MIN_LENGTH = 500  # Minimum text length to include
VERBOSE = True  # Print detailed information

def is_valid_pdf(file_path):
    """
    Check if a file is a valid PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a valid PDF, False otherwise
    """
    try:
        # Try to open the file as PDF
        doc = fitz.open(file_path)
        is_valid = doc.is_pdf
        doc.close()
        return is_valid
    except Exception as e:
        logger.warning(f"Invalid PDF {file_path}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get text with JSON formatting for better structure preservation
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n\n"
        
        # Close the document
        doc.close()
                    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
    
    return text

def clean_text(text):
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove strange characters and normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def find_all_pdfs(directory):
    """
    Find all PDF files in a directory and its subdirectories.
    
    Args:
        directory: Root directory to search
        
    Returns:
        list: List of paths to PDF files
    """
    pdf_paths = []
    
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return pdf_paths
    
    logger.info(f"Searching for PDFs in {directory} and subdirectories")
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Check if this is actually a directory
        if not os.path.isdir(root):
            logger.warning(f"Not a directory: {root}")
            continue
            
        if VERBOSE:
            logger.info(f"Checking directory: {root}")
        
        # Check each file in the directory
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file exists
            if not os.path.isfile(file_path):
                continue
                
            # Check if file has .pdf extension
            if file.lower().endswith('.pdf'):
                if VERBOSE:
                    logger.info(f"Found PDF: {file_path}")
                
                # Verify it's a valid PDF
                if is_valid_pdf(file_path):
                    pdf_paths.append(file_path)
                    if VERBOSE:
                        logger.info(f"Added valid PDF: {file_path}")
    
    return pdf_paths

def process_pdfs(directory, output_file, min_length):
    """
    Process all PDFs in a directory and its subdirectories.
    
    Args:
        directory: Directory containing PDFs
        output_file: Path to output JSON file
        min_length: Minimum text length to include
    """
    all_documents = []
    
    # Find all PDF files
    pdf_paths = find_all_pdfs(directory)
    
    if not pdf_paths:
        logger.error("No PDF files found.")
        return
    
    logger.info(f"Found {len(pdf_paths)} PDF files")
    
    # Process each PDF
    processed_count = 0
    failed_count = 0
    empty_count = 0
    short_count = 0
    
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            # Extract text
            raw_text = extract_text_from_pdf(pdf_path)
            
            if not raw_text:
                logger.warning(f"No text extracted from {pdf_path}")
                empty_count += 1
                continue
                
            # Clean the text
            clean_content = clean_text(raw_text)
            
            # Check if the content is long enough
            if len(clean_content) < min_length:
                short_count += 1
                if VERBOSE:
                    logger.info(f"Text too short ({len(clean_content)} chars) from {pdf_path}")
                continue
                
            # Add to documents
            document = {
                "source": pdf_path,
                "text": clean_content
            }
            all_documents.append(document)
            processed_count += 1
            
            if VERBOSE:
                logger.info(f"Successfully processed {pdf_path} - Text length: {len(clean_content)}")
                
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            failed_count += 1
    
    # Save to JSON file
    if all_documents:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(all_documents)} documents to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")
    else:
        logger.error("No documents to save")
    
    # Log statistics
    logger.info(f"PDF Processing Statistics:")
    logger.info(f"  Total PDFs found: {len(pdf_paths)}")
    logger.info(f"  Successfully processed: {processed_count}")
    logger.info(f"  Failed to process: {failed_count}")
    logger.info(f"  Empty (no text): {empty_count}")
    logger.info(f"  Too short: {short_count}")

def main():
    logger.info(f"Starting PDF text extraction from {INPUT_DIR}")
    process_pdfs(INPUT_DIR, OUTPUT_FILE, MIN_LENGTH)
    logger.info(f"Extraction complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()