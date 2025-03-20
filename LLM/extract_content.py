# extract_content.py - Extract text and images from PDF papers

import os
import re
import json
import fitz  # PyMuPDF
import shutil
from PIL import Image
import io
from tqdm import tqdm
from config import CVPR_PAPER_DIR, IMAGES_DIR, DATASET_PATH

def get_all_pdfs(directory):
    """Recursively fetch all PDF files from a given directory and its subdirectories."""
    if not os.path.exists(directory):
        print(f"ERROR: Directory '{directory}' does not exist!")
        return []
    
    pdf_files = []
    print(f"Scanning directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        pdf_count = sum(1 for f in files if f.lower().endswith('.pdf'))
        if pdf_count > 0:
            print(f"Found {pdf_count} PDFs in {root}")
        
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    
    return pdf_files

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading text from {pdf_path}: {e}")
    return text

def extract_images_from_pdf(pdf_path, output_dir, pdf_name):
    """Extract images from PDF and save them to disk with metadata."""
    images_data = []
    
    try:
        with fitz.open(pdf_path) as pdf:
            # Create a subfolder for this PDF's images
            pdf_images_dir = os.path.join(output_dir, pdf_name)
            os.makedirs(pdf_images_dir, exist_ok=True)
            
            for page_num, page in enumerate(pdf):
                # Get text from this page for context
                page_text = page.get_text("text")
                
                # Extract images
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Try to determine if this is a figure by checking nearby text
                    is_figure = False
                    figure_caption = ""
                    
                    # Simple heuristic to find figure captions
                    caption_match = re.search(r"Fig(?:ure)?\.?\s+\d+\.?\s*([^\n]+)", page_text)
                    if caption_match:
                        is_figure = True
                        figure_caption = caption_match.group(1).strip()
                    
                    # Skip very small images that are likely icons or decorations
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        width, height = img.size
                        if width < 100 or height < 100:
                            continue
                    except Exception:
                        continue
                    
                    # Save the image
                    img_filename = f"page{page_num+1}_img{img_index}.{image_ext}"
                    img_path = os.path.join(pdf_images_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Add image metadata to our list
                    images_data.append({
                        "image_path": img_path,
                        "page_number": page_num + 1,
                        "is_figure": is_figure,
                        "caption": figure_caption,
                        "width": width,
                        "height": height
                    })
    
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")
    
    return images_data

def clean_text(text):
    """Cleans extracted text by removing references and unnecessary newlines."""
    text = re.sub(r"(?i)references\s*\n(.|\n)*", "", text)  # Remove references section
    text = re.sub(r"\n{2,}", "\n", text)  # Remove excessive newlines
    return text.strip()

def extract_sections(text):
    """Try to extract major sections from the paper."""
    sections = {}
    
    # Common section titles in computer vision papers
    section_patterns = [
        (r"(?i)abstract\s*\n", "abstract"),
        (r"(?i)introduction\s*\n", "introduction"),
        (r"(?i)related work\s*\n", "related_work"),
        (r"(?i)methodology|method|approach\s*\n", "methodology"),
        (r"(?i)experiments|experimental results\s*\n", "experiments"),
        (r"(?i)results\s*\n", "results"),
        (r"(?i)discussion\s*\n", "discussion"),
        (r"(?i)conclusion\s*\n", "conclusion")
    ]
    
    # Find the positions of each section
    positions = []
    for pattern, section_name in section_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            positions.append((match.start(), section_name))
    
    # Sort positions by their occurrence in the text
    positions.sort()
    
    # Extract text between sections
    for i, (pos, section_name) in enumerate(positions):
        next_pos = positions[i+1][0] if i < len(positions) - 1 else len(text)
        section_text = text[pos:next_pos].strip()
        
        # Remove the section title from the beginning
        section_text = re.sub(r"^.*?\n", "", section_text, count=1)
        sections[section_name] = section_text
    
    return sections

def create_dummy_data():
    """Create some dummy data for testing when no PDFs are found."""
    print("No PDFs found. Creating dummy data for testing purposes...")
    
    # Create a dummy dataset with one entry
    dummy_dataset = [{
        "paper_id": "dummy_paper",
        "pdf_path": "dummy_path.pdf",
        "full_text": "This is a dummy paper for testing purposes. It contains a section on computer vision techniques.",
        "sections": {
            "abstract": "This paper presents a novel approach to object detection.",
            "introduction": "Computer vision has seen significant advances in recent years.",
            "methodology": "We propose a new architecture based on transformers."
        },
        "images": []
    }]
    
    # Create a dummy image if needed
    os.makedirs(IMAGES_DIR, exist_ok=True)
    dummy_img_dir = os.path.join(IMAGES_DIR, "dummy_paper")
    os.makedirs(dummy_img_dir, exist_ok=True)
    
    # Save dataset
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dummy_dataset, f, indent=4)
    
    print("Dummy dataset created.")
    return dummy_dataset

def main():
    """Extract text and images from all PDFs and create dataset."""
    print(f"Looking for PDFs in {CVPR_PAPER_DIR}...")
    pdf_files = get_all_pdfs(CVPR_PAPER_DIR)
    print(f"Found {len(pdf_files)} PDF files.")
    
    if not pdf_files:
        print("\nDEBUGGING HELP:")
        print("1. Please check that the PDF directory exists and contains PDF files.")
        print(f"2. Current PDF directory path: {os.path.abspath(CVPR_PAPER_DIR)}")
        print("3. Make sure the PDFs have a .pdf extension (case insensitive).")
        print("4. Try setting an absolute path with --pdf_dir /full/path/to/pdf/directory")
        print("\nCreating dummy data for testing...")
        return create_dummy_data()
    
    # Create output directory for images
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    dataset = []
    
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_name}")
        
        # Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        clean_paper_text = clean_text(raw_text)
        
        if not clean_paper_text:
            print(f"Warning: No usable text extracted from {pdf_path}")
            continue
        
        # Try to extract structured sections
        sections = extract_sections(clean_paper_text)
        
        # Extract images
        print(f"Extracting images from {pdf_name}...")
        images_data = extract_images_from_pdf(pdf_path, IMAGES_DIR, pdf_name)
        print(f"Extracted {len(images_data)} usable images")
        
        # Create dataset entry
        paper_data = {
            "paper_id": pdf_name,
            "pdf_path": pdf_path,
            "full_text": clean_paper_text,
            "sections": sections,
            "images": images_data
        }
        
        dataset.append(paper_data)
    
    # Save dataset
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Extracted content from {len(dataset)} papers.")
    print(f"Dataset saved to {DATASET_PATH}")
    
    return dataset

if __name__ == "__main__":
    main()