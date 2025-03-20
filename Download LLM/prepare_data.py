import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm

# Define directories
input_dir = "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2013"  # Path where PDFs are stored
output_dir = "processed_texts/"  # Path to save processed files
os.makedirs(output_dir, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

def process_pdfs(input_dir, output_dir):
    """Extract and save text from all PDFs in the input directory and subdirectories."""
    dataset = []  # List to store processed data
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing PDFs"):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                
                # Save as a text file (optional)
                text_filename = os.path.join(output_dir, file.replace(".pdf", ".txt"))
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Save as JSONL format for Hugging Face
                dataset.append({"text": text})

    # Save JSONL dataset
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for entry in dataset:
            jsonl_file.write(json.dumps(entry) + "\n")
    
    print(f"Processed dataset saved at: {jsonl_path}")

# Run processing
process_pdfs(input_dir, output_dir)
