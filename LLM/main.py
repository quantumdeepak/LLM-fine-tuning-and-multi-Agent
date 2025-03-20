# main.py - Main script to run the entire pipeline

import argparse
import os
import time
import sys
from config import CVPR_PAPER_DIR, OUTPUT_MODEL_DIR, IMAGES_DIR

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gemma fine-tuning pipeline for CVPR papers")
    
    parser.add_argument(
        "--steps", 
        nargs="+", 
        choices=["extract", "prepare", "setup", "train", "test", "all"],
        default=["all"],
        help="Steps to run: extract, prepare, setup, train, test, or all"
    )
    
    parser.add_argument(
        "--pdf_dir", 
        type=str, 
        default=CVPR_PAPER_DIR,
        help=f"Directory containing PDF papers (default: {CVPR_PAPER_DIR})"
    )
    
    parser.add_argument(
        "--images_dir", 
        type=str, 
        default=IMAGES_DIR,
        help=f"Directory to store extracted images (default: {IMAGES_DIR})"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=OUTPUT_MODEL_DIR,
        help=f"Output directory for fine-tuned model (default: {OUTPUT_MODEL_DIR})"
    )
    
    parser.add_argument(
        "--test_image",
        type=str,
        help="Path to test image for model evaluation"
    )
    
    parser.add_argument(
        "--create_dummy",
        action="store_true",
        help="Create dummy data if no PDFs are found"
    )
    
    return parser.parse_args()

def setup_directories(args):
    """Ensure all necessary directories exist."""
    dirs_to_check = [
        args.pdf_dir,
        args.images_dir,
        args.output_dir,
        './logs',
        './cvpr_gemma_checkpoints'
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def check_pdf_dir(pdf_dir):
    """Check if PDF directory exists and contains PDFs."""
    if not os.path.exists(pdf_dir):
        print(f"WARNING: PDF directory '{pdf_dir}' does not exist!")
        return False
    
    # Check for PDFs recursively
    has_pdfs = False
    for root, _, files in os.walk(pdf_dir):
        if any(f.lower().endswith('.pdf') for f in files):
            has_pdfs = True
            break
    
    if not has_pdfs:
        print(f"WARNING: No PDF files found in '{pdf_dir}' or subdirectories!")
        print("Please add PDF files to the directory.")
        return False
    
    return True

def main():
    """Run the specified steps of the pipeline."""
    args = parse_arguments()
    steps = args.steps
    
    # Setup directories
    setup_directories(args)
    
    # Update config with command line arguments
    if args.pdf_dir != CVPR_PAPER_DIR or args.images_dir != IMAGES_DIR or args.output_dir != OUTPUT_MODEL_DIR:
        print("Updating configuration...")
        import config
        config.CVPR_PAPER_DIR = args.pdf_dir
        config.IMAGES_DIR = args.images_dir
        config.OUTPUT_MODEL_DIR = args.output_dir
        
    if args.test_image:
        import config
        config.TEST_IMAGE_PATH = args.test_image
    
    # Check PDF directory
    pdf_dir_valid = check_pdf_dir(args.pdf_dir)
    
    # If no PDFs found and user didn't request dummy data, give options
    if not pdf_dir_valid and not args.create_dummy:
        print("\nNo PDF files found. Options:")
        print("1. Add PDF files to the specified directory")
        print("2. Run with --create_dummy flag to use dummy data for testing")
        print("3. Specify a different PDF directory with --pdf_dir")
        print("\nExiting.")
        sys.exit(1)
    
    start_time = time.time()
    
    # Run all steps or specific steps
    if "all" in steps or "extract" in steps:
        print("\n========== STEP 1: EXTRACTING CONTENT FROM PDFs ==========")
        from extract_content import main as extract_main
        extract_main()
    
    if "all" in steps or "prepare" in steps:
        print("\n========== STEP 2: PREPARING DATASET ==========")
        from prepare_data import prepare_data
        prepare_data()
    
    if "all" in steps or "setup" in steps:
        print("\n========== STEP 3: SETTING UP MODEL ==========")
        from setup_model import main as setup_main
        setup_main()
    
    if "all" in steps or "train" in steps:
        print("\n========== STEP 4: TRAINING MODEL ==========")
        from train_model import train
        train()
    
    if "all" in steps or "test" in steps:
        print("\n========== STEP 5: TESTING MODEL ==========")
        from test_model import main as test_main
        test_main()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nPipeline completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s! 🎉")
    print(f"Fine-tuned model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()