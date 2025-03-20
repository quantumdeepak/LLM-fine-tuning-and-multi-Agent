# config.py - Central configuration for the Gemma fine-tuning pipeline

import os

# Directories and file paths
CVPR_PAPER_DIR = "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2013"  # Directory with PDFs
IMAGES_DIR = "cvpr_images"  # Directory to store extracted images
OUTPUT_MODEL_DIR = "cvpr_finetuned_gemma"
DATASET_PATH = "cvpr_dataset.json"
LOGS_DIR = "./logs"
CHECKPOINT_DIR = "./cvpr_gemma_checkpoints"

# Model configuration
MODEL_NAME = "google/gemma-3-4b-pt"  # Image-text-to-text model

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]

# Test example
TEST_IMAGE_PATH = "test_image.jpg"
TEST_PROMPT = "Describe the key components of this computer vision architecture:"