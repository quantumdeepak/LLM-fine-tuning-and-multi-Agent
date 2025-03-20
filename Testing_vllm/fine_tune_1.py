import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ===== CONFIGURATION VARIABLES =====
# Credentials
HF_TOKEN = os.getenv("HF_TOKEN")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "quantumcosmos")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "d9a3cda658561b245d673f16fc2c6b3f")

# Input/Output
INPUT_JSON = os.getenv("INPUT_JSON", "/mnt/DATA/Glucoma/LLM/Testing_vllm/cvpr_papers.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "fine_tuned_model")

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2b")  # Using gemma-2b by default

# Training parameters
EPOCHS = int(os.getenv("EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # Use batch size of 1
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.1"))
ACCUMULATION_STEPS = int(os.getenv("ACCUMULATION_STEPS", "8"))

# LoRA parameters
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Runtime configuration
SEED = int(os.getenv("SEED", "42"))
# ===================================

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenized_data = []
        
        # Pre-tokenize all data
        for text in texts:
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None  # Important: return lists, not tensors
            )
            self.tokenized_data.append(encodings)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        
        # Convert to tensors here, not earlier
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def prepare_training_data(json_file, validation_split):
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Extract text from documents
    texts = [doc["text"] for doc in documents]
    
    # Create training examples by chunking texts
    training_examples = []
    
    for text in texts:
        # Split text into chunks of appropriate length
        words = text.split()
        for i in range(0, len(words), MAX_SEQ_LENGTH // 2):
            chunk = ' '.join(words[i:i + MAX_SEQ_LENGTH])
            if len(chunk.split()) > 50:  # Only include chunks with at least 50 words
                training_examples.append(chunk)
    
    # Shuffle and split into train/val
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * (1 - validation_split))
    
    train_texts = training_examples[:split_idx]
    val_texts = training_examples[split_idx:]
    
    logger.info(f"Created {len(train_texts)} training examples and {len(val_texts)} validation examples")
    
    return train_texts, val_texts

def load_model_and_tokenizer():
    model_id = MODEL_ID
    logger.info(f"Loading Gemma model from Hugging Face: {model_id}")
    
    # Check if token is available
    if not HF_TOKEN:
        logger.warning("No Hugging Face token provided. This may cause access issues for some models.")
    
    # Load tokenizer
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise ValueError(f"Could not load tokenizer: {e}")
    
    # Load model
    logger.info(f"Loading model weights (this may take a while)...")
    try:
        # Load without device mapping
        logger.info("Loading model without device mapping")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
            trust_remote_code=True,
            # Explicitly disable parallelism
            device_map=None
        )
        
        # Move to first GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Moving model to GPU: {device}")
            model = model.to(device)
                
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Trying to load in float32 precision instead...")
        
        try:
            # Try without mixed precision
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=HF_TOKEN,
                trust_remote_code=True,
                device_map=None
            )
            
            # Move to first GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info(f"Moving model to GPU: {device}")
                model = model.to(device)
        except Exception as e2:
            logger.error(f"Error loading model in float32: {e2}")
            raise ValueError(f"Could not load model: {e2}")
    
    return model, tokenizer

def setup_lora_fine_tuning(model):
    logger.info(f"Setting up LoRA with rank {LORA_R}")
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    
    # Prepare model for LoRA fine-tuning
    model = get_peft_model(model, peft_config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")
    
    return model

def manual_training_loop(model, train_dataloader, optimizer, device, accumulation_steps=8):
    """Manual training loop that avoids Trainer and handles accumulation steps manually"""
    model.train()
    total_loss = 0
    step = 0
    
    progress_bar = tqdm(total=len(train_dataloader), desc="Training")
    
    # Zero the gradients at the beginning
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass and loss calculation
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update gradients after accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            step += 1
            progress_bar.update(accumulation_steps)
            progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
        
        total_loss += loss.item() * accumulation_steps
    
    # Close progress bar
    progress_bar.close()
    
    # Return average loss
    return total_loss / len(train_dataloader)

def train_model_manually(model, tokenizer, train_texts, val_texts):
    """Train the model with a manual training loop instead of using Trainer"""
    logger.info(f"Starting training with {len(train_texts)} examples for {EPOCHS} epochs")
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Prepare datasets
    train_dataset = SimpleDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SimpleDataset(val_texts, tokenizer, MAX_SEQ_LENGTH)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # No parallel loading
        pin_memory=False  # Disable pinned memory
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Prepare optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = manual_training_loop(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            accumulation_steps=ACCUMULATION_STEPS
        )
        
        logger.info(f"Train loss: {train_loss}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
    
    return model

def save_model(model, tokenizer, output_dir):
    # Create output directory if it doesn't exist
    save_dir = f"{output_dir}/gemma_lora_finetuned"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save LoRA adapter weights and tokenizer
    logger.info(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    logger.info(f"Model and tokenizer saved to {save_dir}")

def generate_sample_text(model, tokenizer, prompt, max_length=256):
    logger.info(f"Generating text from prompt: {prompt}")
    
    # Move model to evaluation mode
    model.eval()
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    logger.info("=== Gemma Fine-tuning Script ===")
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} with {gpu_mem:.2f} GB memory")
            except:
                logger.info(f"GPU {i}: Unable to get memory information")
    
    try:
        # Prepare training data
        train_texts, val_texts = prepare_training_data(INPUT_JSON, VALIDATION_SPLIT)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Set up LoRA fine-tuning
        model = setup_lora_fine_tuning(model)
        
        # Train model using manual training loop
        model = train_model_manually(model, tokenizer, train_texts, val_texts)
        
        # Save model
        save_model(model, tokenizer, OUTPUT_DIR)
        
        # Generate sample text
        sample_prompt = "What are the latest advances in computer vision research?"
        generated_text = generate_sample_text(model, tokenizer, sample_prompt)
        
        logger.info("\nGenerated Text Sample:")
        logger.info("---------------------")
        logger.info(f"Prompt: {sample_prompt}")
        logger.info(f"Generated: {generated_text}")
        
    except Exception as e:
        logger.error(f"\nError during model loading or training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n=== Fine-tuning complete ===")

if __name__ == "__main__":
    main()