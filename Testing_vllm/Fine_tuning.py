import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

# ===== CONFIGURATION VARIABLES =====
# Input/Output
INPUT_JSON = "/mnt/DATA/Glucoma/LLM/Testing_vllm/cvpr_papers.json"  # JSON file with extracted text
OUTPUT_DIR = "fine_tuned_model"  # Directory to save fine-tuned model
MODEL_NAME = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1"  # Path to Gemma 3 model

# Training parameters
EPOCHS = 1  # Number of training epochs
BATCH_SIZE = 2  # Batch size for training
LEARNING_RATE = 2e-4  # Learning rate
MAX_SEQ_LENGTH = 512  # Maximum sequence length
VALIDATION_SPLIT = 0.1  # Fraction of data to use for validation
GRADIENT_ACCUMULATION_STEPS = 4  # Number of steps to accumulate gradients
WARMUP_STEPS = 100  # Number of warmup steps for learning rate scheduler
FP16 = True  # Use mixed precision training

# LoRA parameters
LORA_R = 8  # LoRA attention dimension
LORA_ALPHA = 16  # LoRA alpha parameter
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA to

# Runtime configuration
SEED = 42  # Random seed for reproducibility
# ===================================

def setup_kaggle_credentials():
    """Set up Kaggle credentials from environment variables"""
    if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
        print("Kaggle credentials not found in environment variables.")
        username = "quantumcosmos"
        key = "d9a3cda658561b245d673f16fc2c6b3f"
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
    else:
        print("Using configured Kaggle credentials")

class CVPRDataset(Dataset):
    """Dataset for CVPR papers"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # For causal language modeling, labels are the same as input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def prepare_training_data(json_file, validation_split):
    """
    Prepare training data from JSON file for unsupervised learning.
    
    Args:
        json_file: Path to JSON file with extracted text
        validation_split: Fraction of data to use for validation
        
    Returns:
        tuple: (train_texts, val_texts)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    
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
    
    print(f"Created {len(train_texts)} training examples and {len(val_texts)} validation examples")
    
    return train_texts, val_texts

def load_model_and_tokenizer():
    """
    Load the Gemma 3 model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading Gemma 3 model from {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    return model, tokenizer

def setup_lora_fine_tuning(model):
    """
    Set up LoRA for fine-tuning.
    
    Args:
        model: The pre-trained model
        
    Returns:
        The model with LoRA adapters
    """
    print(f"Setting up LoRA with rank {LORA_R}")
    
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
    model.print_trainable_parameters()
    
    return model

def train_model(model, tokenizer, train_texts, val_texts):
    """
    Train the model using Hugging Face Trainer.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        train_texts: Training texts
        val_texts: Validation texts
        
    Returns:
        The fine-tuned model
    """
    print(f"Starting training with {len(train_texts)} examples for {EPOCHS} epochs")
    
    # Create datasets
    train_dataset = CVPRDataset(train_texts, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = CVPRDataset(val_texts, tokenizer, MAX_SEQ_LENGTH)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=FP16,
        warmup_steps=WARMUP_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    # Train model
    trainer.train()
    
    return model, trainer

def save_model(model, tokenizer, output_dir):
    """
    Save the fine-tuned model and tokenizer.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        output_dir: Directory to save the model
    """
    # Create output directory if it doesn't exist
    save_dir = f"{output_dir}/gemma3_4b_lora_finetuned"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save LoRA adapter weights and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model and tokenizer saved to {save_dir}")

def generate_sample_text(model, tokenizer, prompt, max_length=200):
    """
    Generate sample text from the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        
    Returns:
        str: Generated text
    """
    print(f"Generating text from prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    print("=== Gemma 3 4B Fine-tuning Script ===")
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Prepare training data
    train_texts, val_texts = prepare_training_data(INPUT_JSON, VALIDATION_SPLIT)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Set up LoRA fine-tuning
    model = setup_lora_fine_tuning(model)
    
    # Train model
    model, trainer = train_model(model, tokenizer, train_texts, val_texts)
    
    # Save model
    save_model(model, tokenizer, OUTPUT_DIR)
    
    # Generate sample text
    sample_prompt = "The latest advances in computer vision research include"
    generated_text = generate_sample_text(model, tokenizer, sample_prompt)
    
    print("\nGenerated Text Sample:")
    print("---------------------")
    print(f"Prompt: {sample_prompt}")
    print(f"Generated: {generated_text}")
    
    print("\n=== Fine-tuning complete ===")

if __name__ == "__main__":
    main()