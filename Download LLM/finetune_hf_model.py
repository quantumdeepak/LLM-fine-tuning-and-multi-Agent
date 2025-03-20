
import os
import json
import fitz  # PyMuPDF for extracting text from PDFs
from tqdm import tqdm
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# ==========================
# CONFIGURATION PARAMETERS
# ==========================
HF_TOKEN = "your_huggingface_token_here"  # Replace with your HF token
MODEL_NAME = "huggingface_models/gemma-3-4b-pt"  # Path where the model is saved
PDF_DIR = "cvpr_papers/"  # Directory containing CVPR PDFs
OUTPUT_DIR = "processed_texts/"  # Where extracted text is saved
TRAINED_MODEL_DIR = "gemma_lora_finetuned"  # Directory to save fine-tuned model
LOG_FILE = "training_log.json"  # File to store logs
MAX_LENGTH = 512  # Max token length
BATCH_SIZE = 8  # Training batch size
EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 2e-4  # Learning rate for optimizer
LORA_R = 8  # Rank of LoRA adaptation
LORA_ALPHA = 32  # Scaling factor for LoRA
LORA_DROPOUT = 0.05  # Dropout rate for LoRA
MASK_PROB = 0.15  # Probability of masking tokens for MLM training

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# ==========================
# STEP 1: EXTRACT TEXT FROM CVPR PDFS
# ==========================
def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF."""
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

def process_pdfs(pdf_dir, output_dir):
    """Processes all PDFs in a directory and saves extracted text."""
    dataset = []
    for root, _, files in os.walk(pdf_dir):
        for file in tqdm(files, desc="Processing PDFs"):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                
                # Save extracted text
                text_filename = os.path.join(output_dir, file.replace(".pdf", ".txt"))
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(text)

                dataset.append({"text": text})

    # Save as JSONL for HF training
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for entry in dataset:
            jsonl_file.write(json.dumps(entry) + "\n")

    return jsonl_path

# Process PDFs and get dataset path
dataset_path = process_pdfs(PDF_DIR, OUTPUT_DIR)

# ==========================
# STEP 2: LOAD MODEL & TOKENIZER
# ==========================
login(HF_TOKEN)  # Authenticate with Hugging Face

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, device_map="auto")

# ==========================
# STEP 3: PREPARE DATASET FOR MLM TRAINING
# ==========================
dataset = load_dataset("json", data_files={"train": dataset_path})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Apply MLM data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MASK_PROB
)

# ==========================
# STEP 4: APPLY LORA ADAPTATION
# ==========================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="MLM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================
# STEP 5: TRAIN THE MODEL
# ==========================
training_args = TrainingArguments(
    output_dir=TRAINED_MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ==========================
# STEP 6: SAVE & LOG RESULTS
# ==========================
model.save_pretrained(TRAINED_MODEL_DIR)
tokenizer.save_pretrained(TRAINED_MODEL_DIR)

log_data = {
    "model_name": MODEL_NAME,
    "dataset_path": dataset_path,
    "trained_model_path": TRAINED_MODEL_DIR,
    "parameters": {
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "mask_probability": MASK_PROB
    }
}

with open(LOG_FILE, "w", encoding="utf-8") as log_file:
    json.dump(log_data, log_file, indent=4)

print(f"Training complete! Model saved to {TRAINED_MODEL_DIR}")
print(f"Training log saved to {LOG_FILE}")
