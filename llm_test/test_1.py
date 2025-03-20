import os
import fitz
import re
import json
import random
import torch
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# ------------------------- CONFIGURATION -------------------------

CVPR_PAPER_DIR = "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2013"  # Change to your PDF directory
OUTPUT_MODEL_DIR = "cvpr_finetuned_gemma_pt"
BATCH_SIZE = 4  # Reduced for memory efficiency
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Set up Hugging Face authentication
# It's best to use environment variables instead of hardcoding your token
HF_TOKEN ="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"

# ------------------------- STEP 1: EXTRACT TEXT FROM PDFs -------------------------

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def get_all_pdfs(directory):
    """Recursively fetch all PDF files from a given directory and its subdirectories."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def clean_text(text):
    """Cleans extracted text by removing references and unnecessary newlines."""
    text = re.sub(r"(?i)references\s*\n(.|\n)*", "", text)  # Remove references section
    text = re.sub(r"\n{2,}", "\n", text)  # Remove excessive newlines
    return text.strip()

pdf_files = get_all_pdfs(CVPR_PAPER_DIR)
dataset = []

for pdf_file in pdf_files:
    raw_text = extract_text_from_pdf(pdf_file)
    clean_paper_text = clean_text(raw_text)
    if clean_paper_text:
        dataset.append({"text": clean_paper_text})

print(f"Extracted {len(dataset)} papers.")

# ------------------------- STEP 2: DOWNLOAD GEMMA-PT MODEL & PROCESSOR -------------------------

MODEL_NAME = "google/gemma-3-4b-pt"  # Using the multimodal pre-trained Gemma model
print(f"Downloading model and processor for {MODEL_NAME}...")

# Load the processor with your access token
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

# Load the model with your access token
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    device_map="auto",  # Automatically use available devices
    torch_dtype=torch.bfloat16  # Use mixed precision to reduce memory usage
)

# ------------------------- STEP 3: PREPARE TRAINING DATA -------------------------

def prepare_text_only_examples(text):
    """Prepares text-only examples for Gemma-PT fine-tuning."""
    # Split into paragraphs
    paragraphs = text.split("\n\n")
    examples = []
    
    for paragraph in paragraphs:
        if len(paragraph.split()) > 10:  # Only use paragraphs with enough content
            # Create instruction-based examples
            examples.append({
                "text": paragraph.strip(),
                "instruction": "Summarize this paragraph from a computer vision research paper.",
            })
    
    return examples

# Apply text preparation
training_examples = []
for entry in dataset:
    training_examples.extend(prepare_text_only_examples(entry["text"]))

# Sample a manageable subset if dataset is too large
if len(training_examples) > 5000:
    random.shuffle(training_examples)
    training_examples = training_examples[:5000]

# Save dataset for future use
dataset_path = "cvpr_gemma_pt_dataset.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(training_examples, f, indent=4)
print(f"Training dataset with {len(training_examples)} examples saved to {dataset_path}")

# ------------------------- STEP 4: TOKENIZE DATA & CREATE DATASET -------------------------

def tokenize_function(examples):
    """Tokenize text-only examples for Gemma-PT."""
    batch_text = examples["text"]
    batch_instructions = examples["instruction"]
    
    inputs = processor(
        text=batch_instructions,
        text_target=batch_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    return inputs

# Load dataset from JSON
example_dataset = Dataset.from_list(training_examples)

# Tokenize dataset
tokenized_dataset = example_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=16,  # Process in smaller batches to avoid memory issues
    remove_columns=["text", "instruction"]
)

# Split into train and test datasets
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Prepared {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")

# ------------------------- STEP 5: APPLY LoRA FOR EFFICIENT TRAINING -------------------------

print("Applying LoRA for efficient fine-tuning...")
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common attention layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # For text-to-text generation
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------- STEP 6: TRAIN THE MODEL -------------------------

training_args = TrainingArguments(
    output_dir="./cvpr_gemma_pt_finetune",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir="./logs",
    save_total_limit=2,
    learning_rate=LEARNING_RATE,
    save_strategy="epoch",
    gradient_accumulation_steps=16,  # Increased for better memory efficiency
    fp16=True,  # Use mixed precision training
    remove_unused_columns=False,  # Important for custom datasets
    report_to="none",  # Disable Wandb reporting if not needed
    optim="adamw_torch",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor
)

print("Starting fine-tuning...")
trainer.train()

# ------------------------- STEP 7: SAVE THE FINE-TUNED MODEL -------------------------

model.save_pretrained(OUTPUT_MODEL_DIR)
processor.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Fine-tuned model saved to {OUTPUT_MODEL_DIR}")

# ------------------------- STEP 8: TEST THE FINE-TUNED MODEL -------------------------

print("Testing the fine-tuned model...")

# Basic test function for text-only inputs
def test_model(instruction, max_length=100):
    # Process the input
    inputs = processor(
        text=instruction,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move inputs to the appropriate device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the output
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)
    return generated_text[0]

# Test with a relevant instruction
test_instruction = "Explain the concept of convolutional neural networks in computer vision research."
generated_text = test_model(test_instruction)

print(f"Instruction: {test_instruction}")
print(f"Generated: {generated_text}")

# Try another example from our dataset
if training_examples:
    random_example = random.choice(training_examples)
    print("\nTesting with a sample from our dataset:")
    print(f"Instruction: {random_example['instruction']}")
    print(f"Original text: {random_example['text'][:100]}...")  # Show just the beginning
    
    generated = test_model(random_example['instruction'])
    print(f"Generated: {generated}")

print("Fine-tuning complete! 🚀")