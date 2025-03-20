# File: fine_tune_gemma_pymupdf.py
import os
import torch
import fitz  # PyMuPDF
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

# 1. PDF Processing with PyMuPDF
def extract_text_from_pdfs(directory):
    text_data = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                try:
                    doc = fitz.open(path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    text_data.append(text)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
    
    return {"text": text_data}

# 2. Prepare Dataset
pdf_directory = "/mnt/DATA/Glucoma/LLM/cvf_papers/CVPR/2013"
dataset = Dataset.from_dict(extract_text_from_pdfs(pdf_directory))

# 3. Initialize Model and Tokenizer
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 4. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Tokenization and Masking
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 6. Data Collator for Masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir="./gemma-cvpr-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    report_to="none"
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 9. Train and Save
trainer.train()
model.save_pretrained("./gemma-cvpr-lora-final")

# 10. Merge and Save Full Model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./gemma-cvpr-full")
tokenizer.save_pretrained("./gemma-cvpr-full")