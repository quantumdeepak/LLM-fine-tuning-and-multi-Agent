### This is the code file to download the model from the hugging face

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# Set up authentication (replace with your HF token)
HF_TOKEN = "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"  # <<<< INSERT YOUR HF TOKEN
login(HF_TOKEN)

import os
import torch  # Add this import
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# Model repo on Hugging Face
model_name = "google/gemma-3-4b-pt"
save_directory = "huggingface_models/gemma-3-4b-pt"

# Ensure directory exists
os.makedirs(save_directory, exist_ok=True)

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download model
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
model.save_pretrained(save_directory)

print(f"Model and tokenizer downloaded and saved to: {save_directory}")