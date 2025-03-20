from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil
import os

# Define paths
ollama_model_path = "/path/to/ollama/model.gguf"  # Replace with actual model path
hf_model_path = "./hf_model"

# Load model using LlamaCpp (if GGUF)
try:
    from llama_cpp import Llama
    llm = Llama(model_path=ollama_model_path)
    print("Model loaded successfully from Ollama!")
except ImportError:
    print("Install llama-cpp-python to load GGUF models.")

# Save in Hugging Face format
os.makedirs(hf_model_path, exist_ok=True)

# Example: Convert and save model (Adjust for your model type)
model = AutoModelForCausalLM.from_pretrained(ollama_model_path)
tokenizer = AutoTokenizer.from_pretrained(ollama_model_path)

model.save_pretrained(hf_model_path)
tokenizer.save_pretrained(hf_model_path)

print(f"Model saved in Hugging Face format at {hf_model_path}")
