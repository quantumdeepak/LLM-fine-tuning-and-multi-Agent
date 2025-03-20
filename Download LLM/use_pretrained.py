from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define model path (where you downloaded it)
model_path = "/mnt/DATA/Glucoma/LLM/Download LLM/huggingface_models/gemma-3-4b-pt"  # Adjust if needed

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Generate text
input_text = "What is artificial intelligence?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Move to GPU if available
outputs = model.generate(**inputs, max_length=100)

# Decode and print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
