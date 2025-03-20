import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Set it using 'export HF_TOKEN=your_token_here'")

# Model name
model_name = "google/gemma-3-4b-pt"

# Load processor and model with authentication
processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(model_name, token=HF_TOKEN)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load an image (replace with your image path)
image_path = "example.jpg"  # Make sure this file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found!")

image = Image.open(image_path).convert("RGB")

# Process the image
inputs = processor(images=image, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode and print the output
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("\nGenerated Text:", generated_text)
