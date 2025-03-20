import os
import ollama

# Set third GPU (index 2) for running the model
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Run Gemma 12B
response = ollama.chat(model="gemma3:12b", messages=[{"role": "user", "content": "Hello, how are you?"}])

# Print response
print(response["message"]["content"])
