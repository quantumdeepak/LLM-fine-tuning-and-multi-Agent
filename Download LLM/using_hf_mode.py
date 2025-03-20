import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Path to your downloaded model
model_path = "/mnt/DATA/Glucoma/LLM/Download LLM/huggingface_models/gemma-3-4b-pt/models--google--gemma-3-4b-pt"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatically use available GPUs
    torch_dtype=torch.float16  # Use half precision for efficiency
)

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

# Function to get responses from the model
def get_gemma_response(prompt, max_length=500):
    # Format the prompt according to Gemma's expected format
    formatted_prompt = f"{prompt}"
    
    # Generate text
    outputs = generator(
        formatted_prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1
    )
    
    # Extract the generated text
    generated_text = outputs[0]["generated_text"]
    
    # Remove the original prompt to get just the response
    response = generated_text[len(formatted_prompt):]
    
    return response

# Example usage
if __name__ == "__main__":
    prompt = "Explain the concept of neural networks in simple terms."
    print("\nPrompt:", prompt)
    response = get_gemma_response(prompt)
    print("\nGemma's response:", response)