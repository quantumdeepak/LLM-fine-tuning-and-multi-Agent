

# Create a Python script to deploy and test Gemma-3-4b-pt
import os
from vllm import LLM, SamplingParams

# Set your HuggingFace token here directly
# You need to replace this with your actual token
HF_TOKEN = "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"  
os.environ["HF_TOKEN"] = HF_TOKEN

def setup_model():
    """
    Setup the Gemma-3-4b-pt model with vLLM
    
    Returns:
        LLM: Initialized vLLM model
    """
    print("HuggingFace token set")
    
    # Initialize the model with vLLM
    print("Loading Gemma-3-4b-pt model...")
    model = LLM(
        model="google/gemma-3-4b-pt",
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        trust_remote_code=True,
        max_model_len=8192
    )
    
    print("Model loaded successfully!")
    return model

def generate_response(model, prompt, temperature=0.7, max_tokens=512):
    """
    Generate a response using the model
    
    Args:
        model: vLLM model
        prompt: Input text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated response
    """
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )
    
    # Generate response
    outputs = model.generate([prompt], sampling_params)
    
    # Extract generated text
    generated_text = outputs[0].outputs[0].text
    return generated_text

def test_model(model):
    """
    Test the model with a few sample prompts
    
    Args:
        model: vLLM model
    """
    test_prompts = [
        "Explain the concept of deep learning in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the ethical considerations in AI development?",
        "Complete this code snippet: def fibonacci(n):"
    ]
    
    print("\n" + "="*50)
    print("TESTING MODEL RESPONSES")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest #{i+1}: {prompt}")
        print("-"*50)
        response = generate_response(model, prompt)
        print(f"Response: {response}")
        print("-"*50)

def run_custom_test(model, prompt, temperature=0.7, max_tokens=512):
    """
    Run a test with a custom prompt
    
    Args:
        model: vLLM model
        prompt: Custom prompt to test
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    print("\n" + "="*50)
    print("CUSTOM PROMPT TEST")
    print("="*50)
    print(f"Prompt: {prompt}")
    print("-"*50)
    response = generate_response(model, prompt, temperature, max_tokens)
    print(f"Response: {response}")

# Main execution
if __name__ == "__main__":
    # Setup and load the model
    model = setup_model()
    
    # Run predefined tests
    test_model(model)
    
    # You can uncomment this section to test with a custom prompt
    custom_prompt = "Explain quantum computing in simple terms"
    run_custom_test(model, custom_prompt, temperature=0.8, max_tokens=1024)