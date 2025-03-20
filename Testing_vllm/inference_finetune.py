import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

# ===== CONFIGURATION VARIABLES =====
# Model paths
BASE_MODEL = "google/gemma-2b"  # Base model ID from Hugging Face
LORA_WEIGHTS = "fine_tuned_model/gemma_lora_finetuned"  # Path to your LoRA weights

# Generation parameters
MAX_LENGTH = 512  # Maximum length of generated text
TEMPERATURE = 0.7  # Temperature for text generation
TOP_P = 0.9  # Top-p sampling for text generation

# Device configuration
GPU_ID = 0  # Specify which GPU to use (0, 1, 2, etc.)
USE_CPU = False  # Set to True to force CPU usage regardless of GPU availability

# Optional HF token if needed for model access
HF_TOKEN = os.getenv("HF_TOKEN", None)
# ===================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_with_lora():
    logger.info(f"Loading base model: {BASE_MODEL}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Determine device
    if USE_CPU:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as specified in config")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{GPU_ID}")
            logger.info(f"Using GPU: {GPU_ID} ({torch.cuda.get_device_name(GPU_ID)})")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")
    
    # Load base model
    logger.info("Loading base model (this may take a while)...")
    
    try:
        # First try with bfloat16 precision
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if not USE_CPU else torch.float32,
            trust_remote_code=True,
            token=HF_TOKEN,
            # Turn off auto device mapping to have explicit control
            device_map=None
        )
        
        # Move model to the specified device
        model = model.to(device)
        logger.info(f"Model loaded and moved to {device}")
            
    except Exception as e:
        logger.warning(f"Failed to load with bfloat16: {e}")
        logger.info("Trying with float32 precision...")
        
        # Try with float32 precision
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=HF_TOKEN,
            device_map=None
        )
        
        # Move model to the specified device
        model = model.to(device)
        logger.info(f"Model loaded with float32 and moved to {device}")
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {LORA_WEIGHTS}")
    
    # Load the model with LoRA weights
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        is_trainable=False
    )
    
    # Ensure model is in evaluation mode
    model.eval()
    
    logger.info("Model loaded successfully!")
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, device):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Verify model is on the expected device
    for param in model.parameters():
        if param.device != device:
            logger.warning(f"Parameter on unexpected device: {param.device} vs expected {device}")
            # Move this parameter to the correct device
            param.data = param.data.to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    print("\n" + "="*50)
    print("Loading fine-tuned Gemma Model with LoRA adapters...")
    
    try:
        # Load model with LoRA weights
        model, tokenizer, device = load_model_with_lora()
        
        # Interactive generation loop
        print("\n" + "="*50)
        print("Fine-tuned Gemma Model with LoRA - Interactive Mode")
        print("Enter 'quit', 'exit', or 'bye' to terminate")
        print("="*50 + "\n")
        
        while True:
            # Get prompt from user
            prompt = input("\nEnter your prompt: ")
            
            # Check if user wants to quit
            if prompt.lower() in ["quit", "exit", "bye", "terminate"]:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            # Generate and display text
            try:
                print("\nGenerating response...\n")
                generated_text = generate_text(model, tokenizer, prompt, device)
                print("="*80)
                print(generated_text)
                print("="*80)
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                print(f"Error generating text: {e}")
                
                # Try to detect and fix device issues
                if "device" in str(e).lower():
                    print("Attempting to fix device placement issues...")
                    try:
                        # Move the entire model to the device again
                        model = model.to(device)
                        print("Model moved to device. Please try again.")
                    except Exception as move_error:
                        logger.error(f"Failed to fix device issues: {move_error}")
                        print(f"Failed to fix device issues: {move_error}")
    
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        print(f"Error during model loading: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()